#!/usr/bin/env python3
"""Bounded, counterbalanced BF16 RMSNorm and FP32 vendor comparisons."""

from __future__ import annotations

import argparse
import fcntl
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common import capture, gpu_state, measure, provenance, summarize
from compare import calibrate, profile_torch

RMS_SHAPES = (
    (1, 4096),
    (16, 4096),
    (256, 4096),
    (4096, 4096),
    (16, 8192),
    (3, 513),
    (2, 12289),
)
RMS_BACKENDS = ("pytorch", "torch_compile", "cuda_fast", "triton_fast")
GEMV_BACKENDS = ("torch_compile", "triton_fast", "cublas", "cublaslt")


def make_rms_case(rows, hidden, seed, *, device="cuda", pattern="random", offset=0):
    import torch

    from crossdsl_kernels.references.rmsnorm import fused_residual_rmsnorm_reference

    generator = torch.Generator().manual_seed(seed)

    def rand(shape):
        count = 1
        for size in shape:
            count *= size
        storage = torch.randn(count + offset, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        return storage[offset:].view(shape)

    x, residual, weight = rand((rows, hidden)), rand((rows, hidden)), rand((hidden,))
    if pattern == "zero":
        x.zero_()
        residual.zero_()
    elif pattern == "cancel":
        residual.copy_(-x)
    elif pattern == "small":
        x.mul_(1e-4)
        residual.mul_(1e-4)
    elif pattern == "large":
        x.mul_(1e4)
        residual.mul_(1e4)
    elif pattern == "rounding":
        x.fill_(1.0)
        residual.fill_(1.0 / 256)
        weight.fill_(1.0)
        x.flatten()[0] = 2.0
    elif pattern != "random":
        raise ValueError(f"unknown pattern: {pattern}")
    args = (x, residual, weight, 1e-5)
    return {
        "args": args,
        "kwargs": {},
        "expected": fused_residual_rmsnorm_reference(*args),
        "logical_bytes": 2 * (4 * rows * hidden + hidden),
        "pattern": pattern,
        "offset": offset,
    }


def check_rms_output(output, case):
    import torch

    y, residual = output
    target, target_residual = case["expected"]
    finite = bool(torch.isfinite(y).all() and torch.isfinite(residual).all())
    dtype_ok = y.dtype == residual.dtype == torch.bfloat16
    shape_ok = y.shape == residual.shape == target.shape
    if not (finite and dtype_ok and shape_ok):
        return {
            "status": "FAIL",
            "finite": finite,
            "dtype_ok": dtype_ok,
            "shape_ok": shape_ok,
        }
    yf, tf = y.float(), target.float()
    difference = (yf - tf).abs()
    up = torch.nextafter(target, torch.full_like(target, float("inf"))).float()
    down = torch.nextafter(target, torch.full_like(target, -float("inf"))).float()
    # At a power-of-two boundary, use the larger neighboring BF16 spacing.
    spacing = torch.maximum((up - tf).abs(), (tf - down).abs())
    ulps = difference / spacing.clamp_min(torch.finfo(torch.float32).tiny)
    rel_l2 = float(
        torch.linalg.vector_norm(yf - tf)
        / torch.linalg.vector_norm(tf).clamp_min(1e-30)
    )
    max_ulp = float(ulps.max())
    residual_exact = bool(torch.equal(residual, target_residual))
    return {
        "status": "PASS"
        if max_ulp <= 1.0 and rel_l2 <= 5e-4 and residual_exact
        else "FAIL",
        "finite": finite,
        "dtype_ok": dtype_ok,
        "shape_ok": shape_ok,
        "max_abs_err": float(difference.max()),
        "max_neighbor_spacing_err": max_ulp,
        "rel_l2_err": rel_l2,
        "residual_exact": residual_exact,
        "tolerance": {
            "max_neighbor_spacing_err": 1.0,
            "rel_l2_err": 5e-4,
            "residual": "exact BF16 rounding of FP32 add",
        },
    }


def rms_backend(backend, case):
    from comparison_cases import prepare_backend

    return prepare_backend("rmsnorm", backend, case)


def counterbalanced_orders(backends, rounds, seed):
    rng = random.Random(seed)
    orders = []
    for index in range(rounds):
        if index % 2 == 0:
            order = list(backends)
            rng.shuffle(order)
        else:
            order = list(reversed(orders[-1]))
        orders.append(order)
    return orders


def run_suite(args):
    import torch
    from comparison_cases import check_output, make_case, prepare_backend

    result = provenance(torch, args.experiment)
    result.update(
        status="RUNNING",
        rows=[],
        seed=args.seed,
        rounds=args.rounds,
        warmup=args.warmup,
        graph_calls=args.graph_calls,
        suite=args.suite,
        diagnostic=bool(args.profile_iterations),
        telemetry=[],
        orders={},
        semantics="BF16 I/O, FP32 unrounded residual sum and statistics; out-of-place",
        timing_policy="Paired rounds: shuffled order then its reverse; event/wall and graph separate",
        allocation_policy="Public allocating APIs, including eager/compiled outputs; setup excluded",
    )
    if not torch.cuda.is_available():
        result.update(status="SKIP", reason="CUDA unavailable in this process")
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        return 2
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError(
            "This experiment is scoped to SM120; do not mix architecture evidence"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.suite == "rmsnorm":
        specs = [
            {"name": f"bf16_{r}x{h}", "rows": r, "hidden": h} for r, h in RMS_SHAPES
        ]
        defaults = RMS_BACKENDS
        check = check_rms_output
    else:
        catalog = json.loads((ROOT / "benchmarks/shapes/comparison.json").read_text())[
            "cases"
        ]
        specs = [
            s
            for s in catalog
            if s["name"] in {"hidden_m1", "hidden_m16", "mlp_up_m1", "mlp_up_m16"}
        ]
        defaults = GEMV_BACKENDS
        check = lambda output, case: check_output("gemv", output, case)
        result["semantics"] = (
            "FP32, strict IEEE custom dot and FP32 vendor inputs; KN layout"
        )
        result["vendor_limit"] = (
            "Existing cuBLAS dispatch; Lt first successful bounded heuristic, not exhaustive tuning"
        )
    if args.cases:
        wanted = set(args.cases.split(","))
        if wanted - {s["name"] for s in specs}:
            raise ValueError(f"unknown cases: {wanted - {s['name'] for s in specs}}")
        specs = [s for s in specs if s["name"] in wanted]
    backends = args.backends.split(",") if args.backends else list(defaults)
    if set(backends) - set(defaults) or len(set(backends)) != len(backends):
        raise ValueError("use distinct backends supported by the selected suite")
    for index, spec in enumerate(specs):
        case = (
            make_rms_case(spec["rows"], spec["hidden"], args.seed + index)
            if args.suite == "rmsnorm"
            else make_case(spec, args.seed + index)
        )
        runs, graphs, entries = {}, {}, {}
        inputs_before = tuple(
            t.clone() for t in case["args"] if isinstance(t, torch.Tensor)
        )
        for backend in backends:
            begin = time.perf_counter()
            run = (
                rms_backend(backend, case)
                if args.suite == "rmsnorm"
                else prepare_backend("gemv", backend, case)
            )
            output = run()
            torch.cuda.synchronize()
            row = {
                "case": spec["name"],
                "shape": spec,
                "backend": backend,
                "setup_and_first_call_ms": (time.perf_counter() - begin) * 1000,
                "correctness": check(output, case),
                "logical_bytes": case["logical_bytes"],
                "samples": {
                    "event": [],
                    "wall": [],
                    "graph_event": [],
                    "graph_wall": [],
                },
            }
            runs[backend], entries[backend] = run, row
            for _ in range(args.warmup):
                run()
            if args.graph_calls:
                graph, graph_output = capture(torch, run, args.graph_calls)
                tensors = (
                    graph_output
                    if isinstance(graph_output, (tuple, list))
                    else (graph_output,)
                )
                for tensor in tensors:
                    tensor.fill_(12345)
                graph.replay()
                torch.cuda.synchronize()
                row["graph_correctness"] = check(graph_output, case)
                graphs[backend] = graph
            if args.profile_iterations:
                row["torch_profile"] = profile_torch(
                    torch,
                    run,
                    args.output / f"{spec['name']}--{backend}",
                    f"{spec['name']}::{backend}",
                    args.profile_iterations,
                )
            result["rows"].append(row)
            print(
                f"prepared {spec['name']} {backend}: {row['correctness']['status']}",
                flush=True,
            )
        torch.cuda.synchronize()
        orders = counterbalanced_orders(backends, args.rounds, args.seed + index)
        result["orders"][spec["name"]] = orders
        for round_index, order in enumerate(orders):
            before = gpu_state()
            for backend in order:
                row = entries[backend]
                for mode in (
                    ("event", "graph_event") if args.graph_calls else ("event",)
                ):
                    run = runs[backend] if mode == "event" else graphs[backend].replay
                    timing = measure(torch, run, warmup=0, repeat=1)
                    divisor = 1 if mode == "event" else args.graph_calls
                    row["samples"][mode].append(
                        timing["cuda_event"]["median_ms"] / divisor
                    )
                    wall = "wall" if mode == "event" else "graph_wall"
                    row["samples"][wall].append(
                        timing["synchronized_wall"]["median_ms"] / divisor
                    )
            result["telemetry"].append(
                {
                    "case": spec["name"],
                    "round": round_index,
                    "before": before,
                    "after": gpu_state(),
                }
            )
        for row in entries.values():
            row["timing"] = {
                key: summarize(samples)
                for key, samples in row.pop("samples").items()
                if samples
            }
            row["logical_event_gbps"] = (
                row["logical_bytes"] / row["timing"]["event"]["median_ms"] / 1e6
            )
            row["status"] = (
                "PASS"
                if all(
                    row[key]["status"] == "PASS"
                    for key in ("correctness", "graph_correctness")
                    if key in row
                )
                else "FAIL"
            )
        result["inputs_unchanged"] = result.get("inputs_unchanged", True) and all(
            torch.equal(t, old)
            for t, old in zip(
                (t for t in case["args"] if isinstance(t, torch.Tensor)), inputs_before
            )
        )
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
    if args.calibrate:
        result["copy_calibration"] = calibrate(
            torch, args.warmup, args.rounds, args.graph_calls
        )
    result["gpu_after"] = gpu_state()
    result["status"] = (
        "PASS"
        if result["inputs_unchanged"]
        and all(row["status"] == "PASS" for row in result["rows"])
        else "FAIL"
    )
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(f"{result['status']}: {len(result['rows'])} backend-case rows", flush=True)
    return 0 if result["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("rmsnorm", "gemv"), default="rmsnorm")
    parser.add_argument("--cases", default="")
    parser.add_argument("--backends", default="")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--graph-calls", type=int, default=20)
    parser.add_argument("--profile-iterations", type=int, default=0)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    if (
        args.rounds < 2
        or args.rounds % 2
        or args.warmup < 1
        or args.graph_calls < 0
        or args.profile_iterations < 0
    ):
        parser.error(
            "use an even positive number of rounds, positive warmup and nonnegative graph/profile counts"
        )
    if args.output.exists():
        parser.error("choose a fresh output directory")
    args.output.mkdir(parents=True)
    with Path("/tmp/crossdsl-gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(run_suite(args))


if __name__ == "__main__":
    main()
