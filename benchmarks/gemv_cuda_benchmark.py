#!/usr/bin/env python3
"""Matched FP32 projection comparison; historical kernels remain unchanged."""

from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common import capture, gpu_state, provenance, summarize

CLEAN_ENTRY_POINTS = {
    "clean_warp": "decode_gemv_warp_clean",
    "clean_rows": "decode_gemv_rows_clean",
    "tiled": "decode_gemm_tiled",
}
BACKENDS = ("cuda", "cuda_fast", *CLEAN_ENTRY_POINTS, "cublas")
OPTIONAL_BACKENDS = ("cublaslt", "pytorch")
CLEAN_SOURCES = ("gemv_clean_ext.cpp", "gemv_clean.cu", "gemm_tiled.cu")


@functools.lru_cache(maxsize=1)
def clean_module():
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("building the clean CUDA extension requires a GPU-visible process")
    capability = torch.cuda.get_device_capability()
    arch = f"{capability[0]}.{capability[1]}"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)
    os.environ.setdefault("MAX_JOBS", "4")
    sources = [ROOT / "csrc/gemv" / name for name in CLEAN_SOURCES]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()[:12]
    name = re.sub(r"[^A-Za-z0-9_]", "_", f"crossdsl_gemv_clean_{torch.__version__}_{arch}_{digest}")
    directory = ROOT / "results/build/torch_extensions/gemv_clean" / f"sm{capability[0]}{capability[1]}" / name
    directory.mkdir(parents=True, exist_ok=True)
    return load(name=name, sources=[str(path) for path in sources], build_directory=str(directory),
                extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], verbose=False)


def projection_cases():
    catalog = json.loads((ROOT / "benchmarks/shapes/comparison.json").read_text())["cases"]
    wanted = {"hidden_m1", "hidden_m2", "hidden_m4", "hidden_m8", "hidden_m16", "mlp_up_m16",
              "hidden_nk_m1", "gemv_2x33x17_tail", "gemv_3x129x65_nk_tail"}
    cases = [spec for spec in catalog if spec["name"] in wanted]
    for layout in ("KN", "NK"):
        cases.append({"name": f"gemv_5x257x67_{layout.lower()}_tail", "op": "gemv",
                      "m": 5, "k": 257, "n": 67, "layout": layout, "bias": True})
    cases.append({"name": "hidden_nk_m16", "op": "gemv", "m": 16, "k": 4096,
                  "n": 4096, "layout": "NK", "bias": True})
    return cases


def select_cases(names):
    cases = projection_cases()
    requested = set(names.split(",")) if names else {spec["name"] for spec in cases}
    unknown = requested - {spec["name"] for spec in cases}
    if unknown:
        raise ValueError(f"unknown cases: {sorted(unknown)}")
    return [spec for spec in cases if spec["name"] in requested]


def prepare(backend, case):
    from comparison_cases import prepare_backend

    if backend in CLEAN_ENTRY_POINTS:
        return functools.partial(getattr(clean_module(), CLEAN_ENTRY_POINTS[backend]),
                                 *case["args"], case["kwargs"]["weight_layout"])
    if backend not in BACKENDS + OPTIONAL_BACKENDS:
        raise ValueError(f"unknown backend: {backend}")
    return prepare_backend("gemv", backend, case)


def independent_expected(case):
    x, weight, bias = case["args"]
    matrix = weight if case["kwargs"]["weight_layout"] == "KN" else weight.t()
    result = x.double() @ matrix.double()
    if bias is not None:
        result += bias.double()
    return (result.float(),)


def verify_changed_replay(torch, graph, output, case):
    from comparison_cases import check_output

    x, _, bias = case["args"]
    original_address = x.data_ptr()
    saved_x = x.clone()
    saved_bias = bias.clone() if bias is not None else None
    expected = case["expected"]
    try:
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        poisoned = check_output("gemv", output, case)
        x.mul_(0.5).add_(0.125)
        if bias is not None:
            bias.add_(0.25)
        case["expected"] = independent_expected(case)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        changed = check_output("gemv", output, case)
        return {"status": "PASS" if poisoned["status"] == changed["status"] == "PASS" else "FAIL",
                "poisoned": poisoned, "changed_input": changed,
                "input_address_unchanged": x.data_ptr() == original_address}
    finally:
        x.copy_(saved_x)
        if bias is not None:
            bias.copy_(saved_bias)
        case["expected"] = expected
        torch.cuda.synchronize()


def backend_orders(backends, repeat, seed):
    rng = random.Random(seed)
    orders = []
    for index in range(repeat):
        if index % 2 == 0:
            order = list(backends)
            rng.shuffle(order)
        else:
            order = list(reversed(orders[-1]))
        orders.append(order)
    return orders


def make_timer(torch):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Initialize event handles before any sample, then reuse the same pair.
    start.record()
    end.record()
    end.synchronize()

    def sample(run, calls=1):
        begin = time.perf_counter()
        start.record()
        run()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / calls, (time.perf_counter() - begin) * 1000 / calls

    return sample


def save(directory, result):
    (directory / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# CUDA Projection Comparison", "", f"Experiment: `{result['experiment_id']}`", "",
             "Graph-event microseconds/call; all modes and raw samples are retained in JSON.", "",
             "| Case | Backend | Status | p10 | Median | p90 |", "|---|---|---|---:|---:|---:|"]
    for row in result["rows"]:
        timing = row.get("timing", {}).get("graph_event", {})
        values = [f"{timing[key] * 1000:.3f}" if key in timing else "-"
                  for key in ("p10_ms", "median_ms", "p90_ms")]
        lines.append(f"| {row['case']} | {row['backend']} | {row['status']} | {' | '.join(values)} |")
    (directory / "results.md").write_text("\n".join(lines) + "\n")


def run_suite(args, specs, backends):
    import torch
    from comparison_cases import check_output, make_case

    result = provenance(torch, args.experiment)
    result.update(rows=[], status="RUNNING", seed=args.seed, warmup=args.warmup,
                  repeat=args.repeat, graph_calls=args.graph_calls, orders={}, telemetry=[],
                  timing_policy="Primed reused events; paired shuffled/reversed rounds; graph time normalized per call",
                  allocation_policy="Allocating public APIs; graph allocations captured in private pools; setup and vendor heuristics excluded from replay",
                  backend_mapping={"cuda": "historical decode_gemv_warp (not v0)",
                                   "cuda_fast": "historical decode_gemv_rows", **CLEAN_ENTRY_POINTS},
                  diagnostic=bool(args.diagnostic or args.profile_iterations))
    if not torch.cuda.is_available():
        result.update(status="SKIP", reason="CUDA unavailable to this process")
        save(args.output, result)
        return 2
    if torch.cuda.get_device_capability() != (12, 0):
        result.update(status="SKIP", reason="Measurement is scoped to SM120; no architecture substitution")
        save(args.output, result)
        return 2
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    sample = make_timer(torch)
    for index, spec in enumerate(specs):
        case = make_case(spec, args.seed + index)
        originals = [t.clone() for t in case["args"] if t is not None]
        entries, runs, graphs = {}, {}, {}
        order = list(backends)
        random.Random(args.seed + index).shuffle(order)
        for backend in order:
            row = {"case": spec["name"], "shape": spec, "backend": backend, "status": "ERROR"}
            result["rows"].append(row)
            try:
                begin = time.perf_counter()
                run = prepare(backend, case)
                output = run()
                torch.cuda.synchronize()
                row["setup_and_first_call_ms"] = (time.perf_counter() - begin) * 1000
                row["correctness"] = check_output("gemv", output, case)
                for _ in range(args.warmup):
                    run()
                torch.cuda.synchronize()
                if args.graph_calls:
                    graph, graph_output = capture(torch, run, args.graph_calls)
                    row["graph_correctness"] = verify_changed_replay(torch, graph, graph_output, case)
                    graphs[backend] = graph
                row["status"] = "PASS" if all(row[key]["status"] == "PASS" for key in
                    ("correctness", "graph_correctness") if key in row) else "FAIL"
                if row["status"] == "PASS":
                    entries[backend], runs[backend] = row, run
                    row["samples"] = {key: [] for key in ("event", "wall", "graph_event", "graph_wall")}
                    if args.profile_iterations:
                        from compare import profile_torch

                        row["torch_profile"] = profile_torch(torch, run, args.output / f"{spec['name']}--{backend}",
                                                             f"{spec['name']}::{backend}", args.profile_iterations)
            except Exception:
                row["traceback"] = traceback.format_exc()
            print(f"prepared {spec['name']} {backend}: {row['status']}", flush=True)
            save(args.output, result)
        orders = backend_orders(list(entries), args.repeat, args.seed + index)
        result["orders"][spec["name"]] = orders
        for iteration, order in enumerate(orders):
            before = gpu_state()
            for backend in order:
                for mode in (("event", "graph_event") if args.graph_calls else ("event",)):
                    graph_mode = mode == "graph_event"
                    run = graphs[backend].replay if graph_mode else runs[backend]
                    torch.cuda.nvtx.range_push(f"{spec['name']}::{backend}::{mode}")
                    event, wall = sample(run, args.graph_calls if graph_mode else 1)
                    torch.cuda.nvtx.range_pop()
                    entries[backend]["samples"][mode].append(event)
                    entries[backend]["samples"]["graph_wall" if graph_mode else "wall"].append(wall)
            result["telemetry"].append({"case": spec["name"], "round": iteration, "before": before, "after": gpu_state()})
        for row in entries.values():
            row["timing"] = {mode: summarize(values) for mode, values in row.pop("samples").items() if values}
            row["logical_bytes"] = case["logical_bytes"]
            row["flops"] = case["flops"]
        unchanged = all(torch.equal(t, old) for t, old in zip((t for t in case["args"] if t is not None), originals))
        result["inputs_unchanged"] = result.get("inputs_unchanged", True) and unchanged
        save(args.output, result)
        del graphs, runs, entries, case
        torch.cuda.synchronize()
    result["gpu_after"] = gpu_state()
    result["status"] = "PASS" if result["inputs_unchanged"] and all(row["status"] == "PASS" for row in result["rows"]) else "FAIL"
    save(args.output, result)
    print(f"{result['status']}: {len(result['rows'])} backend-case rows", flush=True)
    return 0 if result["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="")
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--graph-calls", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--profile-iterations", type=int, default=0)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"EXP-\d{8}-\d{3}", args.experiment):
        parser.error("use EXP-YYYYMMDD-NNN")
    if args.warmup < 1 or args.repeat < 2 or args.repeat % 2 or args.graph_calls < 0 or args.profile_iterations < 0:
        parser.error("use warmup >= 1, even repeat >= 2 and nonnegative graph/profile counts")
    backends = args.backends.split(",")
    if len(set(backends)) != len(backends) or set(backends) - set(BACKENDS + OPTIONAL_BACKENDS):
        parser.error(f"use distinct backends from {BACKENDS + OPTIONAL_BACKENDS}")
    try:
        specs = select_cases(args.cases)
    except ValueError as error:
        parser.error(str(error))
    if args.list:
        print(json.dumps(specs, indent=2))
        return 0
    if args.output.exists():
        parser.error("preserve old results; choose a fresh output directory")
    with Path("/tmp/crossdsl-gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True)
        return run_suite(args, specs, backends)


if __name__ == "__main__":
    raise SystemExit(main())
