#!/usr/bin/env python3
"""One bounded entry point for W4A16 correctness, timings and Torch traces."""

from __future__ import annotations

import argparse
import fcntl
import json
import random
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import capture, gpu_state, measure, measure_graph, provenance
from compare import profile_torch

BACKENDS = (
    "torch_dense",
    "torch_predequant",
    "torch_dequant",
    "cuda",
    "cuda_rows",
    "triton",
    "triton_rows",
    "cute",
    "cute_rows",
)
TOLERANCE = {"atol": 0.002, "rtol": 0.002, "rel_l2": 0.001}


def check_output(actual, expected):
    import torch

    a, b = actual.float(), expected.float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"status": "FAIL", "finite": False, "tolerance": TOLERANCE}
    diff = a - b
    scaled = (
        (diff.abs() / (TOLERANCE["atol"] + TOLERANCE["rtol"] * b.abs())).max().item()
    )
    l2 = (
        torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(b).clamp_min(1e-12)
    ).item()
    return {
        "status": "PASS" if scaled <= 1 and l2 <= TOLERANCE["rel_l2"] else "FAIL",
        "finite": True,
        "max_abs_err": diff.abs().max().item(),
        "scaled_max_err": scaled,
        "rel_l2_err": l2,
        "tolerance": TOLERANCE,
    }


def error_summary(actual, expected):
    import torch

    a, b = actual.float(), expected.float()
    diff = a - b
    return {
        "max_abs_err": diff.abs().max().item(),
        "rel_l2_err": (
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(b).clamp_min(1e-12)
        ).item(),
        "role": "quantization quality, NOT the kernel correctness gate",
    }


def make_case(spec, seed):
    import torch
    from crossdsl_kernels.quantization import dequantize_int4, quantize_int4

    m, k, n, group = (spec[key] for key in ("m", "k", "n", "group_size"))
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=generator, device="cuda", dtype=torch.float16)
    original = torch.randn(
        n, k, generator=generator, device="cuda", dtype=torch.float16
    )
    if spec.get("zero"):
        original.zero_()
    bias = (
        torch.randn(n, generator=generator, device="cuda", dtype=torch.float16)
        if spec["bias"]
        else None
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    packed, scales = quantize_int4(original, group)
    torch.cuda.synchronize()
    packing_ms = (time.perf_counter() - start) * 1000
    dequantized = dequantize_int4(packed, scales, k, group)
    # CPU FP64 avoids inheriting cuBLAS's reduction policy in the oracle.
    xc = x.cpu().double()

    def oracle(weight):
        output = xc @ weight.cpu().double().t()
        if bias is not None:
            output += bias.cpu().double()
        return output.half().cuda()

    expected, dense_expected = oracle(dequantized), oracle(original)
    inputs = (x, packed, scales) + (() if bias is None else (bias,))
    return {
        "spec": spec,
        "x": x,
        "packed": packed,
        "scales": scales,
        "bias": bias,
        "original": original,
        "dequantized": dequantized,
        "expected": expected,
        "dense_expected": dense_expected,
        "inputs": inputs,
        "snapshots": tuple(t.clone() for t in inputs),
        "packing_wall_ms": packing_ms,
        "weight_quantization_error": error_summary(dequantized, original),
        "output_quantization_error": error_summary(expected, dense_expected),
    }


def prepare_backend(case, backend):
    import torch
    from crossdsl_kernels.quantization import dequantize_int4
    from crossdsl_kernels.w4a16 import w4a16_linear

    x, packed, scales, bias = (case[key] for key in ("x", "packed", "scales", "bias"))
    m, k = x.shape
    n = packed.shape[0]
    group = case["spec"]["group_size"]
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if backend.startswith("torch_"):
        accumulator = torch.empty((m, n), device=x.device, dtype=torch.float32)

        def run():
            if backend == "torch_dequant":
                weight = dequantize_int4(packed, scales, k, group)
            else:
                weight = case["original" if backend == "torch_dense" else "dequantized"]
            torch.mm(x, weight.t(), out_dtype=torch.float32, out=accumulator)
            if bias is not None:
                accumulator.add_(bias)
            y.copy_(accumulator)
            return y

        return run, y
    language = backend.split("_")[0]
    rows = 4 if backend.endswith("_rows") else 1
    if language == "cute":
        from crossdsl_kernels.cute.launch import prepare_launch
        from crossdsl_kernels.cute.w4a16 import _launch
        from crossdsl_kernels.quantization import validate_w4a16

        validate_w4a16(x, packed, scales, bias, group_size=group, out=y)
        tensors = (x, packed, scales, scales if bias is None else bias, y)
        prepared = prepare_launch(
            _launch,
            tuple(t.flatten() for t in tensors),
            (m, k, n, group, rows, bias is not None),
            constexpr_scalars=(1, 2, 3, 4, 5),
        )

        def run():
            prepared()
            return y

        return run, y

    def run():
        return w4a16_linear(
            x,
            packed,
            scales,
            bias,
            group_size=group,
            backend=language,
            rows_per_block=rows,
            out=y,
        )

    return run, y


def write_result(directory, result):
    (directory / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "| Case | Backend | Status | Event us | Graph us |",
        "|---|---|---|---:|---:|",
    ]
    for row in result["rows"]:

        def timing(key):
            value = row.get(key, {}).get("cuda_event", {}).get("median_ms")
            return "-" if value is None else f"{value * 1000:.3f}"

        lines.append(
            f"| {row['case']} | {row['backend']} | {row['status']} | {timing('timing')} | {timing('graph_timing')} |"
        )
    (directory / "results.md").write_text("\n".join(lines) + "\n")


def run_suite(args, specs, backends):
    import torch

    torch.set_num_threads(4)
    result = provenance(torch, args.experiment)
    result.update(
        status="RUNNING",
        rows=[],
        cases=[],
        args=vars(args) | {"output": str(args.output)},
        numerical_contract="FP16 dequantization, FP32 accumulation/bias, FP16 output",
        format="Custom INT4 NK nibble=q+8; not an AWQ/GPTQ checkpoint format",
        tolerance=TOLERANCE,
        build_flags={
            "cxx": ["-O3"],
            "cuda": ["-O3", "-gencode=arch=compute_120,code=sm_120"],
        },
    )
    if not torch.cuda.is_available():
        result.update(status="SKIP", reason="Sandbox CUDA-hidden; no GPU claim")
        write_result(args.output, result)
        return 2
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This first campaign requires SM120")
    result.update(
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    order = random.Random(args.seed)
    for index, spec in enumerate(specs):
        case = make_case(spec, args.seed + index)
        packed_bytes = case["packed"].numel() + 2 * case["scales"].numel()
        result["cases"].append(
            {
                "spec": spec,
                "seed": args.seed + index,
                "packed_weight_and_scale_bytes": packed_bytes,
                "dense_fp16_weight_bytes": 2 * spec["n"] * spec["k"],
                **{
                    key: case[key]
                    for key in (
                        "packing_wall_ms",
                        "weight_quantization_error",
                        "output_quantization_error",
                    )
                },
            }
        )
        selected = backends.copy()
        order.shuffle(selected)
        for backend in selected:
            row = {"case": spec["name"], "backend": backend, "status": "RUNNING"}
            label = f"w4a16/{spec['name']}/{backend}"
            try:
                start = time.perf_counter()
                run, y = prepare_backend(case, backend)
                run()
                torch.cuda.synchronize()
                row["prepare_and_first_call_wall_ms"] = (
                    time.perf_counter() - start
                ) * 1000
                expected = case[
                    "dense_expected" if backend == "torch_dense" else "expected"
                ]
                row["correctness"] = check_output(y, expected)
                row["status"] = row["correctness"]["status"]
                row["inputs_unchanged"] = all(
                    torch.equal(a, b) for a, b in zip(case["inputs"], case["snapshots"])
                )
                if not row["inputs_unchanged"]:
                    row["status"] = "FAIL"
                if row["status"] != "PASS":
                    raise RuntimeError(
                        "direct correctness or input mutation gate failed"
                    )
                if args.mode == "check":
                    # Exercise the public CuTe adapter too, not just its prepared ABI.
                    if backend.startswith("cute"):
                        from crossdsl_kernels.w4a16 import w4a16_linear

                        direct = w4a16_linear(
                            case["x"],
                            case["packed"],
                            case["scales"],
                            case["bias"],
                            backend="cute",
                            group_size=spec["group_size"],
                            rows_per_block=4 if backend.endswith("rows") else 1,
                        )
                        torch.cuda.synchronize()
                        row["public_correctness"] = check_output(direct, expected)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    y.fill_(float("nan"))
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        run()
                    torch.cuda.current_stream().wait_stream(stream)
                    row["side_stream_correctness"] = check_output(y, expected)
                torch.cuda.nvtx.range_push(label)
                try:
                    if args.mode == "benchmark":
                        row["timing"] = measure(torch, run, args.warmup, args.repeat)
                    if args.graph_calls:
                        graph, graph_output = capture(torch, run, args.graph_calls)
                        graph_output.fill_(float("nan"))
                        graph.replay()
                        torch.cuda.synchronize()
                        row["graph_correctness"] = check_output(graph_output, expected)
                        if args.mode == "benchmark":
                            row["graph_timing"] = measure_graph(
                                torch, graph, args.repeat, args.graph_calls
                            )
                        del graph, graph_output
                    if args.mode == "torch":
                        row["torch_profile"] = profile_torch(
                            torch,
                            run,
                            args.output / f"{spec['name']}--{backend}",
                            label,
                            3,
                        )
                    if args.mode == "workload":
                        for _ in range(args.warmup + args.repeat):
                            run()
                        torch.cuda.synchronize()
                finally:
                    torch.cuda.nvtx.range_pop()
                checks = [
                    value for key, value in row.items() if key.endswith("correctness")
                ]
                row["status"] = (
                    "PASS"
                    if all(value["status"] == "PASS" for value in checks)
                    else "FAIL"
                )
                del run, y
            except Exception as error:
                row.update(
                    status="ERROR",
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
            result["rows"].append(row)
            write_result(args.output, result)
            print(f"{spec['name']} {backend}: {row['status']}", flush=True)
        del case
        torch.cuda.synchronize()
    result.update(
        gpu_after=gpu_state(),
        status="PASS"
        if all(row["status"] == "PASS" for row in result["rows"])
        else "FAIL",
    )
    write_result(args.output, result)
    return 0 if result["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("check", "benchmark", "torch", "workload"), default="check"
    )
    parser.add_argument("--cases", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--graph-calls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    if not re.fullmatch(r"EXP-\d{8}-\d{3}", args.experiment):
        parser.error("use an EXP-YYYYMMDD-NNN experiment ID")
    backends = args.backends.split(",")
    if len(set(backends)) != len(backends) or set(backends) - set(BACKENDS):
        parser.error(f"select unique backends from {BACKENDS}")
    if args.warmup < 1 or args.repeat < 2 or args.graph_calls < 0:
        parser.error("warmup >= 1, repeat >= 2, graph-calls >= 0 required")
    specs = json.loads((ROOT / "benchmarks/shapes/w4a16.json").read_text())["cases"]
    if args.cases:
        names = args.cases.split(",")
        if set(names) - {spec["name"] for spec in specs}:
            parser.error("unknown case name")
        specs = [spec for spec in specs if spec["name"] in names]
    elif args.smoke:
        specs = [spec for spec in specs if spec.get("smoke")]
    if args.output.exists():
        parser.error("output already exists; preserve evidence with a fresh path")
    args.output.mkdir(parents=True)
    with Path("/tmp/crossdsl-gpu.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another experiment holds the GPU lock")
        raise SystemExit(run_suite(args, specs, backends))


if __name__ == "__main__":
    main()
