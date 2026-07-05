#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC_PY = ROOT / "src"
if str(SRC_PY) not in sys.path:
    sys.path.insert(0, str(SRC_PY))


def percentile(samples: list[float], pct: float) -> float:
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def time_cuda_ms(fn: Callable[[], Any], warmup: int, repeat: int, torch: Any) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def summarize(samples_ms: list[float], logical_bytes: int, flops: int) -> dict[str, Any]:
    median_ms = statistics.median(samples_ms)
    seconds = median_ms / 1000.0
    return {
        "median_ms": median_ms,
        "min_ms": min(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
        "p10_ms": percentile(samples_ms, 10),
        "p90_ms": percentile(samples_ms, 90),
        "logical_bytes": logical_bytes,
        "flops": flops,
        "logical_gbps": (logical_bytes / seconds) / 1e9,
        "effective_tflops": (flops / seconds) / 1e12,
        "samples_ms": samples_ms,
    }


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return list(data["cases"])


def logical_bytes(m: int, k: int, n: int, dtype_bytes: int, has_bias: bool) -> int:
    bias_bytes = n * dtype_bytes if has_bias else 0
    return (m * k + k * n + m * n) * dtype_bytes + bias_bytes


def flops(m: int, k: int, n: int, has_bias: bool) -> int:
    return 2 * m * k * n + (m * n if has_bias else 0)


def check_correctness(torch: Any, y: Any, y_ref: Any, k: int) -> dict[str, Any]:
    from crossdsl_kernels.tolerances import fp32_gemv_status

    diff = y - y_ref
    max_abs_err = float(diff.abs().max().item())
    max_rel_err = float((diff.abs() / y_ref.abs().clamp_min(1.0e-12)).max().item())
    rel_l2_err = float(torch.linalg.vector_norm(diff).item() / torch.linalg.vector_norm(y_ref).clamp_min(1.0e-12).item())
    correctness_status, tolerance = fp32_gemv_status(k, max_abs_err, rel_l2_err)
    return {
        "status": correctness_status,
        "tolerance": tolerance,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "rel_l2_err": rel_l2_err,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=ROOT / "benchmarks" / "shapes" / "decode_gemv_smoke.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    import torch
    import triton
    from crossdsl_kernels.references.gemv import decode_linear_reference
    from crossdsl_kernels.triton.gemv import decode_gemv_triton, decode_gemv_triton_tiled

    result: dict[str, Any] = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "catalog": str(args.catalog.relative_to(ROOT) if args.catalog.is_relative_to(ROOT) else args.catalog),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "dtype": "float32",
        "rows": [],
    }
    if not torch.cuda.is_available():
        result.update({"status": "SKIP", "detail": "torch.cuda.is_available() is false"})
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    dtype = torch.float32
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    result.update({"device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))})

    for i, case in enumerate(load_cases(args.catalog)):
        m = int(case["m"])
        k = int(case["k"])
        n = int(case["n"])
        layout = str(case.get("weight_layout", "KN"))
        has_bias = bool(case.get("bias", False))
        gen = torch.Generator(device="cpu").manual_seed(20260704 + i)
        x = torch.randn(m, k, generator=gen, dtype=dtype, device="cpu").cuda()
        weight_kn = torch.randn(k, n, generator=gen, dtype=dtype, device="cpu").cuda()
        weight = weight_kn if layout == "KN" else weight_kn.t().contiguous()
        bias = torch.randn(n, generator=gen, dtype=dtype, device="cpu").cuda() if has_bias else None

        y_ref = decode_linear_reference(x, weight, bias, out_dtype=dtype, weight_layout=layout)
        y_triton_v0 = decode_gemv_triton(x, weight, bias, weight_layout=layout)
        y_triton_v1 = decode_gemv_triton_tiled(x, weight, bias, weight_layout=layout)
        torch.cuda.synchronize()
        correctness_v0 = check_correctness(torch, y_triton_v0, y_ref, k)
        correctness_v1 = check_correctness(torch, y_triton_v1, y_ref, k)
        correctness_status = "PASS" if correctness_v0["status"] == "PASS" and correctness_v1["status"] == "PASS" else "FAIL"
        lower_bytes = logical_bytes(m, k, n, dtype_bytes, has_bias)
        flop_count = flops(m, k, n, has_bias)
        pytorch_samples = time_cuda_ms(
            lambda: decode_linear_reference(x, weight, bias, out_dtype=dtype, weight_layout=layout),
            args.warmup,
            args.repeat,
            torch,
        )
        triton_samples = time_cuda_ms(
            lambda: decode_gemv_triton(x, weight, bias, weight_layout=layout),
            args.warmup,
            args.repeat,
            torch,
        )
        triton_tiled_samples = time_cuda_ms(
            lambda: decode_gemv_triton_tiled(x, weight, bias, weight_layout=layout),
            args.warmup,
            args.repeat,
            torch,
        )
        pytorch_summary = summarize(pytorch_samples, lower_bytes, flop_count)
        triton_summary = summarize(triton_samples, lower_bytes, flop_count)
        triton_tiled_summary = summarize(triton_tiled_samples, lower_bytes, flop_count)
        result["rows"].append(
            {
                "name": str(case["name"]),
                "shape": {"m": m, "k": k, "n": n},
                "weight_layout": layout,
                "bias": has_bias,
                "arithmetic_intensity_flop_per_byte": flop_count / lower_bytes,
                "correctness": {
                    "status": correctness_status,
                    "triton_v0_one_program_per_output": correctness_v0,
                    "triton_v1_column_tile": correctness_v1,
                },
                "pytorch_matmul_reference": pytorch_summary,
                "triton_v0_one_program_per_output": triton_summary,
                "triton_v1_column_tile": triton_tiled_summary,
                "triton_v0_speedup_vs_pytorch_median": pytorch_summary["median_ms"] / triton_summary["median_ms"],
                "triton_v1_speedup_vs_pytorch_median": pytorch_summary["median_ms"] / triton_tiled_summary["median_ms"],
                "triton_v1_speedup_vs_v0_median": triton_summary["median_ms"] / triton_tiled_summary["median_ms"],
            }
        )
        del x, weight_kn, weight, bias, y_ref, y_triton_v0, y_triton_v1
        torch.cuda.synchronize()

    result["status"] = "PASS" if all(row["correctness"]["status"] == "PASS" for row in result["rows"]) else "FAIL"
    args.json.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    args.json.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
