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
    matmul_flops = 2 * m * k * n
    bias_flops = m * n if has_bias else 0
    return matmul_flops + bias_flops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=ROOT / "benchmarks" / "shapes" / "decode_gemv_smoke.json")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--dtype", choices=["float32"], default="float32")
    args = parser.parse_args()

    import torch
    from crossdsl_kernels.references.gemv import decode_linear_reference
    from crossdsl_kernels.tolerances import fp32_gemv_status

    result: dict[str, Any] = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "catalog": str(args.catalog.relative_to(ROOT) if args.catalog.is_relative_to(ROOT) else args.catalog),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "dtype": args.dtype,
        "rows": [],
    }
    if not torch.cuda.is_available():
        result.update({"status": "SKIP", "detail": "torch.cuda.is_available() is false"})
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    device = torch.device("cuda")
    dtype = torch.float32
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    result.update({"device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))})

    for i, case in enumerate(load_cases(args.catalog)):
        m = int(case["m"])
        k = int(case["k"])
        n = int(case["n"])
        weight_layout = str(case.get("weight_layout", "KN"))
        has_bias = bool(case.get("bias", False))
        gen = torch.Generator(device="cpu").manual_seed(20260703 + i)
        x = torch.randn(m, k, generator=gen, dtype=dtype, device="cpu").to(device)
        weight_kn = torch.randn(k, n, generator=gen, dtype=dtype, device="cpu").to(device)
        weight = weight_kn if weight_layout == "KN" else weight_kn.t().contiguous()
        bias = torch.randn(n, generator=gen, dtype=dtype, device="cpu").to(device) if has_bias else None

        out = decode_linear_reference(x, weight, bias, out_dtype=dtype, weight_layout=weight_layout)
        expected = x @ weight_kn
        if bias is not None:
            expected = expected + bias
        torch.cuda.synchronize()
        max_abs_err = float((out - expected).abs().max().item())
        max_rel_err = float(((out - expected).abs() / expected.abs().clamp_min(1.0e-12)).max().item())
        rel_l2_err = float(torch.linalg.vector_norm(out - expected).item() / torch.linalg.vector_norm(expected).clamp_min(1.0e-12).item())
        correctness_status, tolerance = fp32_gemv_status(k, max_abs_err, rel_l2_err)

        bytes_lower = logical_bytes(m, k, n, dtype_bytes, has_bias)
        flop_count = flops(m, k, n, has_bias)
        samples = time_cuda_ms(
            lambda: decode_linear_reference(x, weight, bias, out_dtype=dtype, weight_layout=weight_layout),
            args.warmup,
            args.repeat,
            torch,
        )
        summary = summarize(samples, bytes_lower, flop_count)
        result["rows"].append(
            {
                "name": str(case["name"]),
                "shape": {"m": m, "k": k, "n": n},
                "weight_layout": weight_layout,
                "bias": has_bias,
                "arithmetic_intensity_flop_per_byte": flop_count / bytes_lower,
                "correctness": {
                    "status": correctness_status,
                    "tolerance": tolerance,
                    "max_abs_err": max_abs_err,
                    "max_rel_err": max_rel_err,
                    "rel_l2_err": rel_l2_err,
                },
                "pytorch_matmul_reference": summary,
            }
        )
        del x, weight_kn, weight, bias, out, expected
        torch.cuda.synchronize()

    result["status"] = "PASS" if all(row["correctness"]["status"] == "PASS" for row in result["rows"]) else "FAIL"
    args.json.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    args.json.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
