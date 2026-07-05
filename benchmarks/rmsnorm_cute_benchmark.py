#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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


def summarize(samples_ms: list[float], logical_bytes: int) -> dict[str, Any]:
    median_ms = statistics.median(samples_ms)
    return {
        "median_ms": median_ms,
        "min_ms": min(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
        "p10_ms": percentile(samples_ms, 10),
        "p90_ms": percentile(samples_ms, 90),
        "logical_bytes": logical_bytes,
        "logical_gbps": (logical_bytes / (median_ms / 1000.0)) / 1.0e9,
        "samples_ms": samples_ms,
    }


def time_cuda_ms(
    fn: Callable[[], Any], warmup: int, repeat: int, torch: Any
) -> list[float]:
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


def parse_shapes(text: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        rows, hidden = item.lower().split("x", maxsplit=1)
        shapes.append((int(rows), int(hidden)))
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument(
        "--shapes", default="1x1024,4x1024,16x1024,4x4096,16x4096,4x8192"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    args = parser.parse_args()

    import torch
    import cutlass
    from crossdsl_kernels.references.rmsnorm import fused_residual_rmsnorm_reference
    from crossdsl_kernels.cute.rmsnorm import fused_residual_rmsnorm_cute

    result: dict[str, Any] = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "torch": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "cutlass": cutlass.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "eps": args.eps,
        "rows": [],
    }
    if not torch.cuda.is_available():
        result.update(
            {"status": "SKIP", "detail": "torch.cuda.is_available() is false"}
        )
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    result.update(
        {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    )
    for i, (rows, hidden) in enumerate(parse_shapes(args.shapes)):
        gen = torch.Generator(device="cpu").manual_seed(20260704 + i)
        x = torch.randn(
            rows, hidden, generator=gen, dtype=torch.float32, device="cpu"
        ).cuda()
        residual = torch.randn(
            rows, hidden, generator=gen, dtype=torch.float32, device="cpu"
        ).cuda()
        weight = torch.randn(
            hidden, generator=gen, dtype=torch.float32, device="cpu"
        ).cuda()

        y_ref, residual_ref = fused_residual_rmsnorm_reference(
            x, residual, weight, args.eps
        )
        y_cute, residual_cute = fused_residual_rmsnorm_cute(
            x, residual, weight, args.eps
        )
        torch.cuda.synchronize()
        y_max_abs_err = float((y_cute - y_ref).abs().max().item())
        residual_max_abs_err = float((residual_cute - residual_ref).abs().max().item())

        ref_samples = time_cuda_ms(
            lambda: fused_residual_rmsnorm_reference(x, residual, weight, args.eps),
            args.warmup,
            args.repeat,
            torch,
        )
        cute_samples = time_cuda_ms(
            lambda: fused_residual_rmsnorm_cute(x, residual, weight, args.eps),
            args.warmup,
            args.repeat,
            torch,
        )
        logical_bytes = rows * hidden * 4 * 5
        ref_summary = summarize(ref_samples, logical_bytes)
        cute_summary = summarize(cute_samples, logical_bytes)
        result["rows"].append(
            {
                "shape": [rows, hidden],
                "elements": rows * hidden,
                "correctness": {
                    "y_max_abs_err": y_max_abs_err,
                    "residual_max_abs_err": residual_max_abs_err,
                    "status": "PASS"
                    if y_max_abs_err <= 2.0e-5 and residual_max_abs_err == 0.0
                    else "FAIL",
                },
                "pytorch_reference": ref_summary,
                "cute_kernel": cute_summary,
                "speedup_vs_pytorch_median": ref_summary["median_ms"]
                / cute_summary["median_ms"],
            }
        )
        del x, residual, weight, y_ref, residual_ref, y_cute, residual_cute
        torch.cuda.synchronize()

    result["status"] = (
        "PASS"
        if all(row["correctness"]["status"] == "PASS" for row in result["rows"])
        else "FAIL"
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    args.json.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
