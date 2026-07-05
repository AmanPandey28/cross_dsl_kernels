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

SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

BENCH_DIR = ROOT / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

FP32_ROPE_ABS_TOL = 1.0e-6


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
        "logical_gbps": (logical_bytes / (median_ms / 1000.0)) / 1e9,
        "samples_ms": samples_ms,
    }


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


def make_case(torch: Any, spec: dict[str, Any], seed: int) -> dict[str, Any]:
    from rope_kv_cuda_benchmark import make_case as make_cuda_case

    return make_cuda_case(torch, spec, seed)


def run_case(cuda_module: Any, torch: Any, spec: dict[str, Any], seed: int, warmup: int, repeat: int) -> dict[str, Any]:
    from crossdsl_kernels.references.rope_kv import rope_gqa_paged_kv_append_reference
    from crossdsl_kernels.triton.rope_kv import rope_gqa_paged_kv_append_triton
    from rope_kv_cuda_benchmark import logical_bytes

    case = make_case(torch, spec, seed)
    ref_k_cache = case["k_cache"].clone()
    ref_v_cache = case["v_cache"].clone()
    cuda_k_cache = case["k_cache"].clone()
    cuda_v_cache = case["v_cache"].clone()
    triton_k_cache = case["k_cache"].clone()
    triton_v_cache = case["v_cache"].clone()

    q_ref = rope_gqa_paged_kv_append_reference(
        case["q"],
        case["k"],
        case["v"],
        case["cos"],
        case["sin"],
        case["positions"],
        case["page_table"],
        case["sequence_ids"],
        ref_k_cache,
        ref_v_cache,
        page_size=case["page_size"],
        rope_dim=case["rope_dim"],
        interleaved=case["interleaved"],
        kv_layout=case["kv_layout"],
    )
    q_cuda = cuda_module.rope_gqa_paged_kv_append(
        case["q"],
        case["k"],
        case["v"],
        case["cos"],
        case["sin"],
        case["positions"],
        case["page_table"],
        case["sequence_ids"],
        cuda_k_cache,
        cuda_v_cache,
        case["page_size"],
        case["rope_dim"],
        case["interleaved"],
        case["kv_layout"],
    )
    q_triton = rope_gqa_paged_kv_append_triton(
        case["q"],
        case["k"],
        case["v"],
        case["cos"],
        case["sin"],
        case["positions"],
        case["page_table"],
        case["sequence_ids"],
        triton_k_cache,
        triton_v_cache,
        page_size=case["page_size"],
        rope_dim=case["rope_dim"],
        interleaved=case["interleaved"],
        kv_layout=case["kv_layout"],
    )
    torch.cuda.synchronize()
    triton_q_err = float((q_triton - q_ref).abs().max().item())
    triton_k_err = float((triton_k_cache - ref_k_cache).abs().max().item())
    triton_v_err = float((triton_v_cache - ref_v_cache).abs().max().item())
    cuda_q_err = float((q_cuda - q_ref).abs().max().item())
    cuda_k_err = float((cuda_k_cache - ref_k_cache).abs().max().item())
    cuda_v_err = float((cuda_v_cache - ref_v_cache).abs().max().item())
    triton_unwritten_match = bool(
        torch.equal(triton_k_cache == case["sentinel"], ref_k_cache == case["sentinel"])
        and torch.equal(triton_v_cache == case["sentinel"], ref_v_cache == case["sentinel"])
    )

    correctness_status = (
        "PASS"
        if triton_q_err <= FP32_ROPE_ABS_TOL
        and triton_k_err <= FP32_ROPE_ABS_TOL
        and triton_v_err <= FP32_ROPE_ABS_TOL
        and triton_unwritten_match
        else "FAIL"
    )

    def ref_call() -> Any:
        return rope_gqa_paged_kv_append_reference(
            case["q"],
            case["k"],
            case["v"],
            case["cos"],
            case["sin"],
            case["positions"],
            case["page_table"],
            case["sequence_ids"],
            ref_k_cache,
            ref_v_cache,
            page_size=case["page_size"],
            rope_dim=case["rope_dim"],
            interleaved=case["interleaved"],
            kv_layout=case["kv_layout"],
        )

    def cuda_call() -> Any:
        return cuda_module.rope_gqa_paged_kv_append(
            case["q"],
            case["k"],
            case["v"],
            case["cos"],
            case["sin"],
            case["positions"],
            case["page_table"],
            case["sequence_ids"],
            cuda_k_cache,
            cuda_v_cache,
            case["page_size"],
            case["rope_dim"],
            case["interleaved"],
            case["kv_layout"],
        )

    def triton_call() -> Any:
        return rope_gqa_paged_kv_append_triton(
            case["q"],
            case["k"],
            case["v"],
            case["cos"],
            case["sin"],
            case["positions"],
            case["page_table"],
            case["sequence_ids"],
            triton_k_cache,
            triton_v_cache,
            page_size=case["page_size"],
            rope_dim=case["rope_dim"],
            interleaved=case["interleaved"],
            kv_layout=case["kv_layout"],
        )

    bytes_model = logical_bytes(spec["tokens"], spec["q_heads"], spec["kv_heads"], spec["head_dim"], spec["rope_dim"])
    pytorch_summary = summarize(time_cuda_ms(ref_call, warmup, repeat, torch), bytes_model)
    cuda_summary = summarize(time_cuda_ms(cuda_call, warmup, repeat, torch), bytes_model)
    triton_summary = summarize(time_cuda_ms(triton_call, warmup, repeat, torch), bytes_model)
    return {
        "name": spec["name"],
        "shape": {key: spec[key] for key in ("tokens", "q_heads", "kv_heads", "head_dim", "page_size", "rope_dim")},
        "interleaved": spec["interleaved"],
        "kv_layout": spec["kv_layout"],
        "correctness": {
            "status": correctness_status,
            "triton_q_max_abs_err": triton_q_err,
            "triton_k_cache_max_abs_err": triton_k_err,
            "triton_v_cache_max_abs_err": triton_v_err,
            "triton_unwritten_mask_match": triton_unwritten_match,
            "cuda_q_max_abs_err": cuda_q_err,
            "cuda_k_cache_max_abs_err": cuda_k_err,
            "cuda_v_cache_max_abs_err": cuda_v_err,
            "tolerance": {
                "dtype": "float32",
                "max_abs_err": FP32_ROPE_ABS_TOL,
                "policy": "q, rotated k_cache, and copied v_cache max_abs_err <= 1e-6 with identical unwritten-cache mask",
            },
        },
        "pytorch_reference": pytorch_summary,
        "cuda_v0_one_block_per_token_head": cuda_summary,
        "triton_v0_one_program_per_token_head": triton_summary,
        "triton_speedup_vs_pytorch_median": pytorch_summary["median_ms"] / triton_summary["median_ms"] if triton_summary["median_ms"] > 0 else 0.0,
        "triton_speedup_vs_cuda_median": cuda_summary["median_ms"] / triton_summary["median_ms"] if triton_summary["median_ms"] > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import torch
    import triton
    from rope_kv_cuda_smoke import load_extension

    result: dict[str, Any] = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "torch": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "rows": [],
    }
    if not torch.cuda.is_available():
        result.update({"status": "SKIP", "detail": "torch.cuda.is_available() is false"})
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    cuda_module = load_extension(torch, args.verbose)
    result.update(
        {
            "torch_extensions_dir": os.environ["TORCH_EXTENSIONS_DIR"],
            "torch_cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"],
            "max_jobs": os.environ["MAX_JOBS"],
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    )
    specs = [
        {"name": "tiny_decode_nhd", "tokens": 4, "q_heads": 8, "kv_heads": 2, "head_dim": 64, "page_size": 16, "rope_dim": 64, "interleaved": False, "kv_layout": "NHD"},
        {"name": "small_decode_hnd_partial", "tokens": 16, "q_heads": 16, "kv_heads": 4, "head_dim": 128, "page_size": 16, "rope_dim": 64, "interleaved": False, "kv_layout": "HND"},
        {"name": "small_decode_interleaved", "tokens": 32, "q_heads": 16, "kv_heads": 4, "head_dim": 128, "page_size": 32, "rope_dim": 128, "interleaved": True, "kv_layout": "NHD"},
    ]
    for i, spec in enumerate(specs):
        result["rows"].append(run_case(cuda_module, torch, spec, 20260705 + i, args.warmup, args.repeat))
    result["status"] = "PASS" if all(row["correctness"]["status"] == "PASS" for row in result["rows"]) else "FAIL"

    args.json.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    args.json.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
