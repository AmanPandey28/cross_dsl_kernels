#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
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


def time_cuda_ms(fn: Callable[[], Any], warmup: int, repeat: int, torch: Any) -> dict[str, Any]:
    import statistics

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
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "mean_ms": statistics.mean(samples),
        "samples_ms": samples,
    }


def make_profile_case(torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    from rope_kv_triton_benchmark import make_case

    spec = {
        "name": "small_decode_interleaved",
        "tokens": 32,
        "q_heads": 16,
        "kv_heads": 4,
        "head_dim": 128,
        "page_size": 32,
        "rope_dim": 128,
        "interleaved": True,
        "kv_layout": "NHD",
    }
    return make_case(torch, spec, 20260705), spec


def correctness(cuda_module: Any, torch: Any, case: dict[str, Any]) -> dict[str, Any]:
    from crossdsl_kernels.references.rope_kv import rope_gqa_paged_kv_append_reference
    from crossdsl_kernels.triton.rope_kv import rope_gqa_paged_kv_append_triton

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
    status = (
        "PASS"
        if triton_q_err <= FP32_ROPE_ABS_TOL
        and triton_k_err <= FP32_ROPE_ABS_TOL
        and triton_v_err <= FP32_ROPE_ABS_TOL
        and triton_unwritten_match
        else "FAIL"
    )
    return {
        "status": status,
        "triton_q_max_abs_err": triton_q_err,
        "triton_k_cache_max_abs_err": triton_k_err,
        "triton_v_cache_max_abs_err": triton_v_err,
        "triton_unwritten_mask_match": triton_unwritten_match,
        "cuda_q_max_abs_err": cuda_q_err,
        "cuda_k_cache_max_abs_err": cuda_k_err,
        "cuda_v_cache_max_abs_err": cuda_v_err,
    }


def make_calls(cuda_module: Any, case: dict[str, Any]) -> dict[str, Callable[[], Any]]:
    from crossdsl_kernels.references.rope_kv import rope_gqa_paged_kv_append_reference
    from crossdsl_kernels.triton.rope_kv import rope_gqa_paged_kv_append_triton

    def pytorch_ref() -> Any:
        return rope_gqa_paged_kv_append_reference(
            case["q"],
            case["k"],
            case["v"],
            case["cos"],
            case["sin"],
            case["positions"],
            case["page_table"],
            case["sequence_ids"],
            case["k_cache"],
            case["v_cache"],
            page_size=case["page_size"],
            rope_dim=case["rope_dim"],
            interleaved=case["interleaved"],
            kv_layout=case["kv_layout"],
        )

    def cuda_v0() -> Any:
        return cuda_module.rope_gqa_paged_kv_append(
            case["q"],
            case["k"],
            case["v"],
            case["cos"],
            case["sin"],
            case["positions"],
            case["page_table"],
            case["sequence_ids"],
            case["k_cache"],
            case["v_cache"],
            case["page_size"],
            case["rope_dim"],
            case["interleaved"],
            case["kv_layout"],
        )

    def triton_v0() -> Any:
        return rope_gqa_paged_kv_append_triton(
            case["q"],
            case["k"],
            case["v"],
            case["cos"],
            case["sin"],
            case["positions"],
            case["page_table"],
            case["sequence_ids"],
            case["k_cache"],
            case["v_cache"],
            page_size=case["page_size"],
            rope_dim=case["rope_dim"],
            interleaved=case["interleaved"],
            kv_layout=case["kv_layout"],
        )

    return {"pytorch_ref": pytorch_ref, "cuda_v0": cuda_v0, "triton_v0": triton_v0}


def run_torch_profiler(torch: Any, calls: dict[str, Callable[[], Any]], args: argparse.Namespace) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    sched = schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=args.profiler_repeat)
    with profile(
        activities=activities,
        schedule=sched,
        on_trace_ready=tensorboard_trace_handler(str(args.trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range((args.wait + args.warmup + args.active) * args.profiler_repeat):
            for name, fn in calls.items():
                with record_function(f"crossdsl::rope_kv::{name}"):
                    fn()
            torch.cuda.synchronize()
            prof.step()
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
    if args.table:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text(table)
    return {"trace_dir": str(args.trace_dir), "table": str(args.table) if args.table else "", "status": "PASS"}


def run_plain_iterations(torch: Any, calls: dict[str, Callable[[], Any]], iterations: int, use_nvtx: bool) -> None:
    for _ in range(iterations):
        for name, fn in calls.items():
            if use_nvtx:
                torch.cuda.nvtx.range_push(f"crossdsl::rope_kv::{name}")
            fn()
            if use_nvtx:
                torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--profiler", choices=["none", "torch"], default="none")
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "results" / "profiles" / "rope_kv_triton" / "torch")
    parser.add_argument("--table", type=Path)
    parser.add_argument("--event-warmup", type=int, default=5)
    parser.add_argument("--event-repeat", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--wait", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--active", type=int, default=3)
    parser.add_argument("--profiler-repeat", type=int, default=1)
    parser.add_argument("--nvtx", action="store_true")
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
        "profiler": args.profiler,
    }
    if not torch.cuda.is_available():
        result.update({"status": "SKIP", "detail": "torch.cuda.is_available() is false"})
    else:
        cuda_module = load_extension(torch, args.verbose)
        case, spec = make_profile_case(torch)
        calls = make_calls(cuda_module, case)
        result.update(
            {
                "torch_extensions_dir": os.environ["TORCH_EXTENSIONS_DIR"],
                "torch_cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"],
                "max_jobs": os.environ["MAX_JOBS"],
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "case": spec,
                "correctness": correctness(cuda_module, torch, case),
                "event_benchmark": {
                    name: time_cuda_ms(fn, args.event_warmup, args.event_repeat, torch) for name, fn in calls.items()
                },
            }
        )
        if args.profiler == "torch":
            result["torch_profiler"] = run_torch_profiler(torch, calls, args)
        else:
            run_plain_iterations(torch, calls, args.iterations, args.nvtx)
        result["status"] = "PASS" if result["correctness"]["status"] == "PASS" else "FAIL"

    args.json.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    args.json.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if result.get("status") in {"PASS", "SKIP"} else 1)


if __name__ == "__main__":
    main()
