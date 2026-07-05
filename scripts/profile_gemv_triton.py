#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC_PY = ROOT / "src"
if str(SRC_PY) not in sys.path:
    sys.path.insert(0, str(SRC_PY))

TUNED_CONFIGS: dict[str, dict[str, int]] = {
    "tiny_square_m1": {"block_n": 8, "block_k": 512, "num_warps": 4, "num_stages": 4},
    "tiny_square_m8": {"block_n": 16, "block_k": 512, "num_warps": 4, "num_stages": 4},
    "hidden_square_m1": {"block_n": 32, "block_k": 128, "num_warps": 4, "num_stages": 4},
    "hidden_square_m8": {"block_n": 32, "block_k": 128, "num_warps": 4, "num_stages": 3},
    "hidden_square_m16": {"block_n": 32, "block_k": 128, "num_warps": 4, "num_stages": 3},
    "hidden_square_nk_m1": {"block_n": 8, "block_k": 256, "num_warps": 4, "num_stages": 3},
    "llama_mlp_up_m1": {"block_n": 16, "block_k": 128, "num_warps": 8, "num_stages": 4},
    "llama_mlp_up_m8": {"block_n": 16, "block_k": 512, "num_warps": 4, "num_stages": 3},
}


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def load_catalog(path: Path) -> list[dict[str, Any]]:
    return list(json.loads(path.read_text())["cases"])


def select_cases(catalog: list[dict[str, Any]], names: str) -> list[dict[str, Any]]:
    requested = [name.strip() for name in names.split(",") if name.strip()]
    if not requested:
        raise ValueError("at least one case name is required")
    by_name = {str(case["name"]): case for case in catalog}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"unknown case(s): {missing}")
    return [by_name[name] for name in requested]


def percentile(samples: list[float], pct: float) -> float:
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def logical_bytes(m: int, k: int, n: int, dtype_bytes: int, has_bias: bool) -> int:
    bias_bytes = n * dtype_bytes if has_bias else 0
    return (m * k + k * n + m * n) * dtype_bytes + bias_bytes


def flops(m: int, k: int, n: int, has_bias: bool) -> int:
    return 2 * m * k * n + (m * n if has_bias else 0)


def summarize(samples_ms: list[float], logical_byte_count: int, flop_count: int) -> dict[str, Any]:
    median_ms = statistics.median(samples_ms)
    seconds = median_ms / 1000.0
    return {
        "median_ms": median_ms,
        "min_ms": min(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
        "p10_ms": percentile(samples_ms, 10),
        "p90_ms": percentile(samples_ms, 90),
        "logical_bytes": logical_byte_count,
        "flops": flop_count,
        "logical_gbps": (logical_byte_count / seconds) / 1.0e9,
        "effective_tflops": (flop_count / seconds) / 1.0e12,
        "samples_ms": samples_ms,
    }


def time_cuda_ms(fn: Callable[[], Any], warmup: int, repeat: int, torch: Any, logical_byte_count: int, flop_count: int) -> dict[str, Any]:
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
    return summarize(samples, logical_byte_count, flop_count)


def nvtx_range(torch: Any, enabled: bool, name: str):
    class _Range:
        def __enter__(self) -> None:
            if enabled:
                torch.cuda.nvtx.range_push(name)

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            if enabled:
                torch.cuda.nvtx.range_pop()

    return _Range()


@dataclass
class TritonGemvCase:
    name: str
    m: int
    k: int
    n: int
    layout: str
    has_bias: bool
    tuned_config: dict[str, int]
    logical_bytes: int
    flops: int
    x: Any
    weight: Any
    bias: Any | None


class TritonGemvProfilerWorkload:
    def __init__(self, torch: Any, selected_cases: list[dict[str, Any]], *, seed: int) -> None:
        self.torch = torch
        self.last: Any = None
        self.cases = [self._make_case(case, seed + i) for i, case in enumerate(selected_cases)]

    def _make_case(self, case: dict[str, Any], seed: int) -> TritonGemvCase:
        torch = self.torch
        dtype = torch.float32
        dtype_bytes = torch.empty((), dtype=dtype).element_size()
        name = str(case["name"])
        if name not in TUNED_CONFIGS:
            raise ValueError(f"no tuned config recorded for {name}")
        m = int(case["m"])
        k = int(case["k"])
        n = int(case["n"])
        layout = str(case.get("weight_layout", "KN"))
        has_bias = bool(case.get("bias", False))
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(m, k, generator=gen, dtype=dtype, device="cpu").cuda()
        weight_kn = torch.randn(k, n, generator=gen, dtype=dtype, device="cpu").cuda()
        weight = weight_kn if layout == "KN" else weight_kn.t().contiguous()
        bias = torch.randn(n, generator=gen, dtype=dtype, device="cpu").cuda() if has_bias else None
        return TritonGemvCase(
            name=name,
            m=m,
            k=k,
            n=n,
            layout=layout,
            has_bias=has_bias,
            tuned_config=dict(TUNED_CONFIGS[name]),
            logical_bytes=logical_bytes(m, k, n, dtype_bytes, has_bias),
            flops=flops(m, k, n, has_bias),
            x=x,
            weight=weight,
            bias=bias,
        )

    def check_correctness(self) -> list[dict[str, Any]]:
        from crossdsl_kernels.references.gemv import decode_linear_reference
        from crossdsl_kernels.tolerances import fp32_gemv_status
        from crossdsl_kernels.triton.gemv import decode_gemv_triton_tiled

        rows: list[dict[str, Any]] = []
        torch = self.torch
        with torch.no_grad():
            for case in self.cases:
                y_ref = decode_linear_reference(
                    case.x,
                    case.weight,
                    case.bias,
                    out_dtype=torch.float32,
                    weight_layout=case.layout,
                )
                torch.cuda.synchronize()
                variants = {}
                for variant, kwargs in (
                    ("triton_v1_default", {}),
                    ("triton_v1_tuned", case.tuned_config),
                ):
                    y = decode_gemv_triton_tiled(case.x, case.weight, case.bias, weight_layout=case.layout, **kwargs)
                    torch.cuda.synchronize()
                    diff = y - y_ref
                    max_abs_err = float(diff.abs().max().item())
                    max_rel_err = float((diff.abs() / y_ref.abs().clamp_min(1.0e-12)).max().item())
                    rel_l2_err = float(
                        torch.linalg.vector_norm(diff).item()
                        / torch.linalg.vector_norm(y_ref).clamp_min(1.0e-12).item()
                    )
                    status, tolerance = fp32_gemv_status(case.k, max_abs_err, rel_l2_err)
                    variants[variant] = {
                        "status": status,
                        "tolerance": tolerance,
                        "max_abs_err": max_abs_err,
                        "max_rel_err": max_rel_err,
                        "rel_l2_err": rel_l2_err,
                    }
                rows.append(
                    {
                        "name": case.name,
                        "shape": {"m": case.m, "k": case.k, "n": case.n},
                        "weight_layout": case.layout,
                        "bias": case.has_bias,
                        "tuned_config": case.tuned_config,
                        "status": "PASS" if all(row["status"] == "PASS" for row in variants.values()) else "FAIL",
                        "variants": variants,
                    }
                )
        return rows

    def run_pytorch_case(self, case: TritonGemvCase, *, nvtx: bool) -> None:
        from crossdsl_kernels.references.gemv import decode_linear_reference
        from torch.profiler import record_function

        name = f"crossdsl::gemv_pytorch_ref::{case.name}"
        with nvtx_range(self.torch, nvtx, name), record_function(name):
            self.last = decode_linear_reference(
                case.x,
                case.weight,
                case.bias,
                out_dtype=self.torch.float32,
                weight_layout=case.layout,
            )

    def run_default_case(self, case: TritonGemvCase, *, nvtx: bool) -> None:
        from crossdsl_kernels.triton.gemv import decode_gemv_triton_tiled
        from torch.profiler import record_function

        name = f"crossdsl::gemv_triton_default::{case.name}"
        with nvtx_range(self.torch, nvtx, name), record_function(name):
            self.last = decode_gemv_triton_tiled(case.x, case.weight, case.bias, weight_layout=case.layout)

    def run_tuned_case(self, case: TritonGemvCase, *, nvtx: bool) -> None:
        from crossdsl_kernels.triton.gemv import decode_gemv_triton_tiled
        from torch.profiler import record_function

        config = case.tuned_config
        config_name = f"bn{config['block_n']}_bk{config['block_k']}_w{config['num_warps']}_s{config['num_stages']}"
        name = f"crossdsl::gemv_triton_tuned::{case.name}::{config_name}"
        with nvtx_range(self.torch, nvtx, name), record_function(name):
            self.last = decode_gemv_triton_tiled(case.x, case.weight, case.bias, weight_layout=case.layout, **config)

    def run_once(self, *, mode: str, nvtx: bool) -> None:
        for case in self.cases:
            if mode in {"pytorch", "all"}:
                self.run_pytorch_case(case, nvtx=nvtx)
            if mode in {"default", "all"}:
                self.run_default_case(case, nvtx=nvtx)
            if mode in {"tuned", "all"}:
                self.run_tuned_case(case, nvtx=nvtx)

    def event_benchmark(self, *, mode: str, warmup: int, repeat: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for case in self.cases:
            row: dict[str, Any] = {"name": case.name}
            if mode in {"pytorch", "all"}:
                row["pytorch_ref"] = time_cuda_ms(
                    lambda case=case: self.run_pytorch_case(case, nvtx=False),
                    warmup,
                    repeat,
                    self.torch,
                    case.logical_bytes,
                    case.flops,
                )
            if mode in {"default", "all"}:
                row["triton_v1_default"] = time_cuda_ms(
                    lambda case=case: self.run_default_case(case, nvtx=False),
                    warmup,
                    repeat,
                    self.torch,
                    case.logical_bytes,
                    case.flops,
                )
            if mode in {"tuned", "all"}:
                row["triton_v1_tuned"] = time_cuda_ms(
                    lambda case=case: self.run_tuned_case(case, nvtx=False),
                    warmup,
                    repeat,
                    self.torch,
                    case.logical_bytes,
                    case.flops,
                )
            rows.append(row)
        return rows


def event_to_dict(event: Any) -> dict[str, Any]:
    fields = [
        "key",
        "count",
        "cpu_time_total",
        "self_cpu_time_total",
        "device_time_total",
        "self_device_time_total",
        "cuda_time_total",
        "self_cuda_time_total",
        "cpu_memory_usage",
        "device_memory_usage",
        "self_cpu_memory_usage",
        "self_device_memory_usage",
    ]
    row: dict[str, Any] = {}
    for field in fields:
        value = getattr(event, field, None)
        if isinstance(value, (int, float, str)):
            row[field] = value
    return row


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=ROOT / "benchmarks" / "shapes" / "decode_gemv_smoke.json")
    parser.add_argument("--cases", default="llama_mlp_up_m1,hidden_square_m8,hidden_square_nk_m1")
    parser.add_argument("--mode", choices=["pytorch", "default", "tuned", "all"], default="all")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--profiler", choices=["none", "torch"], default="none")
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--wait", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--active", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--sync-each-iteration", action="store_true")
    parser.add_argument("--event-warmup", type=int, default=3)
    parser.add_argument("--event-repeat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    import torch
    import triton

    catalog = load_catalog(args.catalog)
    selected_cases = select_cases(catalog, args.cases)
    result: dict[str, Any] = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "profiler": args.profiler,
        "mode": args.mode,
        "catalog": str(args.catalog.relative_to(ROOT) if args.catalog.is_relative_to(ROOT) else args.catalog),
        "cases": [str(case["name"]) for case in selected_cases],
        "iterations": args.iterations,
        "tuned_configs": {str(case["name"]): TUNED_CONFIGS[str(case["name"])] for case in selected_cases},
    }
    if not torch.cuda.is_available():
        result.update({"status": "SKIP", "detail": "torch.cuda.is_available() is false"})
        write_json(args.json, result)
        print(json.dumps(result, indent=2))
        return

    torch.cuda.set_device(0)
    result.update({"device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))})
    workload = TritonGemvProfilerWorkload(torch, selected_cases, seed=args.seed)
    result["correctness"] = workload.check_correctness()
    if any(row["status"] != "PASS" for row in result["correctness"]):
        result["status"] = "FAIL"
        write_json(args.json, result)
        print(json.dumps(result, indent=2))
        raise SystemExit(1)

    if args.event_repeat > 0:
        result["event_benchmark"] = workload.event_benchmark(mode=args.mode, warmup=args.event_warmup, repeat=args.event_repeat)

    if args.profiler == "none":
        for _ in range(args.iterations):
            workload.run_once(mode=args.mode, nvtx=args.nvtx)
            if args.sync_each_iteration:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        result["status"] = "PASS"
        write_json(args.json, result)
        print(json.dumps(result, indent=2))
        return

    if args.trace_dir is None or args.table is None:
        raise SystemExit("--trace-dir and --table are required with --profiler torch")

    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    total_steps = (args.wait + args.warmup + args.active) * args.repeat
    result["torch_profiler"] = {
        "trace_dir": str(args.trace_dir),
        "table": str(args.table),
        "wait": args.wait,
        "warmup": args.warmup,
        "active": args.active,
        "repeat": args.repeat,
        "record_shapes": bool(args.record_shapes),
        "profile_memory": bool(args.profile_memory),
        "with_stack": bool(args.with_stack),
        "nvtx": bool(args.nvtx),
        "sync_each_iteration": bool(args.sync_each_iteration),
        "total_steps": total_steps,
    }
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=args.repeat),
        on_trace_ready=tensorboard_trace_handler(str(args.trace_dir)),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        with_flops=True,
    ) as prof:
        for _ in range(total_steps):
            workload.run_once(mode=args.mode, nvtx=args.nvtx)
            if args.sync_each_iteration:
                torch.cuda.synchronize()
            prof.step()
    torch.cuda.synchronize()

    key_averages = prof.key_averages(group_by_input_shape=args.record_shapes)
    try:
        table = key_averages.table(sort_by="self_cuda_time_total", row_limit=80)
        sort_field = "self_cuda_time_total"
    except Exception:
        table = key_averages.table(sort_by="self_device_time_total", row_limit=80)
        sort_field = "self_device_time_total"
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(table + "\n")
    events = list(key_averages)
    events.sort(
        key=lambda event: float(getattr(event, "self_cuda_time_total", getattr(event, "self_device_time_total", 0.0))),
        reverse=True,
    )
    result["torch_profiler"]["sort_field"] = sort_field
    result["torch_profiler"]["top_events"] = [event_to_dict(event) for event in events[:80]]
    result["status"] = "PASS"
    write_json(args.json, result)
    print(json.dumps(result, indent=2))
    print(table)


if __name__ == "__main__":
    main()
