#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "csrc" / "gemv"
SRC_PY = ROOT / "src"
if str(SRC_PY) not in sys.path:
    sys.path.insert(0, str(SRC_PY))


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def load_extension(torch: Any, verbose: bool) -> Any:
    from torch.utils.cpp_extension import load

    ext_dir = ROOT / "results" / "build" / "torch_extensions"
    ext_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ext_dir))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", "4")
    name = safe_name(f"crossdsl_gemv_{platform.python_version()}_{torch.__version__}")
    return load(
        name=name,
        sources=[str(SRC_DIR / "gemv_ext.cpp"), str(SRC_DIR / "gemv_kernel.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcublas", "-lcublasLt"],
        verbose=verbose,
    )


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


def summarize(samples_ms: list[float]) -> dict[str, float | list[float]]:
    return {
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
        "p10_ms": percentile(samples_ms, 10),
        "p90_ms": percentile(samples_ms, 90),
        "samples_ms": samples_ms,
    }


def time_cuda_ms(fn: Callable[[], Any], warmup: int, repeat: int, torch: Any) -> dict[str, float | list[float]]:
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
    return summarize(samples)


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
class GemvCase:
    name: str
    m: int
    k: int
    n: int
    layout: str
    has_bias: bool
    x: Any
    weight: Any
    bias: Any
    bias_ref: Any


class GemvProfilerWorkload:
    def __init__(
        self,
        torch: Any,
        module: Any,
        selected_cases: list[dict[str, Any]],
        *,
        seed: int,
    ) -> None:
        self.torch = torch
        self.module = module
        self.last: Any = None
        self.cases = [self._make_case(case, seed + i) for i, case in enumerate(selected_cases)]

    def _make_case(self, case: dict[str, Any], seed: int) -> GemvCase:
        torch = self.torch
        dtype = torch.float32
        m = int(case["m"])
        k = int(case["k"])
        n = int(case["n"])
        layout = str(case.get("weight_layout", "KN"))
        has_bias = bool(case.get("bias", False))
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(m, k, generator=gen, dtype=dtype, device="cpu").cuda()
        weight_kn = torch.randn(k, n, generator=gen, dtype=dtype, device="cpu").cuda()
        weight = weight_kn if layout == "KN" else weight_kn.t().contiguous()
        bias = torch.randn(n, generator=gen, dtype=dtype, device="cpu").cuda() if has_bias else torch.empty(0, device="cuda", dtype=dtype)
        return GemvCase(
            name=str(case["name"]),
            m=m,
            k=k,
            n=n,
            layout=layout,
            has_bias=has_bias,
            x=x,
            weight=weight,
            bias=bias,
            bias_ref=bias if has_bias else None,
        )

    def check_correctness(self) -> list[dict[str, Any]]:
        from crossdsl_kernels.references.gemv import decode_linear_reference
        from crossdsl_kernels.tolerances import fp32_gemv_status

        rows: list[dict[str, Any]] = []
        torch = self.torch
        with torch.no_grad():
            for case in self.cases:
                y_ref = decode_linear_reference(
                    case.x,
                    case.weight,
                    case.bias_ref,
                    out_dtype=torch.float32,
                    weight_layout=case.layout,
                )
                torch.cuda.synchronize()
                variants = {}
                for variant, fn in (
                    ("cuda_v0_one_block_per_output", self.module.decode_gemv),
                    ("cuda_v1_warp_per_output", self.module.decode_gemv_warp),
                    ("cublas_sgemv_or_sgemm", self.module.decode_gemv_cublas),
                    ("cublaslt_matmul", self.module.decode_gemv_cublaslt),
                ):
                    y_cuda = fn(case.x, case.weight, case.bias, case.layout)
                    torch.cuda.synchronize()
                    diff = y_cuda - y_ref
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
                        "status": "PASS" if all(row["status"] == "PASS" for row in variants.values()) else "FAIL",
                        "variants": variants,
                    }
                )
        return rows

    def run_cuda_case(self, case: GemvCase, *, nvtx: bool) -> None:
        from torch.profiler import record_function

        name = f"crossdsl::gemv_cuda_v0::{case.name}"
        with nvtx_range(self.torch, nvtx, name), record_function(name):
            self.last = self.module.decode_gemv(case.x, case.weight, case.bias, case.layout)

    def run_warp_case(self, case: GemvCase, *, nvtx: bool) -> None:
        from torch.profiler import record_function

        name = f"crossdsl::gemv_cuda_v1_warp::{case.name}"
        with nvtx_range(self.torch, nvtx, name), record_function(name):
            self.last = self.module.decode_gemv_warp(case.x, case.weight, case.bias, case.layout)

    def run_cublas_case(self, case: GemvCase, *, nvtx: bool) -> None:
        from torch.profiler import record_function

        name = f"crossdsl::gemv_cublas::{case.name}"
        with nvtx_range(self.torch, nvtx, name), record_function(name):
            self.last = self.module.decode_gemv_cublas(case.x, case.weight, case.bias, case.layout)

    def run_cublaslt_case(self, case: GemvCase, *, nvtx: bool) -> None:
        from torch.profiler import record_function

        name = f"crossdsl::gemv_cublaslt::{case.name}"
        with nvtx_range(self.torch, nvtx, name), record_function(name):
            self.last = self.module.decode_gemv_cublaslt(case.x, case.weight, case.bias, case.layout)

    def run_pytorch_case(self, case: GemvCase, *, nvtx: bool) -> None:
        from crossdsl_kernels.references.gemv import decode_linear_reference
        from torch.profiler import record_function

        name = f"crossdsl::gemv_pytorch_ref::{case.name}"
        with nvtx_range(self.torch, nvtx, name), record_function(name):
            self.last = decode_linear_reference(
                case.x,
                case.weight,
                case.bias_ref,
                out_dtype=self.torch.float32,
                weight_layout=case.layout,
            )

    def run_once(self, *, mode: str, nvtx: bool) -> None:
        for case in self.cases:
            if mode in {"cuda", "both", "all"}:
                self.run_cuda_case(case, nvtx=nvtx)
            if mode in {"warp", "all"}:
                self.run_warp_case(case, nvtx=nvtx)
            if mode in {"cublas", "all"}:
                self.run_cublas_case(case, nvtx=nvtx)
            if mode in {"cublaslt", "all"}:
                self.run_cublaslt_case(case, nvtx=nvtx)
            if mode in {"pytorch", "both", "all"}:
                self.run_pytorch_case(case, nvtx=nvtx)

    def event_benchmark(self, *, mode: str, warmup: int, repeat: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for case in self.cases:
            row: dict[str, Any] = {"name": case.name}
            if mode in {"cuda", "both", "all"}:
                row["cuda_v0"] = time_cuda_ms(lambda case=case: self.run_cuda_case(case, nvtx=False), warmup, repeat, self.torch)
            if mode in {"warp", "all"}:
                row["cuda_v1_warp"] = time_cuda_ms(lambda case=case: self.run_warp_case(case, nvtx=False), warmup, repeat, self.torch)
            if mode in {"cublas", "all"}:
                row["cublas"] = time_cuda_ms(lambda case=case: self.run_cublas_case(case, nvtx=False), warmup, repeat, self.torch)
            if mode in {"cublaslt", "all"}:
                row["cublaslt"] = time_cuda_ms(lambda case=case: self.run_cublaslt_case(case, nvtx=False), warmup, repeat, self.torch)
            if mode in {"pytorch", "both", "all"}:
                row["pytorch_ref"] = time_cuda_ms(lambda case=case: self.run_pytorch_case(case, nvtx=False), warmup, repeat, self.torch)
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
    parser.add_argument("--cases", default="hidden_square_nk_m1,hidden_square_m16")
    parser.add_argument("--mode", choices=["cuda", "warp", "cublas", "cublaslt", "pytorch", "both", "all"], default="cuda")
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
    parser.add_argument("--verbose-extension", action="store_true")
    args = parser.parse_args()

    import torch

    catalog = load_catalog(args.catalog)
    selected_cases = select_cases(catalog, args.cases)
    result: dict[str, Any] = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "torch": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "profiler": args.profiler,
        "mode": args.mode,
        "catalog": str(args.catalog.relative_to(ROOT) if args.catalog.is_relative_to(ROOT) else args.catalog),
        "cases": [str(case["name"]) for case in selected_cases],
        "iterations": args.iterations,
    }
    if not torch.cuda.is_available():
        result.update({"status": "SKIP", "detail": "torch.cuda.is_available() is false"})
        write_json(args.json, result)
        print(json.dumps(result, indent=2))
        return

    torch.cuda.set_device(0)
    result.update(
        {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    )
    module = load_extension(torch, args.verbose_extension)
    result.update(
        {
            "torch_extensions_dir": os.environ["TORCH_EXTENSIONS_DIR"],
            "torch_cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"],
            "max_jobs": os.environ["MAX_JOBS"],
        }
    )
    workload = GemvProfilerWorkload(torch, module, selected_cases, seed=args.seed)
    result["correctness"] = workload.check_correctness()
    if any(row["status"] != "PASS" for row in result["correctness"]):
        result["status"] = "FAIL"
        write_json(args.json, result)
        print(json.dumps(result, indent=2))
        raise SystemExit(1)

    if args.event_repeat > 0:
        result["event_benchmark"] = workload.event_benchmark(
            mode=args.mode,
            warmup=args.event_warmup,
            repeat=args.event_repeat,
        )

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
        key=lambda event: float(
            getattr(event, "self_cuda_time_total", getattr(event, "self_device_time_total", 0.0))
        ),
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
