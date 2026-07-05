#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_PY = ROOT / "src"
RMSNORM_SRC = ROOT / "csrc" / "rmsnorm"
if str(SRC_PY) not in sys.path:
    sys.path.insert(0, str(SRC_PY))


WORKLOADS = {
    "rmsnorm_ref",
    "decode_linear_ref",
    "rope_kv_ref",
    "rmsnorm_cuda",
}


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def load_rmsnorm_extension(torch: Any, verbose: bool) -> Any:
    from torch.utils.cpp_extension import load

    ext_dir = ROOT / "results" / "build" / "torch_extensions"
    ext_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ext_dir))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", "4")
    name = safe_name(f"crossdsl_rmsnorm_{platform.python_version()}_{torch.__version__}")
    return load(
        name=name,
        sources=[str(RMSNORM_SRC / "rmsnorm_ext.cpp"), str(RMSNORM_SRC / "rmsnorm_kernel.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=verbose,
    )


def parse_workloads(text: str) -> list[str]:
    if text == "all":
        return ["rmsnorm_ref", "decode_linear_ref", "rope_kv_ref", "rmsnorm_cuda"]
    if text == "pytorch_refs":
        return ["rmsnorm_ref", "decode_linear_ref", "rope_kv_ref"]
    names = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(names) - WORKLOADS)
    if unknown:
        raise ValueError(f"unknown workload(s): {unknown}")
    return names


def nvtx_range(torch: Any, enabled: bool, name: str):
    class _Range:
        def __enter__(self) -> None:
            if enabled:
                torch.cuda.nvtx.range_push(name)

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            if enabled:
                torch.cuda.nvtx.range_pop()

    return _Range()


class KernelWorkload:
    def __init__(self, torch: Any, workloads: list[str], *, verbose_extension: bool = False) -> None:
        from crossdsl_kernels.references.gemv import decode_linear_reference
        from crossdsl_kernels.references.rmsnorm import fused_residual_rmsnorm_reference
        from crossdsl_kernels.references.rope_kv import rope_gqa_paged_kv_append_reference

        self.torch = torch
        self.workloads = workloads
        self.decode_linear_reference = decode_linear_reference
        self.fused_residual_rmsnorm_reference = fused_residual_rmsnorm_reference
        self.rope_gqa_paged_kv_append_reference = rope_gqa_paged_kv_append_reference
        self.module = load_rmsnorm_extension(torch, verbose_extension) if "rmsnorm_cuda" in workloads else None
        self.last: Any = None

        gen = torch.Generator(device="cpu").manual_seed(20260703)
        self.eps = 1.0e-5
        self.rms_x = torch.randn(16, 4096, generator=gen, dtype=torch.float32, device="cpu").cuda()
        self.rms_residual = torch.randn(16, 4096, generator=gen, dtype=torch.float32, device="cpu").cuda()
        self.rms_weight = torch.randn(4096, generator=gen, dtype=torch.float32, device="cpu").cuda()

        self.linear_x = torch.randn(8, 1024, generator=gen, dtype=torch.float32, device="cpu").cuda()
        self.linear_w = torch.randn(1024, 2048, generator=gen, dtype=torch.float32, device="cpu").cuda()
        self.linear_b = torch.randn(2048, generator=gen, dtype=torch.float32, device="cpu").cuda()

        tokens, hq, hkv, dim = 4, 8, 2, 64
        self.rope_page_size = 16
        self.rope_dim = dim
        self.rope_q = torch.randn(tokens, hq, dim, generator=gen, dtype=torch.float32, device="cpu").cuda()
        self.rope_k = torch.randn(tokens, hkv, dim, generator=gen, dtype=torch.float32, device="cpu").cuda()
        self.rope_v = torch.randn(tokens, hkv, dim, generator=gen, dtype=torch.float32, device="cpu").cuda()
        self.rope_positions = torch.tensor([0, 5, 16, 17], dtype=torch.long, device="cuda")
        self.rope_page_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.long, device="cuda")
        self.rope_sequence_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long, device="cuda")
        theta = torch.arange(32, dtype=torch.float32, device="cuda").unsqueeze(1) / 100.0
        freqs = torch.arange(dim // 2, dtype=torch.float32, device="cuda").unsqueeze(0) + 1.0
        self.rope_cos = torch.cos(theta * freqs)
        self.rope_sin = torch.sin(theta * freqs)
        self.rope_sentinel = -999.0
        self.rope_k_cache = torch.full((5, self.rope_page_size, hkv, dim), self.rope_sentinel, device="cuda")
        self.rope_v_cache = torch.full((5, self.rope_page_size, hkv, dim), self.rope_sentinel, device="cuda")

    def check_correctness(self) -> dict[str, Any]:
        torch = self.torch
        checks: dict[str, Any] = {}
        with torch.no_grad():
            if "rmsnorm_ref" in self.workloads or "rmsnorm_cuda" in self.workloads:
                y_ref, residual_ref = self.fused_residual_rmsnorm_reference(
                    self.rms_x, self.rms_residual, self.rms_weight, self.eps
                )
                manual_r = self.rms_x + self.rms_residual
                manual_y = manual_r * torch.rsqrt((manual_r * manual_r).mean(dim=-1, keepdim=True) + self.eps) * self.rms_weight
                checks["rmsnorm_ref"] = {
                    "y_max_abs_err": float((y_ref - manual_y).abs().max().item()),
                    "residual_max_abs_err": float((residual_ref - manual_r).abs().max().item()),
                }
                if self.module is not None:
                    y_cuda, residual_cuda = self.module.fused_residual_rmsnorm(
                        self.rms_x, self.rms_residual, self.rms_weight, self.eps
                    )
                    checks["rmsnorm_cuda"] = {
                        "y_max_abs_err": float((y_cuda - y_ref).abs().max().item()),
                        "residual_max_abs_err": float((residual_cuda - residual_ref).abs().max().item()),
                    }
            if "decode_linear_ref" in self.workloads:
                out = self.decode_linear_reference(self.linear_x, self.linear_w, self.linear_b, out_dtype=torch.float32)
                manual = self.linear_x @ self.linear_w + self.linear_b
                checks["decode_linear_ref"] = {"max_abs_err": float((out - manual).abs().max().item())}
            if "rope_kv_ref" in self.workloads:
                self.rope_k_cache.fill_(self.rope_sentinel)
                self.rope_v_cache.fill_(self.rope_sentinel)
                q_out = self.rope_gqa_paged_kv_append_reference(
                    self.rope_q,
                    self.rope_k,
                    self.rope_v,
                    self.rope_cos,
                    self.rope_sin,
                    self.rope_positions,
                    self.rope_page_table,
                    self.rope_sequence_ids,
                    self.rope_k_cache,
                    self.rope_v_cache,
                    page_size=self.rope_page_size,
                    rope_dim=self.rope_dim,
                    interleaved=False,
                    kv_layout="NHD",
                )
                torch.cuda.synchronize()
                checks["rope_kv_ref"] = {
                    "q_shape": list(q_out.shape),
                    "untouched_page0_ok": bool(torch.all(self.rope_k_cache[0] == self.rope_sentinel).item()),
                    "v_written_ok": bool(
                        torch.allclose(self.rope_v_cache[1, 0], self.rope_v[0])
                        and torch.allclose(self.rope_v_cache[1, 5], self.rope_v[1])
                        and torch.allclose(self.rope_v_cache[4, 1], self.rope_v[3])
                    ),
                }
        torch.cuda.synchronize()
        return checks

    def run_one(self, name: str, *, nvtx: bool) -> None:
        torch = self.torch
        from torch.profiler import record_function

        with nvtx_range(torch, nvtx, f"crossdsl::{name}"), record_function(f"crossdsl::{name}"):
            if name == "rmsnorm_ref":
                self.last = self.fused_residual_rmsnorm_reference(
                    self.rms_x, self.rms_residual, self.rms_weight, self.eps
                )
            elif name == "decode_linear_ref":
                self.last = self.decode_linear_reference(
                    self.linear_x, self.linear_w, self.linear_b, out_dtype=torch.float32
                )
            elif name == "rope_kv_ref":
                self.last = self.rope_gqa_paged_kv_append_reference(
                    self.rope_q,
                    self.rope_k,
                    self.rope_v,
                    self.rope_cos,
                    self.rope_sin,
                    self.rope_positions,
                    self.rope_page_table,
                    self.rope_sequence_ids,
                    self.rope_k_cache,
                    self.rope_v_cache,
                    page_size=self.rope_page_size,
                    rope_dim=self.rope_dim,
                    interleaved=False,
                    kv_layout="NHD",
                )
            elif name == "rmsnorm_cuda":
                if self.module is None:
                    raise RuntimeError("rmsnorm_cuda requested without extension module")
                self.last = self.module.fused_residual_rmsnorm(
                    self.rms_x, self.rms_residual, self.rms_weight, self.eps
                )
            else:  # pragma: no cover - guarded by parse_workloads
                raise ValueError(name)

    def run_bundle(self, *, nvtx: bool) -> None:
        for name in self.workloads:
            self.run_one(name, nvtx=nvtx)


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
    parser.add_argument("--workload", default="all")
    parser.add_argument("--iterations", type=int, default=6)
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
    parser.add_argument("--verbose-extension", action="store_true")
    args = parser.parse_args()

    import torch

    result: dict[str, Any] = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "torch": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "profiler": args.profiler,
        "workloads": parse_workloads(args.workload),
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
            "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR", ""),
            "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
            "max_jobs": os.environ.get("MAX_JOBS", ""),
        }
    )
    workload = KernelWorkload(torch, result["workloads"], verbose_extension=args.verbose_extension)
    result.update(
        {
            "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR", ""),
            "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
            "max_jobs": os.environ.get("MAX_JOBS", ""),
        }
    )
    result["correctness"] = workload.check_correctness()

    if args.profiler == "none":
        for _ in range(args.iterations):
            workload.run_bundle(nvtx=args.nvtx)
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

    trace_dir = args.trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    total_steps = (args.wait + args.warmup + args.active) * args.repeat
    result["torch_profiler"] = {
        "trace_dir": str(trace_dir),
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

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        schedule=schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=args.repeat),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        with_flops=True,
    ) as prof:
        for _ in range(total_steps):
            workload.run_bundle(nvtx=args.nvtx)
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
    top_events = [event_to_dict(event) for event in events[:80]]
    result["torch_profiler"]["sort_field"] = sort_field
    result["torch_profiler"]["top_events"] = top_events
    result["status"] = "PASS"
    write_json(args.json, result)
    print(json.dumps(result, indent=2))
    print(table)


if __name__ == "__main__":
    main()
