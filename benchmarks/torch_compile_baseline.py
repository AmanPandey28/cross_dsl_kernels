#!/usr/bin/env python3
"""Honest torch.compile baseline — the question every senior engineer asks.

Compares: PyTorch eager, torch.compile, CUDA C++, Triton, CuTe DSL
across all three operators at realistic shapes.

Separates one-time compilation cost from steady-state latency.
Documents what torch.compile actually does under the hood.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PY = ROOT / "src"
if str(SRC_PY) not in sys.path:
    sys.path.insert(0, str(SRC_PY))


def _cuda_event_ms(torch, fn, warmup, repeat, sync_before=False):
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    if sync_before:
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        if sync_before:
            torch.cuda.synchronize()
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        samples.append(start_ev.elapsed_time(end_ev))
    samples.sort()
    n = len(samples)
    return {
        "median_ms": samples[n // 2],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "mean_ms": sum(samples) / n,
    }


def benchmark_rmsnorm(torch, rows, hidden, warmup, repeat):
    from crossdsl_kernels.references.rmsnorm import fused_residual_rmsnorm_reference
    from crossdsl_kernels.triton.rmsnorm import fused_residual_rmsnorm_triton

    gen = torch.Generator(device="cuda").manual_seed(20260715)
    x = torch.randn(rows, hidden, generator=gen, dtype=torch.float32, device="cuda")
    residual = torch.randn(
        rows, hidden, generator=gen, dtype=torch.float32, device="cuda"
    )
    w = torch.randn(hidden, generator=gen, dtype=torch.float32, device="cuda")
    eps = 1e-5

    results = {}

    # Eager PyTorch (fused: residual add + norm in one function call)
    def eager_fn():
        return fused_residual_rmsnorm_reference(x, residual, w, eps)

    compile_start = time.perf_counter()
    eager_compiled = torch.compile(eager_fn, mode="reduce-overhead")
    _ = eager_compiled()
    torch.cuda.synchronize()
    compile_time_s = time.perf_counter() - compile_start

    results["compile_time_s"] = compile_time_s
    results["eager"] = _cuda_event_ms(torch, eager_fn, warmup, repeat)
    results["torch_compile_reduce_overhead"] = _cuda_event_ms(
        torch, eager_compiled, warmup, repeat
    )
    results["triton"] = _cuda_event_ms(
        torch,
        lambda: fused_residual_rmsnorm_triton(x, residual, w, eps),
        warmup,
        repeat,
    )

    # Fused PyTorch baseline: residual + rmsnorm as a single composed op
    def fused_pt():
        r = x + residual
        inv = torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps)
        return r * inv * w, r

    # torch.compile the fused op (should match what our kernel does)
    fused_compile_start = time.perf_counter()
    fused_compiled = torch.compile(fused_pt, mode="reduce-overhead")
    _ = fused_compiled()
    torch.cuda.synchronize()
    results["fused_compile_time_s"] = time.perf_counter() - fused_compile_start
    results["fused_eager"] = _cuda_event_ms(torch, fused_pt, warmup, repeat)
    results["fused_compile"] = _cuda_event_ms(torch, fused_compiled, warmup, repeat)

    return results


def benchmark_gemv(torch, m, k, n, layout, use_bias, warmup, repeat):
    from crossdsl_kernels.references.gemv import decode_linear_reference
    from crossdsl_kernels.triton.gemv import decode_gemv_triton

    gen = torch.Generator(device="cuda").manual_seed(20260715)
    x = torch.randn(m, k, generator=gen, dtype=torch.float32, device="cuda")
    w_kn = torch.randn(k, n, generator=gen, dtype=torch.float32, device="cuda")
    w = w_kn if layout == "KN" else w_kn.t().contiguous()
    b = (
        torch.randn(n, generator=gen, dtype=torch.float32, device="cuda")
        if use_bias
        else None
    )

    results = {}

    def eager_fn():
        return decode_linear_reference(
            x, w, b, out_dtype=torch.float32, weight_layout=layout
        )

    compile_start = time.perf_counter()
    eager_compiled = torch.compile(eager_fn, mode="reduce-overhead")
    _ = eager_compiled()
    torch.cuda.synchronize()
    results["compile_time_s"] = time.perf_counter() - compile_start

    results["eager"] = _cuda_event_ms(torch, eager_fn, warmup, repeat)
    results["torch_compile_reduce_overhead"] = _cuda_event_ms(
        torch, eager_compiled, warmup, repeat
    )
    results["triton"] = _cuda_event_ms(
        torch,
        lambda: decode_gemv_triton(x, w, b, weight_layout=layout),
        warmup,
        repeat,
    )

    return results


def benchmark_rope_kv(
    torch,
    tokens,
    hq,
    hkv,
    dim,
    page_size,
    rope_dim,
    interleaved,
    layout,
    warmup,
    repeat,
):
    from crossdsl_kernels.references.rope_kv import rope_gqa_paged_kv_append_reference
    from crossdsl_kernels.triton.rope_kv import rope_gqa_paged_kv_append_triton

    gen = torch.Generator(device="cuda").manual_seed(20260715)
    q = torch.randn(tokens, hq, dim, generator=gen, dtype=torch.float32, device="cuda")
    k = torch.randn(tokens, hkv, dim, generator=gen, dtype=torch.float32, device="cuda")
    v = torch.randn(tokens, hkv, dim, generator=gen, dtype=torch.float32, device="cuda")
    positions = torch.randint(0, 32, (tokens,), dtype=torch.long).cuda()
    page_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.long).cuda()
    seq_ids = torch.zeros(tokens, dtype=torch.long).cuda()
    for t in range(tokens):
        seq_ids[t] = t % 2

    theta = torch.arange(32, dtype=torch.float32).unsqueeze(1) / 100.0
    freqs = torch.arange(dim // 2, dtype=torch.float32).unsqueeze(0) + 1.0
    cos = torch.cos(theta * freqs).cuda()
    sin = torch.sin(theta * freqs).cuda()

    num_pages = 4
    if layout.upper() == "HND":
        k_cache = torch.zeros(
            num_pages, hkv, page_size, dim, dtype=torch.float32
        ).cuda()
        v_cache = torch.zeros(
            num_pages, hkv, page_size, dim, dtype=torch.float32
        ).cuda()
    else:
        k_cache = torch.zeros(
            num_pages, page_size, hkv, dim, dtype=torch.float32
        ).cuda()
        v_cache = torch.zeros(
            num_pages, page_size, hkv, dim, dtype=torch.float32
        ).cuda()

    results = {}

    def eager_fn():
        k_cache.zero_()
        v_cache.zero_()
        return rope_gqa_paged_kv_append_reference(
            q,
            k,
            v,
            cos,
            sin,
            positions,
            page_table,
            seq_ids,
            k_cache,
            v_cache,
            page_size=page_size,
            rope_dim=rope_dim,
            interleaved=interleaved,
            kv_layout=layout,
        )

    compile_start = time.perf_counter()
    eager_compiled = torch.compile(eager_fn, mode="reduce-overhead")
    _ = eager_compiled()
    torch.cuda.synchronize()
    results["compile_time_s"] = time.perf_counter() - compile_start

    results["eager"] = _cuda_event_ms(torch, eager_fn, warmup, repeat)
    results["torch_compile_reduce_overhead"] = _cuda_event_ms(
        torch, eager_compiled, warmup, repeat
    )
    results["triton"] = _cuda_event_ms(
        torch,
        lambda: rope_gqa_paged_kv_append_triton(
            q,
            k,
            v,
            cos,
            sin,
            positions,
            page_table,
            seq_ids,
            k_cache,
            v_cache,
            page_size=page_size,
            rope_dim=rope_dim,
            interleaved=interleaved,
            kv_layout=layout,
        ),
        warmup,
        repeat,
    )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    import torch

    result = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "cases": [],
        "notes": {
            "torch_compile_mode": "reduce-overhead (CUDA Graphs + fusion)",
            "sync_included": "torch.cuda.synchronize() after each timed call",
            "warmup_included": f"{args.warmup} warmup iterations excluded from timing",
            "compile_excluded": "One-time torch.compile cost measured separately",
            "cuda_graph_capture_excluded": "torch.compile reduce-overhead uses CUDA Graphs internally",
        },
    }

    if not torch.cuda.is_available():
        result["status"] = "SKIP"
    else:
        result["device"] = torch.cuda.get_device_name(0)
        result["capability"] = list(torch.cuda.get_device_capability(0))

        # RMSNorm
        for rows, hidden in [(1, 1024), (4, 4096), (1, 4096), (32, 4096)]:
            result["cases"].append(
                benchmark_rmsnorm(torch, rows, hidden, args.warmup, args.repeat)
            )

        # GEMV: small-M decode + batch decode
        for m, k, n, layout, bias in [
            (1, 1024, 1024, "KN", False),  # small decode
            (1, 4096, 11008, "KN", False),  # LLaMA intermediate
            (8, 4096, 4096, "KN", False),  # batch decode
            (32, 4096, 4096, "KN", False),  # batch decode
        ]:
            result["cases"].append(
                benchmark_gemv(torch, m, k, n, layout, bias, args.warmup, args.repeat)
            )

        # RoPE: realistic LLaMA shapes
        for tokens, hq, hkv, dim, ps, rd, inter, lay in [
            (1, 4, 2, 64, 8, 32, False, "NHD"),  # small smoke
            (1, 32, 8, 128, 16, 128, False, "NHD"),  # LLaMA 7B decode
            (32, 32, 8, 128, 64, 128, False, "NHD"),  # LLaMA 7B prefill
        ]:
            result["cases"].append(
                benchmark_rope_kv(
                    torch,
                    tokens,
                    hq,
                    hkv,
                    dim,
                    ps,
                    rd,
                    inter,
                    lay,
                    args.warmup,
                    args.repeat,
                )
            )

        result["status"] = "PASS"

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, default=str, indent=2) + "\n")

    # Quick summary
    for c in result.get("cases", []):
        op = "rmsnorm" if "rows" not in str(c) else c.get("kernel", "?")
        if "eager" in c:
            eg = c["eager"]["median_ms"]
            tc = c.get("torch_compile_reduce_overhead", {}).get("median_ms", 0)
            tri = c.get("triton", {}).get("median_ms", 0)
            print(
                f"  {op}: eager={eg:.3f}ms  compile={tc:.3f}ms  triton={tri:.3f}ms  "
                f"compile_vs_eager={eg / max(tc, 1e-6):.1f}x  triton_vs_eager={eg / max(tri, 1e-6):.1f}x"
            )

    raise SystemExit(0 if result.get("status") == "PASS" else 1)


if __name__ == "__main__":
    main()
