#!/usr/bin/env python3
"""CUDA Graphs overhead reduction benchmark for CuTe DSL kernels.

Compares un-graphed vs graphed CuTe kernel launch latency with proper
CUDA event timing, isolating Python/CUDA-driver overhead.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PY = ROOT / "src"
if str(SRC_PY) not in sys.path:
    sys.path.insert(0, str(SRC_PY))


def _cuda_event_ms(torch, fn, warmup, repeat):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    n = len(samples)
    return {
        "median_ms": samples[n // 2],
        "min_ms": samples[0],
        "mean_ms": sum(samples) / n,
        "p10_ms": samples[max(0, n // 10)],
        "p90_ms": samples[min(n - 1, 9 * n // 10)],
        "samples_ms": samples,
    }


def benchmark_rmsnorm(torch, rows, hidden, warmup, repeat):
    from crossdsl_kernels.cute.rmsnorm import fused_residual_rmsnorm_cute

    gen = torch.Generator(device="cuda").manual_seed(20260705)
    x = torch.randn(rows, hidden, generator=gen, dtype=torch.float32, device="cuda")
    residual = torch.randn(
        rows, hidden, generator=gen, dtype=torch.float32, device="cuda"
    )
    weight = torch.randn(hidden, generator=gen, dtype=torch.float32, device="cuda")

    def ungraph():
        fused_residual_rmsnorm_cute(x, residual, weight, eps=1e-5)

    ungraph_result = _cuda_event_ms(torch, ungraph, warmup, repeat)

    x_s = x.clone()
    r_s = residual.clone()
    w_s = weight.clone()

    y_s, ro_s = fused_residual_rmsnorm_cute(x_s, r_s, w_s, eps=1e-5)
    del y_s, ro_s

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fused_residual_rmsnorm_cute(x_s, r_s, w_s, eps=1e-5)

    def graph():
        x_s.copy_(x)
        r_s.copy_(residual)
        g.replay()

    graph_result = _cuda_event_ms(torch, graph, warmup, repeat)
    return {
        "kernel": "rmsnorm",
        "shape": [rows, hidden],
        "ungraphed": ungraph_result,
        "graphed": graph_result,
        "speedup_median": ungraph_result["median_ms"]
        / max(graph_result["median_ms"], 1e-12),
    }


def benchmark_gemv(torch, m, k, n, layout, use_bias, warmup, repeat):
    from crossdsl_kernels.cute.gemv import decode_gemv_cute

    gen = torch.Generator(device="cuda").manual_seed(20260705)
    x = torch.randn(m, k, generator=gen, dtype=torch.float32, device="cuda")
    w_kn = torch.randn(k, n, generator=gen, dtype=torch.float32, device="cuda")
    weight = w_kn if layout == "KN" else w_kn.t().contiguous()
    bias = (
        torch.randn(n, generator=gen, dtype=torch.float32, device="cuda")
        if use_bias
        else None
    )

    def ungraph():
        out = decode_gemv_cute(x, weight, bias, weight_layout=layout)
        out[0, 0].item()

    ungraph_result = _cuda_event_ms(torch, ungraph, warmup, repeat)

    x_s = x.clone()
    w_s = weight.clone()
    b_s = bias.clone() if bias is not None else None

    decode_gemv_cute(x_s, w_s, b_s, weight_layout=layout)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        decode_gemv_cute(x_s, w_s, b_s, weight_layout=layout)

    def graph():
        x_s.copy_(x)
        w_s.copy_(weight)
        if b_s is not None:
            b_s.copy_(bias)
        g.replay()

    graph_result = _cuda_event_ms(torch, graph, warmup, repeat)
    return {
        "kernel": "gemv",
        "shape": [m, k, n],
        "layout": layout,
        "bias": use_bias,
        "ungraphed": ungraph_result,
        "graphed": graph_result,
        "speedup_median": ungraph_result["median_ms"]
        / max(graph_result["median_ms"], 1e-12),
    }


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
    from crossdsl_kernels.cute.rope_kv import rope_gqa_paged_kv_append_cute

    gen = torch.Generator(device="cuda").manual_seed(20260705)
    q = torch.randn(tokens, hq, dim, generator=gen, dtype=torch.float32).cuda()
    k = torch.randn(tokens, hkv, dim, generator=gen, dtype=torch.float32).cuda()
    v = torch.randn(tokens, hkv, dim, generator=gen, dtype=torch.float32).cuda()
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

    def ungraph():
        k_cache.zero_()
        v_cache.zero_()
        rope_gqa_paged_kv_append_cute(
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

    ungraph_result = _cuda_event_ms(torch, ungraph, warmup, repeat)

    q_s = q.clone()
    k_s = k.clone()
    v_s = v.clone()
    cos_s = cos.clone()
    sin_s = sin.clone()
    pos_s = positions.clone()
    pt_s = page_table.clone()
    sid_s = seq_ids.clone()
    kc_s = k_cache.clone()
    vc_s = v_cache.clone()

    rope_gqa_paged_kv_append_cute(
        q_s,
        k_s,
        v_s,
        cos_s,
        sin_s,
        pos_s,
        pt_s,
        sid_s,
        kc_s,
        vc_s,
        page_size=page_size,
        rope_dim=rope_dim,
        interleaved=interleaved,
        kv_layout=layout,
    )
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        rope_gqa_paged_kv_append_cute(
            q_s,
            k_s,
            v_s,
            cos_s,
            sin_s,
            pos_s,
            pt_s,
            sid_s,
            kc_s,
            vc_s,
            page_size=page_size,
            rope_dim=rope_dim,
            interleaved=interleaved,
            kv_layout=layout,
        )

    def graph():
        q_s.copy_(q)
        k_s.copy_(k)
        v_s.copy_(v)
        cos_s.copy_(cos)
        sin_s.copy_(sin)
        pos_s.copy_(positions)
        pt_s.copy_(page_table)
        sid_s.copy_(seq_ids)
        kc_s.copy_(k_cache)
        vc_s.copy_(v_cache)
        g.replay()

    graph_result = _cuda_event_ms(torch, graph, warmup, repeat)
    return {
        "kernel": "rope_kv",
        "shape": [tokens, hq, hkv, dim],
        "page_size": page_size,
        "rope_dim": rope_dim,
        "interleaved": interleaved,
        "layout": layout,
        "ungraphed": ungraph_result,
        "graphed": graph_result,
        "speedup_median": ungraph_result["median_ms"]
        / max(graph_result["median_ms"], 1e-12),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    result = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "cases": [],
    }

    import cutlass
    import torch

    result.update(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cutlass": getattr(cutlass, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    )
    if not torch.cuda.is_available():
        result["status"] = "SKIP"
        result["detail"] = "torch.cuda.is_available() == false"
    else:
        result["device"] = torch.cuda.get_device_name(0)
        result["capability"] = list(torch.cuda.get_device_capability(0))

        cases = []
        for rows, hidden in [(1, 1024), (4, 4096)]:
            try:
                cases.append(
                    benchmark_rmsnorm(torch, rows, hidden, args.warmup, args.repeat)
                )
            except Exception as exc:
                cases.append(
                    {"kernel": "rmsnorm", "shape": [rows, hidden], "error": str(exc)}
                )

        for m, k, n, layout, use_bias in [
            (1, 1024, 1024, "KN", False),
            (1, 4096, 11008, "KN", False),
        ]:
            try:
                cases.append(
                    benchmark_gemv(
                        torch, m, k, n, layout, use_bias, args.warmup, args.repeat
                    )
                )
            except Exception as exc:
                cases.append({"kernel": "gemv", "shape": [m, k, n], "error": str(exc)})

        for tokens, hq, hkv, dim, ps, rd, inter, lay in [
            (1, 4, 2, 64, 8, 32, False, "NHD"),
            (1, 8, 4, 128, 16, 64, True, "NHD"),
        ]:
            try:
                cases.append(
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
            except Exception as exc:
                cases.append(
                    {
                        "kernel": "rope_kv",
                        "shape": [tokens, hq, hkv, dim],
                        "error": str(exc),
                    }
                )

        result["cases"] = cases
        statuses = []
        for c in cases:
            if "error" in c:
                statuses.append("FAIL")
            else:
                sp = c.get("speedup_median", 0)
                statuses.append("PASS" if sp > 0.5 else "INFO")
        result["status"] = "PASS" if all(s == "PASS" for s in statuses) else "SOME_OK"

    args.json.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, default=str, indent=2)
    args.json.write_text(text + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k in ("status", "device", "capability")}
        )
    )
    for c in result.get("cases", []):
        sp = c.get("speedup_median", 0)
        err = c.get("error", "")
        print(
            f"  {c.get('kernel', '?'):10s} {c.get('shape', '')!s:30s} "
            + (f"speedup: {sp:,.1f}x" if sp else f"ERROR: {err[:80]}")
        )

    raise SystemExit(0 if result.get("status") in {"PASS", "SOME_OK", "SKIP"} else 1)


if __name__ == "__main__":
    main()
