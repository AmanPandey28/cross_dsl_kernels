from __future__ import annotations

import functools
import math
import os
import platform
import re
from pathlib import Path
from collections.abc import Callable
from typing import Any

import torch

from crossdsl_kernels.references.gemv import decode_linear_reference
from crossdsl_kernels.references.rmsnorm import fused_residual_rmsnorm_reference
from crossdsl_kernels.references.rope_kv import (
    rope_gqa_paged_kv_append_reference,
    rope_gqa_paged_kv_append_vectorized,
    validate_paged_kv_indices,
)
from crossdsl_kernels.tolerances import fp32_gemv_tolerance


def make_case(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> Any:
        return torch.randn(*shape, generator=gen, dtype=torch.float32).cuda()

    op = spec["op"]
    if op == "rmsnorm":
        rows, hidden = spec["rows"], spec["hidden"]
        x, residual, weight = rand(rows, hidden), rand(rows, hidden), rand(hidden)
        r = x + residual
        inv = torch.rsqrt(r.double().square().mean(-1, keepdim=True) + 1e-5)
        expected = ((r.double() * inv * weight.double()).float(), r)
        return {
            "args": (x, residual, weight, 1e-5),
            "kwargs": {},
            "expected": expected,
            "tolerance": {"max_abs_err": 2e-5, "rel_l2_err": 1e-5},
            "logical_bytes": 4 * (4 * rows * hidden + hidden),
            "flops": 0,
        }
    if op == "gemv":
        m, k, n = spec["m"], spec["k"], spec["n"]
        x, weight = rand(m, k), rand(k, n)
        expected = x.double() @ weight.double()
        if spec["layout"] == "NK":
            weight = weight.t().contiguous()
        bias = rand(n) if spec["bias"] else None
        if bias is not None:
            expected = expected + bias.double()
        return {
            "args": (x, weight, bias),
            "kwargs": {"weight_layout": spec["layout"]},
            "expected": (expected.float(),),
            "tolerance": fp32_gemv_tolerance(k),
            "logical_bytes": 4 * (m * k + k * n + m * n + (n if spec["bias"] else 0)),
            "flops": 2 * m * k * n + (m * n if spec["bias"] else 0),
        }

    tokens, hq, hkv, dim = (
        spec[key] for key in ("tokens", "q_heads", "kv_heads", "head_dim")
    )
    page_size = spec["page_size"]
    # Every active request has its own three pages. Reverse physical order to
    # exercise translation; one final sentinel page is never mapped.
    positions = torch.tensor(
        [page_size - 1 + i % 3 for i in range(tokens)], device="cuda"
    )
    sequences = torch.arange(tokens, device="cuda")
    table = (
        torch.arange(tokens * 3, device="cuda").flip(0).reshape(tokens, 3).contiguous()
    )
    trig = rand(page_size * 3, spec["rope_dim"] // 2)
    q, k, v = rand(tokens, hq, dim), rand(tokens, hkv, dim), rand(tokens, hkv, dim)
    shape = (tokens * 3 + 1, page_size, hkv, dim)
    if spec["layout"] == "HND":
        shape = (tokens * 3 + 1, hkv, page_size, dim)
    k_cache, v_cache = (torch.full(shape, -999.0, device="cuda") for _ in range(2))
    args = (
        q,
        k,
        v,
        trig.cos(),
        trig.sin(),
        positions,
        table,
        sequences,
        k_cache,
        v_cache,
    )
    kwargs = {
        "page_size": page_size,
        "rope_dim": spec["rope_dim"],
        "interleaved": spec["interleaved"],
        "kv_layout": spec["layout"],
    }
    validate_paged_kv_indices(
        positions,
        table,
        sequences,
        page_size=page_size,
        num_cache_pages=shape[0],
        max_position=trig.shape[0],
    )
    cpu_args = tuple(t.cpu() for t in args)
    cpu_q = rope_gqa_paged_kv_append_reference(*cpu_args, **kwargs)
    expected = tuple(t.cuda() for t in (cpu_q, cpu_args[-2], cpu_args[-1]))
    # Count unique Q/K/V payload reads and writes, unique trig-table rows,
    # and one position, sequence ID, and page-table entry per token.
    logical_bytes = (
        8 * tokens * dim * (hq + 2 * hkv)
        + 8 * tokens * (spec["rope_dim"] // 2)
        + 24 * tokens
    )
    return {
        "args": args,
        "kwargs": kwargs,
        "expected": expected,
        "tolerance": {"max_abs_err": 1e-6, "rel_l2_err": 1e-5},
        "logical_bytes": logical_bytes,
        "flops": 0,
    }


@functools.lru_cache(maxsize=3)
def cuda_module(op: str) -> Any:
    from torch.utils.cpp_extension import load

    if op not in ("rmsnorm", "gemv", "rope_kv"):
        raise ValueError(f"unknown CUDA workload: {op}")
    root = Path(__file__).resolve().parents[1]
    directory = root / "results/build/torch_extensions"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(directory))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", "4")
    name = re.sub(
        r"[^A-Za-z0-9_]+", "_",
        f"crossdsl_{op}_{platform.python_version()}_{torch.__version__}",
    )
    return load(
        name=name,
        sources=[str(root / f"csrc/{op}/{op}_ext.cpp"),
                 str(root / f"csrc/{op}/{op}_kernel.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcublas", "-lcublasLt"] if op == "gemv" else [],
        verbose=False,
    )


def prepare_cute(
    op: str, case: dict[str, Any], fast: bool = False
) -> Callable[[], Any]:
    from crossdsl_kernels.cute.launch import prepare_launch

    args, kwargs = case["args"], case["kwargs"]
    if op == "rmsnorm":
        from crossdsl_kernels.cute.rmsnorm import (
            _launch,
            _launch_fast,
            _launch_resident,
        )

        x, residual, weight, eps = args
        y, r = torch.empty_like(x), torch.empty_like(residual)
        tensors = (x.flatten(), residual.flatten(), weight, y.flatten(), r.flatten())
        run = prepare_launch(
            (_launch_resident if x.shape[1] <= 8192 else _launch_fast)
            if fast
            else _launch,
            tensors,
            (x.shape[1], eps, x.shape[0]),
            constexpr_scalars=(0,) if fast and x.shape[1] <= 8192 else (),
        )

        def rmsnorm() -> tuple[Any, Any]:
            run()
            return y, r

        return rmsnorm
    if op == "gemv":
        from crossdsl_kernels.cute.gemv import _load_kernels

        _, _, launcher = _load_kernels()
        x, weight, bias = args
        m, k = x.shape
        nk = kwargs["weight_layout"] == "NK"
        n = weight.shape[0] if nk else weight.shape[1]
        y = torch.empty((m, n), device=x.device)
        if fast:
            from crossdsl_kernels.cute.gemv_rows import _launch

            run = prepare_launch(
                _launch,
                tuple(
                    t.flatten() for t in (x, weight, bias if bias is not None else y, y)
                ),
                (m, k, n, int(nk), int(bias is not None)),
            )

            def gemv_rows() -> Any:
                run()
                return y

            return gemv_rows
        run = prepare_launch(
            launcher,
            (x, weight, bias if bias is not None else y, y),
            (m * n, k, n, int(nk), int(bias is not None)),
        )

        def gemv() -> Any:
            run()
            return y

        return gemv
    from crossdsl_kernels.cute.rope_kv import (
        _launch_append_kv_hnd,
        _launch_append_kv_nhd,
        _launch_rotate_q,
    )

    q, k, v, cos, sin, positions, table, sequences, kc, vc = args
    tokens, hq, dim = q.shape
    hkv = k.shape[1]
    q_out = torch.empty_like(q)
    if fast:
        from crossdsl_kernels.cute.rope_kv_fused import _launch

        fused = prepare_launch(
            _launch,
            tuple(t.flatten() for t in (*args, q_out)),
            (
                tokens,
                hq,
                hkv,
                dim,
                cos.shape[1],
                table.shape[1],
                kwargs["page_size"],
                kwargs["rope_dim"],
                int(kwargs["interleaved"]),
                int(kwargs["kv_layout"] == "HND"),
            ),
        )

        def rope_fused() -> Any:
            fused()
            return q_out

        return rope_fused
    rotate = prepare_launch(
        _launch_rotate_q,
        tuple(t.flatten() for t in (q, cos, sin, positions, q_out)),
        (tokens, hq, dim, cos.shape[1], kwargs["rope_dim"], int(kwargs["interleaved"])),
    )
    launcher = (
        _launch_append_kv_hnd if kwargs["kv_layout"] == "HND" else _launch_append_kv_nhd
    )
    append = prepare_launch(
        launcher,
        tuple(
            t.flatten() for t in (k, v, cos, sin, positions, table, sequences, kc, vc)
        ),
        (
            tokens,
            hkv,
            dim,
            cos.shape[1],
            table.shape[1],
            kwargs["page_size"],
            kwargs["rope_dim"],
            int(kwargs["interleaved"]),
        ),
    )

    def rope() -> Any:
        rotate()
        append()
        return q_out

    return rope


def prepare_backend(op: str, backend: str, case: dict[str, Any]) -> Callable[[], Any]:
    args, kwargs = case["args"], case["kwargs"]
    if backend in {"pytorch", "torch_compile"}:
        reference = {
            "rmsnorm": fused_residual_rmsnorm_reference,
            "gemv": decode_linear_reference,
            "rope_kv": rope_gqa_paged_kv_append_vectorized,
        }[op]
        run = functools.partial(reference, *args, **kwargs)
        if backend == "torch_compile":
            # A shape sweep must not share the wrapper's eight-entry Dynamo budget.
            torch.compiler.reset()
            return torch.compile(
                run, fullgraph=True, options={"triton.cudagraphs": False}
            )
        return run
    if backend in {"cute", "cute_fast"}:
        return prepare_cute(op, case, fast=backend == "cute_fast")
    if backend == "triton_fast":
        if op == "rmsnorm":
            from crossdsl_kernels.triton.rmsnorm import fused_residual_rmsnorm_triton

            return functools.partial(
                fused_residual_rmsnorm_triton,
                *args,
                num_warps=2 if args[0].shape[1] <= 2048 else 4,
            )
        if op == "gemv":
            from crossdsl_kernels.triton.gemv_rows import (
                decode_gemv_triton_rows as kernel,
            )
        else:
            from crossdsl_kernels.triton.rope_kv_fused import (
                rope_gqa_paged_kv_append_triton_fused as kernel,
            )
        return functools.partial(kernel, *args, **kwargs)
    if backend == "triton":
        if op == "rmsnorm":
            from crossdsl_kernels.triton.rmsnorm import (
                fused_residual_rmsnorm_triton as kernel,
            )
        elif op == "gemv":
            from crossdsl_kernels.triton.gemv import decode_gemv_triton_tiled as kernel
        else:
            from crossdsl_kernels.triton.rope_kv import (
                rope_gqa_paged_kv_append_triton as kernel,
            )
        return functools.partial(kernel, *args, **kwargs)
    module = cuda_module(op)
    if op == "rmsnorm":
        return functools.partial(
            module.fused_residual_rmsnorm, *args, fast=backend == "cuda_fast"
        )
    if op == "gemv":
        x, weight, bias = args
        bias_tensor = bias if bias is not None else torch.empty(0, device=x.device)
        kernel = {
            "cuda": module.decode_gemv_warp,
            "cuda_fast": module.decode_gemv_rows,
            "cublas": module.decode_gemv_cublas,
            "cublaslt": module.decode_gemv_cublaslt,
        }[backend]
        return functools.partial(
            kernel, x, weight, bias_tensor, kwargs["weight_layout"]
        )
    return functools.partial(
        module.rope_gqa_paged_kv_append,
        *args,
        kwargs["page_size"],
        kwargs["rope_dim"],
        kwargs["interleaved"],
        kwargs["kv_layout"],
        fused=backend == "cuda_fast",
    )


def check_output(op: str, output: Any, case: dict[str, Any]) -> dict[str, Any]:
    if op == "rmsnorm":
        outputs = output
    elif op == "gemv":
        outputs = (output,)
    else:
        outputs = (output, *case["args"][-2:])
    if len(outputs) != len(case["expected"]):
        raise ValueError("backend returned an unexpected number of outputs")
    errors = []
    passed = True
    for index, (actual, expected) in enumerate(zip(outputs, case["expected"])):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError("backend output shape or dtype differs from the contract")
        diff = actual.double() - expected.double()
        abs_err = float(diff.abs().max())
        rel_l2 = float(diff.norm() / expected.double().norm().clamp_min(1e-12))
        finite = math.isfinite(abs_err) and math.isfinite(rel_l2)
        exact = bool(torch.equal(actual, expected))
        ok = (
            finite
            and abs_err <= case["tolerance"]["max_abs_err"]
            and rel_l2 <= case["tolerance"]["rel_l2_err"]
        )
        if op == "rmsnorm" and index == 1:
            ok = exact
        if op == "rope_kv":
            sentinel_matches = (
                bool(torch.equal(actual == -999.0, expected == -999.0))
                if index
                else True
            )
            ok = ok and sentinel_matches and (exact if index == 2 else True)
        passed = passed and ok
        errors.append(
            {
                "output": index,
                "max_abs_err": abs_err if finite else None,
                "rel_l2_err": rel_l2 if finite else None,
                "finite": finite,
                "exact": exact,
            }
        )
    return {
        "status": "PASS" if passed else "FAIL",
        "errors": errors,
        "tolerance": case["tolerance"],
    }
