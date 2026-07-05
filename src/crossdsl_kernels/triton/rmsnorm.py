from __future__ import annotations

from functools import lru_cache
from typing import Any


def _next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


@lru_cache(maxsize=1)
def _load_kernel() -> Any:
    import triton
    import triton.language as tl

    @triton.jit
    def _fused_residual_rmsnorm_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        y_ptr,
        residual_out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < hidden
        row_offsets = row * hidden + offsets

        x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        r = x + residual
        sum_squares = tl.sum(tl.where(mask, r * r, 0.0), axis=0)
        inv_rms = tl.rsqrt(sum_squares / hidden + eps)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = r * inv_rms * weight

        tl.store(residual_out_ptr + row_offsets, r, mask=mask)
        tl.store(y_ptr + row_offsets, y, mask=mask)

    return triton, _fused_residual_rmsnorm_kernel


def fused_residual_rmsnorm_triton(x: Any, residual: Any, weight: Any, eps: float) -> tuple[Any, Any]:
    import torch

    if not x.is_cuda or not residual.is_cuda or not weight.is_cuda:
        raise ValueError("x, residual, and weight must be CUDA tensors")
    if x.dtype != torch.float32 or residual.dtype != torch.float32 or weight.dtype != torch.float32:
        raise ValueError("Triton RMSNorm v0 supports float32 tensors only")
    if x.dim() != 2:
        raise ValueError("x must be shaped [rows, hidden]")
    if residual.shape != x.shape:
        raise ValueError("residual must match x shape")
    if weight.dim() != 1 or weight.numel() != x.size(1):
        raise ValueError("weight must be shaped [hidden]")
    if not x.is_contiguous() or not residual.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x, residual, and weight must be contiguous")
    if x.device != residual.device or x.device != weight.device:
        raise ValueError("x, residual, and weight must be on the same CUDA device")

    rows = int(x.size(0))
    hidden = int(x.size(1))
    if rows <= 0 or hidden <= 0:
        raise ValueError("rows and hidden must be positive")

    triton, kernel = _load_kernel()
    block_size = _next_power_of_2(hidden)
    y = torch.empty_like(x)
    residual_out = torch.empty_like(residual)
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 8192:
        num_warps = 16

    kernel[(rows,)](
        x,
        residual,
        weight,
        y,
        residual_out,
        hidden,
        float(eps),
        block_size,
        num_warps=num_warps,
    )
    return y, residual_out
