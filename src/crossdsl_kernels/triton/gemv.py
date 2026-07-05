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
    def _decode_gemv_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        y_ptr,
        m_size: tl.constexpr,
        k_size: tl.constexpr,
        n_size: tl.constexpr,
        block_k: tl.constexpr,
        weight_is_nk: tl.constexpr,
        has_bias: tl.constexpr,
    ):
        out_col = tl.program_id(0)
        out_row = tl.program_id(1)
        offsets = tl.arange(0, block_k)
        mask = offsets < k_size

        x = tl.load(x_ptr + out_row * k_size + offsets, mask=mask, other=0.0).to(tl.float32)
        if weight_is_nk:
            weight_offsets = out_col * k_size + offsets
        else:
            weight_offsets = offsets * n_size + out_col
        weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0).to(tl.float32)
        acc = tl.sum(tl.where(mask, x * weight, 0.0), axis=0)
        if has_bias:
            acc += tl.load(bias_ptr + out_col).to(tl.float32)
        tl.store(y_ptr + out_row * n_size + out_col, acc)

    return triton, _decode_gemv_kernel


@lru_cache(maxsize=1)
def _load_tiled_kernel() -> Any:
    import triton
    import triton.language as tl

    @triton.jit
    def _decode_gemv_tiled_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        y_ptr,
        k_size: tl.constexpr,
        n_size: tl.constexpr,
        block_k: tl.constexpr,
        block_n: tl.constexpr,
        weight_is_nk: tl.constexpr,
        has_bias: tl.constexpr,
    ):
        tile_col = tl.program_id(0) * block_n
        out_row = tl.program_id(1)
        n_offsets = tile_col + tl.arange(0, block_n)
        n_mask = n_offsets < n_size
        k_offsets_base = tl.arange(0, block_k)
        acc = tl.zeros((block_n,), dtype=tl.float32)

        for k_start in range(0, k_size, block_k):
            k_offsets = k_start + k_offsets_base
            k_mask = k_offsets < k_size
            x = tl.load(x_ptr + out_row * k_size + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
            if weight_is_nk:
                weight_offsets = n_offsets[None, :] * k_size + k_offsets[:, None]
            else:
                weight_offsets = k_offsets[:, None] * n_size + n_offsets[None, :]
            weight_mask = k_mask[:, None] & n_mask[None, :]
            weight = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0).to(tl.float32)
            acc += tl.sum(x[:, None] * weight, axis=0)

        if has_bias:
            acc += tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + out_row * n_size + n_offsets, acc, mask=n_mask)

    return triton, _decode_gemv_tiled_kernel


def _select_tiled_block_k(k_size: int) -> int:
    return min(_next_power_of_2(k_size), 256)


def _validate_decode_gemv_inputs(x: Any, weight: Any, bias: Any | None, weight_layout: str) -> tuple[int, int, int, bool]:
    import torch

    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("x and weight must be CUDA tensors")
    if bias is not None and not bias.is_cuda:
        raise ValueError("bias must be a CUDA tensor when provided")
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise ValueError("Triton GEMV supports float32 x and weight tensors only")
    if bias is not None and bias.dtype != torch.float32:
        raise ValueError("Triton GEMV supports float32 bias tensors only")
    if x.dim() != 2:
        raise ValueError("x must be shaped [M, K]")
    if weight.dim() != 2:
        raise ValueError("weight must be 2D")
    if weight_layout == "KN":
        k_size, n_size = int(weight.size(0)), int(weight.size(1))
        weight_is_nk = False
    elif weight_layout == "NK":
        n_size, k_size = int(weight.size(0)), int(weight.size(1))
        weight_is_nk = True
    else:
        raise ValueError("weight_layout must be 'KN' or 'NK'")
    if int(x.size(1)) != k_size:
        raise ValueError("x K dimension must match weight K dimension")
    if bias is not None and (bias.dim() != 1 or int(bias.size(0)) != n_size):
        raise ValueError("bias must be shaped [N]")
    if not x.is_contiguous() or not weight.is_contiguous() or (bias is not None and not bias.is_contiguous()):
        raise ValueError("x, weight, and bias must be contiguous")
    if x.device != weight.device or (bias is not None and x.device != bias.device):
        raise ValueError("x, weight, and bias must be on the same CUDA device")

    m_size = int(x.size(0))
    if m_size <= 0 or k_size <= 0 or n_size <= 0:
        raise ValueError("M, K, and N must be positive")
    return m_size, k_size, n_size, weight_is_nk


def decode_gemv_triton(x: Any, weight: Any, bias: Any | None = None, *, weight_layout: str = "KN") -> Any:
    import torch

    m_size, k_size, n_size, weight_is_nk = _validate_decode_gemv_inputs(x, weight, bias, weight_layout)
    triton, kernel = _load_kernel()
    block_k = _next_power_of_2(k_size)
    y = torch.empty((m_size, n_size), device=x.device, dtype=torch.float32)
    bias_ptr = bias if bias is not None else y
    num_warps = 4
    if block_k >= 2048:
        num_warps = 8
    if block_k >= 8192:
        num_warps = 16

    kernel[(n_size, m_size)](
        x,
        weight,
        bias_ptr,
        y,
        m_size,
        k_size,
        n_size,
        block_k,
        weight_is_nk,
        bias is not None,
        num_warps=num_warps,
    )
    return y


def decode_gemv_triton_tiled(
    x: Any,
    weight: Any,
    bias: Any | None = None,
    *,
    weight_layout: str = "KN",
    block_n: int = 16,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Any:
    import torch

    m_size, k_size, n_size, weight_is_nk = _validate_decode_gemv_inputs(x, weight, bias, weight_layout)
    if block_n <= 0 or block_n & (block_n - 1):
        raise ValueError("block_n must be a positive power of two")
    selected_block_k = int(block_k) if block_k is not None else _select_tiled_block_k(k_size)
    if selected_block_k <= 0 or selected_block_k & (selected_block_k - 1):
        raise ValueError("block_k must be a positive power of two")
    selected_num_warps = int(num_warps) if num_warps is not None else 4
    if num_warps is None and block_n >= 16 and selected_block_k >= 256:
        selected_num_warps = 8
    if selected_num_warps <= 0:
        raise ValueError("num_warps must be positive")
    selected_num_stages = int(num_stages) if num_stages is not None else 3
    if selected_num_stages <= 0:
        raise ValueError("num_stages must be positive")

    triton, kernel = _load_tiled_kernel()
    y = torch.empty((m_size, n_size), device=x.device, dtype=torch.float32)
    bias_ptr = bias if bias is not None else y

    kernel[(triton.cdiv(n_size, block_n), m_size)](
        x,
        weight,
        bias_ptr,
        y,
        k_size,
        n_size,
        selected_block_k,
        block_n,
        weight_is_nk,
        bias is not None,
        num_warps=selected_num_warps,
        num_stages=selected_num_stages,
    )
    return y
