from __future__ import annotations

from typing import Any

import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import T
from cutlass.cute.arch import alloc_smem, sync_threads
from cutlass.cute.runtime import from_dlpack

_THREADS = 256


@cute.kernel
def _fused_residual_rmsnorm_kernel(
    x,
    residual,
    weight,
    y,
    residual_out,
    hidden,
    eps,
):
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32())
    bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())
    bdim = nvvm.read_ptx_sreg_ntid_x(T.i32())

    row_offset = bidx * hidden

    smem_ptr = alloc_smem(cute.Float32, _THREADS)
    smem = cute.make_tensor(smem_ptr, _THREADS)

    thread_sum = 0.0
    col = tidx
    while col < hidden:
        idx = row_offset + col
        r = x[idx] + residual[idx]
        residual_out[idx] = r
        thread_sum = thread_sum + r * r
        col = col + bdim

    smem[tidx] = thread_sum
    sync_threads()

    stride = bdim >> 1
    while stride > 0:
        if tidx < stride:
            smem[tidx] = smem[tidx] + smem[tidx + stride]
        sync_threads()
        stride = stride >> 1

    inv_hidden = 1.0 / hidden
    inv_rms = cute.rsqrt(smem[0] * inv_hidden + eps)

    col = tidx
    while col < hidden:
        idx = row_offset + col
        r = residual_out[idx]
        y[idx] = r * inv_rms * weight[col]
        col = col + bdim


@cute.jit
def _launch(
    x_cute,
    residual_cute,
    weight_cute,
    y_cute,
    residual_out_cute,
    hidden,
    eps,
    rows,
):
    _fused_residual_rmsnorm_kernel(
        x_cute,
        residual_cute,
        weight_cute,
        y_cute,
        residual_out_cute,
        hidden,
        eps,
    ).launch(grid=[rows, 1, 1], block=[_THREADS, 1, 1])


def fused_residual_rmsnorm_cute(
    x: Any,
    residual: Any,
    weight: Any,
    eps: float,
) -> tuple[Any, Any]:
    import torch

    if not x.is_cuda or not residual.is_cuda or not weight.is_cuda:
        raise ValueError("x, residual, and weight must be CUDA tensors")
    if (
        x.dtype != torch.float32
        or residual.dtype != torch.float32
        or weight.dtype != torch.float32
    ):
        raise ValueError("CuTe RMSNorm supports float32 tensors only")
    if x.dim() != 2:
        raise ValueError("x must be shaped [rows, hidden]")
    if residual.shape != x.shape:
        raise ValueError("residual must match x shape")
    if weight.dim() != 1 or weight.numel() != x.size(1):
        raise ValueError("weight must be shaped [hidden]")
    if (
        not x.is_contiguous()
        or not residual.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("x, residual, and weight must be contiguous")
    if x.device != residual.device or x.device != weight.device:
        raise ValueError("x, residual, and weight must be on the same CUDA device")

    rows = int(x.size(0))
    hidden = int(x.size(1))
    if rows <= 0 or hidden <= 0:
        raise ValueError("rows and hidden must be positive")

    x_flat = x.reshape(-1)
    residual_flat = residual.reshape(-1)
    y_flat = torch.empty_like(x_flat)
    residual_out_flat = torch.empty_like(residual_flat)

    _launch(
        from_dlpack(x_flat),
        from_dlpack(residual_flat),
        from_dlpack(weight),
        from_dlpack(y_flat),
        from_dlpack(residual_out_flat),
        hidden,
        float(eps),
        rows,
    )
    return y_flat.reshape_as(x), residual_out_flat.reshape_as(residual)
