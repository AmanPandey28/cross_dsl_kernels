from __future__ import annotations

import math
from typing import Any

from cuda.bindings import driver as cuda
from cutlass import Constexpr, cute, range_constexpr
from cutlass._mlir.dialects import nvvm
from cutlass.cute.arch import alloc_smem, sync_threads
from cutlass.cute.arch.nvvm_wrappers import warp_reduction_sum
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T

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
    stream: cuda.CUstream,
):
    _fused_residual_rmsnorm_kernel(
        x_cute,
        residual_cute,
        weight_cute,
        y_cute,
        residual_out_cute,
        hidden,
        eps,
    ).launch(grid=[rows, 1, 1], block=[_THREADS, 1, 1], stream=stream)


@cute.kernel
def _warp_rmsnorm_kernel(x, residual, weight, y, residual_out, hidden, eps):
    tid = nvvm.read_ptx_sreg_tid_x(T.i32())
    row = nvvm.read_ptx_sreg_ctaid_x(T.i32())
    lane = tid % 32
    warp = tid // 32
    sums = cute.make_tensor(alloc_smem(cute.Float32, 8), 8)
    total = 0.0
    for col in range(tid, hidden, _THREADS):
        offset = row * hidden + col
        r = x[offset] + residual[offset]
        residual_out[offset] = r
        total = total + r * r
    total = warp_reduction_sum(total)
    if lane == 0:
        sums[warp] = total
    sync_threads()
    if warp == 0:
        partial = 0.0
        if lane < 8:
            partial = sums[lane]
        partial = warp_reduction_sum(partial)
        if lane == 0:
            sums[0] = partial
    sync_threads()
    inv = cute.rsqrt(sums[0] / hidden + eps)
    for col in range(tid, hidden, _THREADS):
        offset = row * hidden + col
        y[offset] = residual_out[offset] * inv * weight[col]


@cute.jit
def _launch_fast(x, r, w, y, ro, hidden, eps, rows, stream: cuda.CUstream):
    _warp_rmsnorm_kernel(x, r, w, y, ro, hidden, eps).launch(
        grid=[rows, 1, 1], block=[_THREADS, 1, 1], stream=stream
    )


@cute.kernel
def _register_rmsnorm_kernel(
    x, residual, weight, y, residual_out, hidden: Constexpr, eps
):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane, warp = tid % 32, tid // 32
    items = (hidden + _THREADS - 1) // _THREADS
    values = cute.make_rmem_tensor(cute.make_layout((items,)), cute.Float32)
    sums = cute.make_tensor(alloc_smem(cute.Float32, 8), 8)
    total = 0.0
    for i in range_constexpr(items):
        col = tid + i * _THREADS
        values[i] = 0.0
        if col < hidden:
            offset = row * hidden + col
            r = x[offset] + residual[offset]
            values[i] = r
            residual_out[offset] = r
            total = total + r * r
    total = warp_reduction_sum(total)
    if lane == 0:
        sums[warp] = total
    sync_threads()
    if warp == 0:
        partial = 0.0
        if lane < 8:
            partial = sums[lane]
        partial = warp_reduction_sum(partial)
        if lane == 0:
            sums[0] = partial
    sync_threads()
    inv = cute.rsqrt(sums[0] / hidden + eps)
    for i in range_constexpr(items):
        col = tid + i * _THREADS
        if col < hidden:
            y[row * hidden + col] = values[i] * inv * weight[col]


@cute.jit
def _launch_resident(
    x, r, w, y, ro, hidden: Constexpr, eps, rows, stream: cuda.CUstream
):
    _register_rmsnorm_kernel(x, r, w, y, ro, hidden, eps).launch(
        grid=[rows, 1, 1], block=[_THREADS, 1, 1], stream=stream
    )


def fused_residual_rmsnorm_cute(
    x: Any,
    residual: Any,
    weight: Any,
    eps: float,
    *,
    fast: bool = False,
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
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    x_flat = x.reshape(-1)
    residual_flat = residual.reshape(-1)
    y_flat = torch.empty_like(x_flat)
    residual_out_flat = torch.empty_like(residual_flat)

    launcher = (
        (_launch_resident if hidden <= 8192 else _launch_fast) if fast else _launch
    )
    launcher(
        from_dlpack(x_flat),
        from_dlpack(residual_flat),
        from_dlpack(weight),
        from_dlpack(y_flat),
        from_dlpack(residual_out_flat),
        hidden,
        float(eps),
        rows,
        cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )
    return y_flat.reshape_as(x), residual_out_flat.reshape_as(residual)
