"""Warp-owned NK dots with one or four live row accumulators."""

import cutlass
from cutlass import cute
from cuda.bindings import driver as cuda
from cutlass.cute.arch.nvvm_wrappers import warp_reduction_sum


@cute.kernel
def _w4a16_kernel(
    x,
    packed,
    scales,
    bias,
    y,
    m: cutlass.Int32,
    k: cutlass.Constexpr,
    n: cutlass.Constexpr,
    group: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    has_bias: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    lane = tid % 32
    col = bx * 8 + tid // 32
    row = by * rows
    pairs = (k + 1) // 2
    groups = (k + group - 1) // group
    acc = cute.make_rmem_tensor(cute.make_layout(rows), cutlass.Float32)
    for r in cutlass.range_constexpr(rows):
        acc[r] = cutlass.Float32(0)
    if col < n:
        for pair in range(lane, cutlass.Int32(pairs), 32):
            kk = pair * 2
            code = cutlass.Int32(packed[col * pairs + pair])
            scale = cutlass.Float32(scales[col * groups + kk // group])
            w0 = cutlass.Float32(
                cutlass.Float16(cutlass.Float32((code & 15) - 8) * scale)
            )
            w1 = cutlass.Float32(
                cutlass.Float16(cutlass.Float32((code >> 4) - 8) * scale)
            )
            for r in cutlass.range_constexpr(rows):
                if row + r < m:
                    acc[r] = acc[r] + cutlass.Float32(x[(row + r) * k + kk]) * w0
                    if kk + 1 < k:
                        acc[r] = (
                            acc[r] + cutlass.Float32(x[(row + r) * k + kk + 1]) * w1
                        )
        for r in cutlass.range_constexpr(rows):
            value = warp_reduction_sum(acc[r])
            if lane == 0 and row + r < m:
                if has_bias:
                    value = value + cutlass.Float32(bias[col])
                y[(row + r) * n + col] = value.to(cutlass.Float16)


@cute.jit
def _launch(
    x,
    packed,
    scales,
    bias,
    y,
    m: cutlass.Int32,
    k: cutlass.Constexpr,
    n: cutlass.Constexpr,
    group: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    has_bias: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _w4a16_kernel(x, packed, scales, bias, y, m, k, n, group, rows, has_bias).launch(
        grid=[(n + 7) // 8, (m + rows - 1) // rows, 1],
        block=[256, 1, 1],
        stream=stream,
    )


def launch(x, packed, scales, bias, y, group_size, rows_per_block):
    import torch
    from cutlass.cute.runtime import from_dlpack

    m, k = x.shape
    tensors = (x, packed, scales, scales if bias is None else bias, y)
    _launch(
        *(from_dlpack(t.flatten()) for t in tensors),
        m,
        k,
        packed.shape[0],
        group_size,
        rows_per_block,
        bias is not None,
        cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )
