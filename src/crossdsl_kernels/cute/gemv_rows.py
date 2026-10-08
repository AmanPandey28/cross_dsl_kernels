"""Four-row SIMT GEMV with layout-specific lane ownership."""

from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.arch import alloc_smem, sync_threads
from cutlass.cute.arch.nvvm_wrappers import warp_reduction_sum


@cute.kernel
def _rows_kernel(x, weight, bias, y, m, k, n, nk, has_bias):
    tid, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    lane, warp = tid % 32, tid // 32
    col = bx * 32 + lane
    start = warp
    step = 8
    if nk != 0:
        col = bx * 8 + warp
        start = lane
        step = 32
    row = by * 4
    a0, a1, a2, a3 = 0.0, 0.0, 0.0, 0.0
    for kk in range(start, k, step):
        w = 0.0
        if col < n:
            if nk != 0:
                w = weight[col * k + kk]
            else:
                w = weight[kk * n + col]
        if row < m:
            a0 = a0 + x[row * k + kk] * w
        if row + 1 < m:
            a1 = a1 + x[(row + 1) * k + kk] * w
        if row + 2 < m:
            a2 = a2 + x[(row + 2) * k + kk] * w
        if row + 3 < m:
            a3 = a3 + x[(row + 3) * k + kk] * w
    if nk != 0:
        a0 = warp_reduction_sum(a0)
        a1 = warp_reduction_sum(a1)
        a2 = warp_reduction_sum(a2)
        a3 = warp_reduction_sum(a3)
    else:
        partial = cute.make_tensor(alloc_smem(cute.Float32, 1024), 1024)
        partial[tid] = a0
        partial[256 + tid] = a1
        partial[512 + tid] = a2
        partial[768 + tid] = a3
        sync_threads()
        a0, a1, a2, a3 = 0.0, 0.0, 0.0, 0.0
        if warp == 0:
            for w in range(8):
                a0 = a0 + partial[w * 32 + lane]
                a1 = a1 + partial[256 + w * 32 + lane]
                a2 = a2 + partial[512 + w * 32 + lane]
                a3 = a3 + partial[768 + w * 32 + lane]
    owns_output = (nk != 0 and lane == 0) or (nk == 0 and warp == 0)
    if owns_output and col < n:
        b = 0.0
        if has_bias != 0:
            b = bias[col]
        if row < m:
            y[row * n + col] = a0 + b
        if row + 1 < m:
            y[(row + 1) * n + col] = a1 + b
        if row + 2 < m:
            y[(row + 2) * n + col] = a2 + b
        if row + 3 < m:
            y[(row + 3) * n + col] = a3 + b


@cute.jit
def _launch(x, w, b, y, m, k, n, nk, has_bias, stream: cuda.CUstream):
    columns = 32
    if nk != 0:
        columns = 8
    _rows_kernel(x, w, b, y, m, k, n, nk, has_bias).launch(
        grid=[(n + columns - 1) // columns, (m + 3) // 4, 1],
        block=[256, 1, 1],
        stream=stream,
    )


def decode_gemv_cute_rows(x, weight, bias=None, *, weight_layout="KN"):
    import torch
    from cutlass.cute.runtime import from_dlpack

    from crossdsl_kernels.cute.gemv import _validate_decode_gemv_inputs

    m, k, n, _, nk = _validate_decode_gemv_inputs(x, weight, bias, weight_layout)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    tensors = (x, weight, bias if bias is not None else y, y)
    _launch(
        *(from_dlpack(t.flatten()) for t in tensors),
        m,
        k,
        n,
        nk,
        int(bias is not None),
        cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )
    return y
