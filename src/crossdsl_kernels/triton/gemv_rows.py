"""Strict FP32 matrix tiles reuse each weight across up to sixteen rows."""

import triton
import triton.language as tl


@triton.jit
def _rows_kernel_serial(
    x,
    weight,
    bias,
    y,
    m: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
    nk: tl.constexpr,
    has_bias: tl.constexpr,
):
    rows = tl.program_id(1) * 16 + tl.arange(0, 16)
    cols = tl.program_id(0) * 64 + tl.arange(0, 64)
    reduction = tl.arange(0, 32)
    acc = tl.zeros((16, 64), tl.float32)
    for start in range(0, k, 32):
        kk = start + reduction
        a = tl.load(
            x + rows[:, None] * k + kk[None, :],
            (rows[:, None] < m) & (kk[None, :] < k),
            other=0.0,
        )
        if nk:
            offsets = cols[None, :] * k + kk[:, None]
        else:
            offsets = kk[:, None] * n + cols[None, :]
        b = tl.load(
            weight + offsets, (kk[:, None] < k) & (cols[None, :] < n), other=0.0
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    if has_bias:
        acc += tl.load(bias + cols, cols < n, other=0.0)[None, :]
    tl.store(
        y + rows[:, None] * n + cols[None, :],
        acc,
        (rows[:, None] < m) & (cols[None, :] < n),
    )


@triton.jit
def _split_rows_kernel(
    x,
    weight,
    partial,
    m: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
    nk: tl.constexpr,
    chunk: tl.constexpr,
):
    part = tl.program_id(2)
    rows = tl.program_id(1) * 16 + tl.arange(0, 16)
    cols = tl.program_id(0) * 64 + tl.arange(0, 64)
    reduction = tl.arange(0, 32)
    acc = tl.zeros((16, 64), tl.float32)
    for start in range(0, chunk, 32):
        kk = part * chunk + start + reduction
        a = tl.load(
            x + rows[:, None] * k + kk[None, :],
            (rows[:, None] < m) & (kk[None, :] < k),
            other=0.0,
        )
        if nk:
            offsets = cols[None, :] * k + kk[:, None]
        else:
            offsets = kk[:, None] * n + cols[None, :]
        b = tl.load(
            weight + offsets, (kk[:, None] < k) & (cols[None, :] < n), other=0.0
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(
        partial + part * m * n + rows[:, None] * n + cols[None, :],
        acc,
        (rows[:, None] < m) & (cols[None, :] < n),
    )


@triton.jit
def _finish_split_kernel(
    partial, bias, y, total: tl.constexpr, n: tl.constexpr, has_bias: tl.constexpr
):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < total
    a = tl.load(partial + offsets, mask, other=0.0)
    b = tl.load(partial + total + offsets, mask, other=0.0)
    c = tl.load(partial + 2 * total + offsets, mask, other=0.0)
    d = tl.load(partial + 3 * total + offsets, mask, other=0.0)
    value = (a + b) + (c + d)
    if has_bias:
        value += tl.load(bias + offsets % n, mask, other=0.0)
    tl.store(y + offsets, value, mask)


def decode_gemv_triton_rows(x, weight, bias=None, *, weight_layout="KN"):
    import torch

    from crossdsl_kernels.triton.gemv import _validate_decode_gemv_inputs

    m, k, n, nk = _validate_decode_gemv_inputs(x, weight, bias, weight_layout)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if k < 1024:
        _rows_kernel_serial[(triton.cdiv(n, 64), triton.cdiv(m, 16))](
            x,
            weight,
            bias if bias is not None else y,
            y,
            m,
            k,
            n,
            nk,
            bias is not None,
            num_warps=4,
            num_stages=2,
        )
    else:
        partial = torch.empty((4, m, n), dtype=x.dtype, device=x.device)
        chunk = triton.cdiv(k, 128) * 32
        _split_rows_kernel[(triton.cdiv(n, 64), triton.cdiv(m, 16), 4)](
            x,
            weight,
            partial,
            m,
            k,
            n,
            nk,
            chunk,
            num_warps=4,
            num_stages=2,
        )
        _finish_split_kernel[(triton.cdiv(m * n, 256),)](
            partial,
            bias if bias is not None else y,
            y,
            m * n,
            n,
            bias is not None,
            num_warps=4,
        )
    return y
