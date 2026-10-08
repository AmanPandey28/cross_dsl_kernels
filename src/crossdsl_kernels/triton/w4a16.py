"""SIMT packed-weight reduction; no Tensor Core claim."""

import triton
import triton.language as tl


@triton.jit
def _w4a16_kernel(
    X,
    Q,
    S,
    B,
    Y,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP: tl.constexpr,
    ROWS: tl.constexpr,
    BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    local_k = tl.arange(0, BK)
    pairs = tl.cdiv(K, 2)
    groups = tl.cdiv(K, GROUP)
    acc = tl.full((ROWS, BN, BK), 0.0, tl.float32)
    for base in range(tl.cdiv(K, BK)):
        kk = base * BK + local_k
        mask = (cols[:, None] < N) & (kk[None, :] < K)
        code = tl.load(
            Q + cols[:, None] * pairs + kk[None, :] // 2, mask, other=136
        ).to(tl.int32)
        q = ((code >> ((kk[None, :] % 2) * 4)) & 15) - 8
        scale = tl.load(
            S + cols[:, None] * groups + kk[None, :] // GROUP, mask, other=0.0
        ).to(tl.float32)
        w = (q.to(tl.float32) * scale).to(tl.float16).to(tl.float32)
        a = tl.load(
            X + rows[:, None] * K + kk[None, :],
            (rows[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        acc = tl.fma(a[:, None, :], w[None, :, :], acc)
    value = tl.sum(acc, axis=2)
    if BIAS:
        value += tl.load(B + cols, cols < N, other=0.0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        value,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def launch(x, packed, scales, bias, y, group_size, rows_per_block):
    m, k = x.shape
    n = packed.shape[0]
    _w4a16_kernel[(triton.cdiv(n, 4), triton.cdiv(m, rows_per_block))](
        x,
        packed,
        scales,
        scales if bias is None else bias,
        y,
        m,
        k,
        n,
        group_size,
        rows_per_block,
        bias is not None,
        4,
        256,
        num_warps=4,
        num_stages=1,
    )
