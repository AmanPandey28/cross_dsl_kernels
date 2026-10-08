"""One launch for Q rotation and direct rotated-K/V cache writes."""

import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    q,
    k,
    v,
    cos,
    sin,
    positions,
    table,
    sequences,
    kc,
    vc,
    qo,
    hq: tl.constexpr,
    hkv: tl.constexpr,
    dim: tl.constexpr,
    width: tl.constexpr,
    max_pages: tl.constexpr,
    page_size: tl.constexpr,
    rope: tl.constexpr,
    interleaved: tl.constexpr,
    hnd: tl.constexpr,
    block: tl.constexpr,
):
    token, slot = tl.program_id(0), tl.program_id(1)
    is_q = slot < hq
    head = tl.where(is_q, slot, slot - hq)
    base = (token * tl.where(is_q, hq, hkv) + head) * dim
    src = tl.where(is_q, q, k)
    offsets = tl.arange(0, block)
    position = tl.load(positions + token)
    pairs = rope // 2
    if interleaved:
        first = offsets % 2 == 0
        pair = offsets // 2
        partner = tl.where(first, offsets + 1, offsets - 1)
    else:
        first = offsets < pairs
        pair = tl.where(first, offsets, offsets - pairs)
        partner = tl.where(first, offsets + pairs, offsets - pairs)
    value = tl.load(src + base + offsets, offsets < dim, other=0.0)
    other = tl.load(src + base + partner, offsets < rope, other=0.0)
    c = tl.load(cos + position * width + pair, offsets < rope, other=1.0)
    s = tl.load(sin + position * width + pair, offsets < rope, other=0.0)
    rotated = tl.where(first, value * c - other * s, value * c + other * s)
    out = tl.where(offsets < rope, rotated, value)
    if is_q:
        tl.store(qo + base + offsets, out, offsets < dim)
    else:
        sequence = tl.load(sequences + token)
        page = tl.load(table + sequence * max_pages + position // page_size)
        offset = position % page_size
        if hnd:
            dest = ((page * hkv + head) * page_size + offset) * dim
        else:
            dest = ((page * page_size + offset) * hkv + head) * dim
        vv = tl.load(v + base + offsets, offsets < dim, other=0.0)
        tl.store(kc + dest + offsets, out, offsets < dim)
        tl.store(vc + dest + offsets, vv, offsets < dim)


def rope_gqa_paged_kv_append_triton_fused(
    q,
    k,
    v,
    cos,
    sin,
    positions,
    page_table,
    sequence_ids,
    k_cache,
    v_cache,
    *,
    page_size,
    rope_dim,
    interleaved,
    kv_layout="NHD",
):
    import torch

    from crossdsl_kernels.triton.rope_kv import _validate_inputs

    args = (q, k, v, cos, sin, positions, page_table, sequence_ids, k_cache, v_cache)
    tokens, hq, hkv, dim, width, max_pages, hnd = _validate_inputs(
        *args, page_size, rope_dim, kv_layout
    )
    out = torch.empty_like(q)
    _fused_kernel[(tokens, hq + hkv)](
        *args,
        out,
        hq,
        hkv,
        dim,
        width,
        max_pages,
        page_size,
        rope_dim,
        interleaved,
        hnd,
        triton.next_power_of_2(dim),
        num_warps=1 if dim <= 128 else 4,
    )
    return out
