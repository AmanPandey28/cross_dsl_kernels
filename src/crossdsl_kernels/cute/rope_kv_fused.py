"""One block per Q or KV head, dispatched together in one launch."""

from cuda.bindings import driver as cuda
from cutlass import cute


@cute.kernel
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
    hq,
    hkv,
    dim,
    width,
    max_pages,
    page_size,
    rope,
    interleaved,
    hnd,
):
    tid, _, _ = cute.arch.thread_idx()
    token, slot, _ = cute.arch.block_idx()
    is_q = slot < hq
    head = slot
    heads = hq
    if not is_q:
        head = slot - hq
        heads = hkv
    base = cute.Int64(token * heads + head) * dim
    position = positions[token]
    dest = base
    if not is_q:
        page = table[sequences[token] * max_pages + position // page_size]
        offset = position % page_size
        if hnd != 0:
            dest = ((page * hkv + head) * page_size + offset) * dim
        else:
            dest = ((page * page_size + offset) * hkv + head) * dim
    pairs = rope // 2
    for d in range(tid, dim, 64):
        value = 0.0
        if is_q:
            value = q[base + d]
        else:
            value = k[base + d]
        if d < rope:
            first = d < pairs
            pair = d
            partner = d + pairs
            if interleaved != 0:
                first = d % 2 == 0
                pair = d // 2
                partner = d + 1
                if not first:
                    partner = d - 1
            else:
                if not first:
                    pair = d - pairs
                    partner = d - pairs
            other = 0.0
            if is_q:
                other = q[base + partner]
            else:
                other = k[base + partner]
            c = cos[position * width + pair]
            s = sin[position * width + pair]
            if first:
                value = value * c - other * s
            else:
                value = value * c + other * s
        if is_q:
            qo[base + d] = value
        else:
            kc[dest + d] = value
            vc[dest + d] = v[base + d]


@cute.jit
def _launch(
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
    tokens,
    hq,
    hkv,
    dim,
    width,
    max_pages,
    page_size,
    rope,
    interleaved,
    hnd,
    stream: cuda.CUstream,
):
    _fused_kernel(
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
        hq,
        hkv,
        dim,
        width,
        max_pages,
        page_size,
        rope,
        interleaved,
        hnd,
    ).launch(
        grid=[tokens, hq + hkv, 1],
        block=[64, 1, 1],
        stream=stream,
    )


def rope_gqa_paged_kv_append_cute_fused(
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
    from cutlass.cute.runtime import from_dlpack

    from crossdsl_kernels.cute.rope_kv import _validate_inputs

    args = (q, k, v, cos, sin, positions, page_table, sequence_ids, k_cache, v_cache)
    tokens, hq, hkv, dim, width, max_pages, hnd = _validate_inputs(
        *args, page_size, rope_dim, kv_layout
    )
    out = torch.empty_like(q)
    _launch(
        *(from_dlpack(t.flatten()) for t in (*args, out)),
        tokens,
        hq,
        hkv,
        dim,
        width,
        max_pages,
        page_size,
        rope_dim,
        int(interleaved),
        int(hnd),
        cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream),
    )
    return out
