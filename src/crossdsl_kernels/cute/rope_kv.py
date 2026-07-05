from __future__ import annotations

from typing import Any

import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import T
from cutlass.cute.runtime import from_dlpack


def _next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


_THREADS = 128


@cute.kernel
def _rotate_q_kernel(
    q,
    cos,
    sin,
    positions,
    q_out,
    tokens,
    q_heads,
    head_dim,
    cos_width,
    rope_dim,
    interleaved,
):
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32())
    bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())
    bidy = nvvm.read_ptx_sreg_ctaid_y(T.i32())
    bdim = nvvm.read_ptx_sreg_ntid_x(T.i32())

    token = bidx
    head = bidy
    pairs = rope_dim // 2
    pos = positions[token]
    base = (token * q_heads + head) * head_dim

    a_dim = 0
    b_dim = 0
    p = tidx
    while p < pairs:
        c = cos[pos * cos_width + p]
        s = sin[pos * cos_width + p]
        if interleaved:
            a_dim = p * 2
            b_dim = a_dim + 1
        else:
            a_dim = p
            b_dim = p + pairs
        a = q[base + a_dim]
        b = q[base + b_dim]
        q_out[base + a_dim] = a * c - b * s
        q_out[base + b_dim] = a * s + b * c
        p = p + bdim

    d = rope_dim + tidx
    while d < head_dim:
        q_out[base + d] = q[base + d]
        d = d + bdim


@cute.jit
def _launch_rotate_q(
    q_cute,
    cos_cute,
    sin_cute,
    positions_cute,
    q_out_cute,
    tokens,
    q_heads,
    head_dim,
    cos_width,
    rope_dim,
    interleaved,
):
    _rotate_q_kernel(
        q_cute,
        cos_cute,
        sin_cute,
        positions_cute,
        q_out_cute,
        tokens,
        q_heads,
        head_dim,
        cos_width,
        rope_dim,
        interleaved,
    ).launch(grid=[tokens, q_heads, 1], block=[_THREADS, 1, 1])


@cute.kernel
def _append_kv_nhd_kernel(
    k,
    v,
    cos,
    sin,
    positions,
    page_table,
    sequence_ids,
    k_cache,
    v_cache,
    tokens,
    kv_heads,
    head_dim,
    cos_width,
    max_pages,
    page_size,
    rope_dim,
    interleaved,
):
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32())
    bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())
    bidy = nvvm.read_ptx_sreg_ctaid_y(T.i32())
    bdim = nvvm.read_ptx_sreg_ntid_x(T.i32())

    token = bidx
    head = bidy
    pairs = rope_dim // 2

    pos = positions[token]
    logical_page = pos // page_size
    off = pos - logical_page * page_size
    seq = sequence_ids[token]
    physical_page = page_table[seq * max_pages + logical_page]
    src_base = (token * kv_heads + head) * head_dim

    p = tidx
    a_dim = 0
    b_dim = 0
    while p < pairs:
        c = cos[pos * cos_width + p]
        s = sin[pos * cos_width + p]
        if interleaved:
            a_dim = p * 2
            b_dim = a_dim + 1
        else:
            a_dim = p
            b_dim = p + pairs

        a_cache = (
            (physical_page * page_size + off) * kv_heads + head
        ) * head_dim + a_dim
        b_cache = (
            (physical_page * page_size + off) * kv_heads + head
        ) * head_dim + b_dim

        a_val = k[src_base + a_dim]
        b_val = k[src_base + b_dim]
        k_cache[a_cache] = a_val * c - b_val * s
        k_cache[b_cache] = a_val * s + b_val * c
        p = p + bdim

    d = tidx
    while d < head_dim:
        cache_idx = ((physical_page * page_size + off) * kv_heads + head) * head_dim + d
        v_cache[cache_idx] = v[src_base + d]
        if d >= rope_dim:
            k_cache[cache_idx] = k[src_base + d]
        d = d + bdim


@cute.kernel
def _append_kv_hnd_kernel(
    k,
    v,
    cos,
    sin,
    positions,
    page_table,
    sequence_ids,
    k_cache,
    v_cache,
    tokens,
    kv_heads,
    head_dim,
    cos_width,
    max_pages,
    page_size,
    rope_dim,
    interleaved,
):
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32())
    bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())
    bidy = nvvm.read_ptx_sreg_ctaid_y(T.i32())
    bdim = nvvm.read_ptx_sreg_ntid_x(T.i32())

    token = bidx
    head = bidy
    pairs = rope_dim // 2

    pos = positions[token]
    logical_page = pos // page_size
    off = pos - logical_page * page_size
    seq = sequence_ids[token]
    physical_page = page_table[seq * max_pages + logical_page]
    src_base = (token * kv_heads + head) * head_dim

    p = tidx
    a_dim = 0
    b_dim = 0
    while p < pairs:
        c = cos[pos * cos_width + p]
        s = sin[pos * cos_width + p]
        if interleaved:
            a_dim = p * 2
            b_dim = a_dim + 1
        else:
            a_dim = p
            b_dim = p + pairs

        a_cache = (
            (physical_page * kv_heads + head) * page_size + off
        ) * head_dim + a_dim
        b_cache = (
            (physical_page * kv_heads + head) * page_size + off
        ) * head_dim + b_dim

        a_val = k[src_base + a_dim]
        b_val = k[src_base + b_dim]
        k_cache[a_cache] = a_val * c - b_val * s
        k_cache[b_cache] = a_val * s + b_val * c
        p = p + bdim

        a_dim_init = a_dim
        b_dim_init = b_dim

    d = tidx
    while d < head_dim:
        cache_idx = ((physical_page * kv_heads + head) * page_size + off) * head_dim + d
        v_cache[cache_idx] = v[src_base + d]
        if d >= rope_dim:
            k_cache[cache_idx] = k[src_base + d]
        d = d + bdim


@cute.jit
def _launch_append_kv_nhd(
    k_cute,
    v_cute,
    cos_cute,
    sin_cute,
    positions_cute,
    page_table_cute,
    sequence_ids_cute,
    k_cache_cute,
    v_cache_cute,
    tokens,
    kv_heads,
    head_dim,
    cos_width,
    max_pages,
    page_size,
    rope_dim,
    interleaved,
):
    _append_kv_nhd_kernel(
        k_cute,
        v_cute,
        cos_cute,
        sin_cute,
        positions_cute,
        page_table_cute,
        sequence_ids_cute,
        k_cache_cute,
        v_cache_cute,
        tokens,
        kv_heads,
        head_dim,
        cos_width,
        max_pages,
        page_size,
        rope_dim,
        interleaved,
    ).launch(grid=[tokens, kv_heads, 1], block=[_THREADS, 1, 1])


@cute.jit
def _launch_append_kv_hnd(
    k_cute,
    v_cute,
    cos_cute,
    sin_cute,
    positions_cute,
    page_table_cute,
    sequence_ids_cute,
    k_cache_cute,
    v_cache_cute,
    tokens,
    kv_heads,
    head_dim,
    cos_width,
    max_pages,
    page_size,
    rope_dim,
    interleaved,
):
    _append_kv_hnd_kernel(
        k_cute,
        v_cute,
        cos_cute,
        sin_cute,
        positions_cute,
        page_table_cute,
        sequence_ids_cute,
        k_cache_cute,
        v_cache_cute,
        tokens,
        kv_heads,
        head_dim,
        cos_width,
        max_pages,
        page_size,
        rope_dim,
        interleaved,
    ).launch(grid=[tokens, kv_heads, 1], block=[_THREADS, 1, 1])


def _validate_inputs(
    q: Any,
    k: Any,
    v: Any,
    cos: Any,
    sin: Any,
    positions: Any,
    page_table: Any,
    sequence_ids: Any,
    k_cache: Any,
    v_cache: Any,
    page_size: int,
    rope_dim: int,
    kv_layout: str,
) -> tuple[int, int, int, int, int, int, bool]:
    import torch

    tensors = {
        "q": (q, torch.float32),
        "k": (k, torch.float32),
        "v": (v, torch.float32),
        "cos": (cos, torch.float32),
        "sin": (sin, torch.float32),
        "positions": (positions, torch.long),
        "page_table": (page_table, torch.long),
        "sequence_ids": (sequence_ids, torch.long),
        "k_cache": (k_cache, torch.float32),
        "v_cache": (v_cache, torch.float32),
    }
    for name, (tensor, dtype) in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} has an unexpected dtype")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if q.dim() != 3:
        raise ValueError("q must be shaped [tokens, q_heads, head_dim]")
    if k.dim() != 3 or v.dim() != 3:
        raise ValueError("k and v must be shaped [tokens, kv_heads, head_dim]")
    if cos.dim() != 2 or sin.dim() != 2:
        raise ValueError("cos and sin must be shaped [max_position, cos_width]")
    if positions.dim() != 1 or sequence_ids.dim() != 1:
        raise ValueError("positions and sequence_ids must be rank-1")
    if page_table.dim() != 2:
        raise ValueError("page_table must be shaped [sequences, max_pages]")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("k_cache and v_cache must be rank-4")

    tokens = int(q.size(0))
    q_heads = int(q.size(1))
    kv_heads = int(k.size(1))
    head_dim = int(q.size(2))
    if tokens <= 0 or q_heads <= 0 or kv_heads <= 0 or head_dim <= 0:
        raise ValueError("tokens, q_heads, kv_heads, and head_dim must be positive")
    if int(k.size(0)) != tokens or int(v.size(0)) != tokens:
        raise ValueError("q, k, and v token counts must match")
    if int(k.size(2)) != head_dim or int(v.size(2)) != head_dim:
        raise ValueError("q, k, and v head_dim must match")
    if int(v.size(1)) != kv_heads:
        raise ValueError("k and v kv_heads must match")
    if int(positions.size(0)) != tokens or int(sequence_ids.size(0)) != tokens:
        raise ValueError("positions and sequence_ids lengths must match tokens")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError("rope_dim must be a positive even integer")
    if rope_dim > head_dim:
        raise ValueError("rope_dim cannot exceed head_dim")
    if int(cos.size(1)) < rope_dim // 2:
        raise ValueError("cos must have at least rope_dim / 2 columns")
    if sin.shape != cos.shape:
        raise ValueError("sin must match cos shape")
    if int(page_table.size(1)) <= 0:
        raise ValueError("page_table must contain at least one page per sequence row")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache shapes must match")

    cache_layout_hnd = kv_layout.upper() == "HND"
    if kv_layout.upper() not in {"NHD", "HND"}:
        raise ValueError("kv_layout must be NHD or HND")
    if cache_layout_hnd:
        if (
            int(k_cache.size(1)) != kv_heads
            or int(k_cache.size(2)) != page_size
            or int(k_cache.size(3)) != head_dim
        ):
            raise ValueError(
                "HND cache shape must be [pages, kv_heads, page_size, head_dim]"
            )
    elif (
        int(k_cache.size(1)) != page_size
        or int(k_cache.size(2)) != kv_heads
        or int(k_cache.size(3)) != head_dim
    ):
        raise ValueError(
            "NHD cache shape must be [pages, page_size, kv_heads, head_dim]"
        )

    device = q.device
    for name, (tensor, _) in tensors.items():
        if tensor.device != device:
            raise ValueError(f"{name} must be on the same CUDA device as q")

    return (
        tokens,
        q_heads,
        kv_heads,
        head_dim,
        int(cos.size(1)),
        int(page_table.size(1)),
        cache_layout_hnd,
    )


def rope_gqa_paged_kv_append_cute(
    q: Any,
    k: Any,
    v: Any,
    cos: Any,
    sin: Any,
    positions: Any,
    page_table: Any,
    sequence_ids: Any,
    k_cache: Any,
    v_cache: Any,
    *,
    page_size: int,
    rope_dim: int,
    interleaved: bool,
    kv_layout: str = "NHD",
) -> Any:
    import torch

    tokens, q_heads, kv_heads, head_dim, cos_width, max_pages, cache_layout_hnd = (
        _validate_inputs(
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
            page_size,
            rope_dim,
            kv_layout,
        )
    )

    q_out = torch.empty_like(q)

    q_flat = q.reshape(-1)
    q_out_flat = q_out.reshape(-1)
    k_flat = k.reshape(-1)
    v_flat = v.reshape(-1)
    cos_flat = cos.reshape(-1)
    sin_flat = sin.reshape(-1)
    positions_flat = positions.reshape(-1)
    page_table_flat = page_table.reshape(-1)
    sequence_ids_flat = sequence_ids.reshape(-1)
    k_cache_flat = k_cache.reshape(-1)
    v_cache_flat = v_cache.reshape(-1)

    _launch_rotate_q(
        from_dlpack(q_flat),
        from_dlpack(cos_flat),
        from_dlpack(sin_flat),
        from_dlpack(positions_flat),
        from_dlpack(q_out_flat),
        tokens,
        q_heads,
        head_dim,
        cos_width,
        rope_dim,
        int(interleaved),
    )
    if cache_layout_hnd:
        _launch_append_kv_hnd(
            from_dlpack(k_flat),
            from_dlpack(v_flat),
            from_dlpack(cos_flat),
            from_dlpack(sin_flat),
            from_dlpack(positions_flat),
            from_dlpack(page_table_flat),
            from_dlpack(sequence_ids_flat),
            from_dlpack(k_cache_flat),
            from_dlpack(v_cache_flat),
            tokens,
            kv_heads,
            head_dim,
            cos_width,
            max_pages,
            int(page_size),
            int(rope_dim),
            int(interleaved),
        )
    else:
        _launch_append_kv_nhd(
            from_dlpack(k_flat),
            from_dlpack(v_flat),
            from_dlpack(cos_flat),
            from_dlpack(sin_flat),
            from_dlpack(positions_flat),
            from_dlpack(page_table_flat),
            from_dlpack(sequence_ids_flat),
            from_dlpack(k_cache_flat),
            from_dlpack(v_cache_flat),
            tokens,
            kv_heads,
            head_dim,
            cos_width,
            max_pages,
            int(page_size),
            int(rope_dim),
            int(interleaved),
        )

    return q_out
