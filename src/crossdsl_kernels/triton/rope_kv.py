from __future__ import annotations

from functools import lru_cache
from typing import Any


def _next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


@lru_cache(maxsize=1)
def _load_kernels() -> Any:
    import triton
    import triton.language as tl

    @triton.jit
    def _rope_rotate_q_kernel(
        q_ptr,
        cos_ptr,
        sin_ptr,
        positions_ptr,
        q_out_ptr,
        q_heads: tl.constexpr,
        head_dim: tl.constexpr,
        cos_width: tl.constexpr,
        rope_dim: tl.constexpr,
        interleaved: tl.constexpr,
        block_d: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        offsets = tl.arange(0, block_d)
        head_mask = offsets < head_dim
        rope_mask = offsets < rope_dim
        pairs = rope_dim // 2
        position = tl.load(positions_ptr + token)
        base = (token * q_heads + head) * head_dim

        x = tl.load(q_ptr + base + offsets, mask=head_mask, other=0.0).to(tl.float32)
        if interleaved:
            pair_offsets = offsets // 2
            paired_offsets = tl.where((offsets % 2) == 0, offsets + 1, offsets - 1)
            pair_value = tl.load(q_ptr + base + paired_offsets, mask=rope_mask, other=0.0).to(tl.float32)
            c = tl.load(cos_ptr + position * cos_width + pair_offsets, mask=rope_mask, other=1.0).to(tl.float32)
            s = tl.load(sin_ptr + position * cos_width + pair_offsets, mask=rope_mask, other=0.0).to(tl.float32)
            first = (offsets % 2) == 0
            rotated = tl.where(first, x * c - pair_value * s, pair_value * s + x * c)
        else:
            first = offsets < pairs
            pair_offsets = tl.where(first, offsets, offsets - pairs)
            paired_offsets = tl.where(first, offsets + pairs, offsets - pairs)
            pair_value = tl.load(q_ptr + base + paired_offsets, mask=rope_mask, other=0.0).to(tl.float32)
            c = tl.load(cos_ptr + position * cos_width + pair_offsets, mask=rope_mask, other=1.0).to(tl.float32)
            s = tl.load(sin_ptr + position * cos_width + pair_offsets, mask=rope_mask, other=0.0).to(tl.float32)
            rotated = tl.where(first, x * c - pair_value * s, pair_value * s + x * c)

        out = tl.where(rope_mask, rotated, x)
        tl.store(q_out_ptr + base + offsets, out, mask=head_mask)

    @triton.jit
    def _rope_append_kv_kernel(
        k_ptr,
        v_ptr,
        cos_ptr,
        sin_ptr,
        positions_ptr,
        page_table_ptr,
        sequence_ids_ptr,
        k_cache_ptr,
        v_cache_ptr,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        cos_width: tl.constexpr,
        max_pages_per_sequence: tl.constexpr,
        page_size: tl.constexpr,
        rope_dim: tl.constexpr,
        interleaved: tl.constexpr,
        cache_layout_hnd: tl.constexpr,
        block_d: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        offsets = tl.arange(0, block_d)
        head_mask = offsets < head_dim
        rope_mask = offsets < rope_dim
        pairs = rope_dim // 2

        position = tl.load(positions_ptr + token)
        logical_page = position // page_size
        offset_in_page = position - logical_page * page_size
        sequence = tl.load(sequence_ids_ptr + token)
        physical_page = tl.load(page_table_ptr + sequence * max_pages_per_sequence + logical_page)
        src_base = (token * kv_heads + head) * head_dim

        k_value = tl.load(k_ptr + src_base + offsets, mask=head_mask, other=0.0).to(tl.float32)
        if interleaved:
            pair_offsets = offsets // 2
            paired_offsets = tl.where((offsets % 2) == 0, offsets + 1, offsets - 1)
            pair_value = tl.load(k_ptr + src_base + paired_offsets, mask=rope_mask, other=0.0).to(tl.float32)
            c = tl.load(cos_ptr + position * cos_width + pair_offsets, mask=rope_mask, other=1.0).to(tl.float32)
            s = tl.load(sin_ptr + position * cos_width + pair_offsets, mask=rope_mask, other=0.0).to(tl.float32)
            first = (offsets % 2) == 0
            rotated_k = tl.where(first, k_value * c - pair_value * s, pair_value * s + k_value * c)
        else:
            first = offsets < pairs
            pair_offsets = tl.where(first, offsets, offsets - pairs)
            paired_offsets = tl.where(first, offsets + pairs, offsets - pairs)
            pair_value = tl.load(k_ptr + src_base + paired_offsets, mask=rope_mask, other=0.0).to(tl.float32)
            c = tl.load(cos_ptr + position * cos_width + pair_offsets, mask=rope_mask, other=1.0).to(tl.float32)
            s = tl.load(sin_ptr + position * cos_width + pair_offsets, mask=rope_mask, other=0.0).to(tl.float32)
            rotated_k = tl.where(first, k_value * c - pair_value * s, pair_value * s + k_value * c)

        if cache_layout_hnd:
            cache_offsets = ((physical_page * kv_heads + head) * page_size + offset_in_page) * head_dim + offsets
        else:
            cache_offsets = ((physical_page * page_size + offset_in_page) * kv_heads + head) * head_dim + offsets

        k_out = tl.where(rope_mask, rotated_k, k_value)
        v_value = tl.load(v_ptr + src_base + offsets, mask=head_mask, other=0.0).to(tl.float32)
        tl.store(k_cache_ptr + cache_offsets, k_out, mask=head_mask)
        tl.store(v_cache_ptr + cache_offsets, v_value, mask=head_mask)

    return triton, _rope_rotate_q_kernel, _rope_append_kv_kernel


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
        if int(k_cache.size(1)) != kv_heads or int(k_cache.size(2)) != page_size or int(k_cache.size(3)) != head_dim:
            raise ValueError("HND cache shape must be [pages, kv_heads, page_size, head_dim]")
    elif int(k_cache.size(1)) != page_size or int(k_cache.size(2)) != kv_heads or int(k_cache.size(3)) != head_dim:
        raise ValueError("NHD cache shape must be [pages, page_size, kv_heads, head_dim]")

    device = q.device
    for name, (tensor, _) in tensors.items():
        if tensor.device != device:
            raise ValueError(f"{name} must be on the same CUDA device as q")

    return tokens, q_heads, kv_heads, head_dim, int(cos.size(1)), int(page_table.size(1)), cache_layout_hnd


def rope_gqa_paged_kv_append_triton(
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
    num_warps: int | None = None,
) -> Any:
    import torch

    tokens, q_heads, kv_heads, head_dim, cos_width, max_pages, cache_layout_hnd = _validate_inputs(
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
    triton, rotate_q_kernel, append_kv_kernel = _load_kernels()
    block_d = _next_power_of_2(head_dim)
    selected_num_warps = int(num_warps) if num_warps is not None else 4
    if selected_num_warps <= 0:
        raise ValueError("num_warps must be positive")

    q_out = torch.empty_like(q)
    rotate_q_kernel[(tokens, q_heads)](
        q,
        cos,
        sin,
        positions,
        q_out,
        q_heads,
        head_dim,
        cos_width,
        rope_dim,
        bool(interleaved),
        block_d,
        num_warps=selected_num_warps,
    )
    append_kv_kernel[(tokens, kv_heads)](
        k,
        v,
        cos,
        sin,
        positions,
        page_table,
        sequence_ids,
        k_cache,
        v_cache,
        kv_heads,
        head_dim,
        cos_width,
        max_pages,
        int(page_size),
        int(rope_dim),
        bool(interleaved),
        cache_layout_hnd,
        block_d,
        num_warps=selected_num_warps,
    )
    return q_out
