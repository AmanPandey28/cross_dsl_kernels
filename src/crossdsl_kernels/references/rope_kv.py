from __future__ import annotations

from typing import Any

from .paged_address import address_for_token


def _rotate(x: Any, cos: Any, sin: Any, positions: Any, rope_dim: int, interleaved: bool) -> Any:
    import torch

    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError("rope_dim must be a positive even integer")
    if rope_dim > x.shape[-1]:
        raise ValueError("rope_dim cannot exceed the head dimension")

    out = x.clone()
    pairs = rope_dim // 2
    c = cos[positions, :pairs].to(torch.float32).unsqueeze(1)
    s = sin[positions, :pairs].to(torch.float32).unsqueeze(1)
    src = x[..., :rope_dim].to(torch.float32)

    if interleaved:
        a = src[..., 0::2]
        b = src[..., 1::2]
        out[..., :rope_dim:2] = (a * c - b * s).to(dtype=x.dtype)
        out[..., 1:rope_dim:2] = (a * s + b * c).to(dtype=x.dtype)
    else:
        a = src[..., :pairs]
        b = src[..., pairs:rope_dim]
        out[..., :pairs] = (a * c - b * s).to(dtype=x.dtype)
        out[..., pairs:rope_dim] = (a * s + b * c).to(dtype=x.dtype)
    return out


def rope_gqa_paged_kv_append_reference(
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
    rotated_q = _rotate(q, cos, sin, positions, rope_dim, interleaved)
    rotated_k = _rotate(k, cos, sin, positions, rope_dim, interleaved)

    layout = kv_layout.upper()
    if layout not in {"NHD", "HND"}:
        raise ValueError("kv_layout must be NHD or HND")

    for token in range(k.shape[0]):
        seq = int(sequence_ids[token])
        pos = int(positions[token])
        addr = address_for_token(pos, page_size, page_table[seq].tolist())
        for head in range(k.shape[1]):
            if layout == "NHD":
                k_cache[addr.physical_page, addr.offset, head, :] = rotated_k[token, head, :]
                v_cache[addr.physical_page, addr.offset, head, :] = v[token, head, :]
            else:
                k_cache[addr.physical_page, head, addr.offset, :] = rotated_k[token, head, :]
                v_cache[addr.physical_page, head, addr.offset, :] = v[token, head, :]
    return rotated_q


def validate_paged_kv_indices(
    positions: Any,
    page_table: Any,
    sequence_ids: Any,
    *,
    page_size: int,
    num_cache_pages: int,
    max_position: int,
) -> None:
    """Check address data before launching a kernel; this may synchronize CUDA."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if positions.dim() != 1 or sequence_ids.shape != positions.shape:
        raise ValueError("positions and sequence_ids must be matching vectors")
    if page_table.dim() != 2:
        raise ValueError("page_table must be shaped [sequences, logical_pages]")

    table = page_table.cpu().tolist()
    slots = set()
    for position, sequence in zip(positions.cpu().tolist(), sequence_ids.cpu().tolist()):
        if not 0 <= sequence < len(table):
            raise ValueError("sequence ID is out of bounds")
        if not 0 <= position < max_position:
            raise ValueError("position is outside the trigonometric table")
        address = address_for_token(position, page_size, table[sequence])
        if address.physical_page >= num_cache_pages:
            raise ValueError("physical page ID is outside the cache")
        slot = (address.physical_page, address.offset)
        if slot in slots:
            raise ValueError("tokens must write distinct physical cache slots")
        slots.add(slot)


def rope_gqa_paged_kv_append_vectorized(
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
    """PyTorch baseline with batched page lookup and writes, without CPU loops.

    Call validate_paged_kv_indices once during setup. Like the custom kernels,
    this path requires valid, distinct destination slots.
    """
    import torch

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    layout = kv_layout.upper()
    if layout not in {"NHD", "HND"}:
        raise ValueError("kv_layout must be NHD or HND")
    if q.shape[1] % k.shape[1]:
        raise ValueError("q_heads must be divisible by kv_heads")

    rotated_q = _rotate(q, cos, sin, positions, rope_dim, interleaved)
    rotated_k = _rotate(k, cos, sin, positions, rope_dim, interleaved)
    logical_pages = torch.div(positions, page_size, rounding_mode="floor")
    offsets = positions.remainder(page_size)
    physical_pages = page_table[sequence_ids, logical_pages]
    if layout == "NHD":
        k_cache[physical_pages, offsets] = rotated_k
        v_cache[physical_pages, offsets] = v
    else:
        k_cache[physical_pages, :, offsets] = rotated_k
        v_cache[physical_pages, :, offsets] = v
    return rotated_q
