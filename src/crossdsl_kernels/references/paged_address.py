from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PagedAddress:
    logical_page: int
    offset: int
    physical_page: int


def address_for_token(position: int, page_size: int, page_table_row: Sequence[int]) -> PagedAddress:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if position < 0:
        raise ValueError("position must be non-negative")

    logical_page = position // page_size
    offset = position % page_size
    if logical_page >= len(page_table_row):
        raise IndexError("position requires a page beyond page_table_row")

    physical_page = int(page_table_row[logical_page])
    if physical_page < 0:
        raise ValueError("physical page IDs must be non-negative")
    return PagedAddress(logical_page=logical_page, offset=offset, physical_page=physical_page)


def nhd_linear_index(
    physical_page: int,
    offset: int,
    head: int,
    dim: int,
    *,
    page_size: int,
    num_heads: int,
    head_dim: int,
) -> int:
    """Flatten cache[page, offset, head, dim] for NHD page layout."""
    return (((physical_page * page_size + offset) * num_heads + head) * head_dim) + dim


def hnd_linear_index(
    physical_page: int,
    offset: int,
    head: int,
    dim: int,
    *,
    page_size: int,
    num_heads: int,
    head_dim: int,
) -> int:
    """Flatten cache[page, head, offset, dim] for HND page layout."""
    return (((physical_page * num_heads + head) * page_size + offset) * head_dim) + dim
