"""Metadata-only checks shared by the Python kernel frontends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Evidence(str, Enum):
    DESIGNED = "DESIGNED"
    IMPLEMENTED = "IMPLEMENTED"
    COMPILED = "COMPILED"
    GPU_TESTED = "GPU-TESTED"
    BENCHMARKED = "BENCHMARKED"
    PROFILED = "PROFILED"
    RELEASED = "RELEASED"


@dataclass(frozen=True)
class KernelContract:
    op: str
    layout: str
    dtype_policy: str
    accumulation: str
    mutation: str
    evidence: Evidence = Evidence.DESIGNED


def validate_cache_storage(inputs: tuple[Any, ...], k_cache: Any, v_cache: Any) -> None:
    # All callers require contiguous tensors, so byte intervals are exact.
    def overlaps(a: Any, b: Any) -> bool:
        if not a.numel() or not b.numel():
            return False
        a0, b0 = a.data_ptr(), b.data_ptr()
        return (
            a0 < b0 + b.numel() * b.element_size()
            and b0 < a0 + a.numel() * a.element_size()
        )

    if overlaps(k_cache, v_cache) or any(
        overlaps(cache, src) for cache in (k_cache, v_cache) for src in inputs
    ):
        raise ValueError("KV caches must not overlap each other or the input tensors")
