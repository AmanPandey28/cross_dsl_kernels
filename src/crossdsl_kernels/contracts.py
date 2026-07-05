from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
