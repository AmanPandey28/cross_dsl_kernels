from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class KernelKey:
    op: str
    backend: str
    arch: str
    dtype: str
    layout: str
    shape_family: str


REGISTRY: dict[KernelKey, Callable] = {}
