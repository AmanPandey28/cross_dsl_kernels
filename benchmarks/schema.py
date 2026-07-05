from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ResultRow:
    run_id: str
    timestamp: str
    git_commit: str
    dirty: bool
    gpu_name: str
    compute_capability: str
    driver: str
    cuda_runtime: str
    cuda_toolkit: str
    pytorch: str
    triton: str
    cutlass_dsl: str
    op: str
    backend: str
    variant: str
    config: str
    shape: dict[str, Any]
    dtype: str
    layout: str
    warmup: int
    repetitions: int
    median_us: float | None
    p05_us: float | None
    p95_us: float | None
    correct: bool | None
    max_abs_err: float | None
    max_rel_err: float | None
    rel_l2_err: float | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
