from __future__ import annotations

import math
from typing import Any


def fp32_gemv_tolerance(k: int) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be positive")

    max_abs = 2.0e-4 * max(1.0, math.sqrt(k / 1024.0))
    return {
        "dtype": "float32",
        "max_abs_err": max_abs,
        "rel_l2_err": 1.0e-5,
        "k": k,
        "policy": "max_abs_err <= 2e-4 * max(1, sqrt(K / 1024)) and rel_l2_err <= 1e-5",
        "max_rel_err_role": "diagnostic-only; near-zero reference values can dominate elementwise relative error",
    }


def fp32_gemv_status(k: int, max_abs_err: float, rel_l2_err: float) -> tuple[str, dict[str, Any]]:
    tolerance = fp32_gemv_tolerance(k)
    passed = max_abs_err <= tolerance["max_abs_err"] and rel_l2_err <= tolerance["rel_l2_err"]
    return ("PASS" if passed else "FAIL", tolerance)
