from __future__ import annotations

from typing import Any


def fused_residual_rmsnorm_reference(
    x: Any,
    residual: Any,
    weight: Any,
    eps: float,
    *,
    return_residual: bool = True,
) -> tuple[Any, Any | None]:
    """Out-of-place residual RMSNorm; normalize the unrounded FP32 sum.

    BF16 inputs are promoted before addition. Only the two returned tensors
    are rounded to the input dtype; neither input is modified.
    """
    import torch

    r_fp32 = x.to(torch.float32) + residual.to(torch.float32)
    inv_rms = torch.rsqrt(torch.mean(r_fp32 * r_fp32, dim=-1, keepdim=True) + eps)
    y = (r_fp32 * inv_rms * weight.to(torch.float32)).to(dtype=x.dtype)
    residual_out = r_fp32.to(dtype=residual.dtype) if return_residual else None
    return y, residual_out
