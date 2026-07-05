from __future__ import annotations

from typing import Any


def decode_linear_reference(
    x: Any,
    weight: Any,
    bias: Any | None = None,
    *,
    out_dtype: Any | None = None,
    weight_layout: str = "KN",
) -> Any:
    import torch

    if x.dim() != 2:
        raise ValueError("x must be a 2D tensor shaped [M, K]")
    if weight.dim() != 2:
        raise ValueError("weight must be a 2D tensor")
    if weight_layout == "KN":
        k, n = weight.shape
        mat = weight
    elif weight_layout == "NK":
        n, k = weight.shape
        mat = weight.transpose(0, 1)
    else:
        raise ValueError("weight_layout must be 'KN' or 'NK'")
    if x.shape[1] != k:
        raise ValueError("x K dimension must match weight K dimension")
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != n):
        raise ValueError("bias must be a 1D tensor shaped [N]")

    out = torch.matmul(x.to(torch.float32), mat.to(torch.float32))
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(dtype=out_dtype or x.dtype)
