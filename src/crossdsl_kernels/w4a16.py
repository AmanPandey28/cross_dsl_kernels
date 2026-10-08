"""Explicit backend selection for the local W4A16 learning workload."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os

from .quantization import validate_w4a16, w4a16_reference


@lru_cache(maxsize=1)
def load_cuda_extension():
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2]
    directory = root / "results/build/torch_extensions"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(directory))
    os.environ.setdefault("MAX_JOBS", "4")
    # This first campaign has no B200 binary or automatic architecture dispatch.
    return load(
        name="crossdsl_w4a16_sm120",
        sources=[
            str(root / "csrc/w4a16/w4a16_ext.cpp"),
            str(root / "csrc/w4a16/w4a16_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_120,code=sm_120"],
        verbose=False,
    )


def w4a16_linear(
    x,
    packed,
    scales,
    bias=None,
    *,
    group_size=128,
    backend="cuda",
    rows_per_block=1,
    out=None,
):
    import torch

    if backend == "reference":
        if out is not None or rows_per_block != 1:
            raise ValueError("reference does not accept out or a launch configuration")
        return w4a16_reference(x, packed, scales, bias, group_size=group_size)
    if backend not in ("cuda", "triton", "cute"):
        raise ValueError("backend must be reference, cuda, triton or cute")
    if type(rows_per_block) is not int or rows_per_block not in (1, 4):
        raise ValueError("rows_per_block must be 1 (baseline) or 4 (reuse candidate)")
    m, _, n = validate_w4a16(x, packed, scales, bias, group_size=group_size, out=out)
    if torch.cuda.get_device_capability(x.device) != (12, 0):
        raise ValueError("this first W4A16 campaign supports SM120 only")
    y = out if out is not None else torch.empty((m, n), device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        if backend == "cuda":
            placeholder = scales if bias is None else bias
            load_cuda_extension().linear_out(
                x,
                packed,
                scales,
                placeholder,
                y,
                group_size,
                rows_per_block,
                bias is not None,
            )
        elif backend == "triton":
            from .triton.w4a16 import launch

            launch(x, packed, scales, bias, y, group_size, rows_per_block)
        else:
            from .cute.w4a16 import launch

            launch(x, packed, scales, bias, y, group_size, rows_per_block)
    return y
