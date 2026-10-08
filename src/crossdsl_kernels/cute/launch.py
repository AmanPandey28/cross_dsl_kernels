from __future__ import annotations

from collections.abc import Callable
from typing import Any


def prepare_launch(
    launcher: Any,
    tensors: tuple[Any, ...],
    scalars: tuple[Any, ...],
    *,
    constexpr_scalars: tuple[int, ...] = (),
) -> Callable[[], None]:
    """Compile and build tensor adapters once for fixed benchmark buffers."""
    import torch
    from cuda.bindings import driver as cuda
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack

    adapters = tuple(from_dlpack(tensor) for tensor in tensors)
    device = tensors[0].device
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    compiled = cute.compile(launcher, *adapters, *scalars, stream)
    runtime_scalars = tuple(
        value for index, value in enumerate(scalars) if index not in constexpr_scalars
    )

    def run() -> None:
        # Use the caller's stream, including a CUDA Graph capture stream.
        current = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        # Compile-time arguments are absent from the compiled runtime ABI.
        compiled(*adapters, *runtime_scalars, current)

    return run
