from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _load_kernels() -> Any:
    import torch
    import cutlass.cute as cute
    from cutlass._mlir.dialects import nvvm
    from cutlass.cutlass_dsl import T
    from cutlass.cute.runtime import from_dlpack

    @cute.kernel
    def _decode_gemv_kernel(
        x,
        weight,
        bias,
        y,
        total_outputs,
        k_size,
        n_size,
        weight_is_nk,
        has_bias,
    ):
        tidx = nvvm.read_ptx_sreg_tid_x(T.i32())
        bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())
        bdim = nvvm.read_ptx_sreg_ntid_x(T.i32())
        idx = bidx * bdim + tidx
        if idx < total_outputs:
            row = idx // n_size
            col = idx - row * n_size
            acc = 0.0
            for kk in range(k_size):
                x_val = x[row, kk]
                w_val = 0.0
                if weight_is_nk != 0:
                    w_val = weight[col, kk]
                else:
                    w_val = weight[kk, col]
                acc = acc + x_val * w_val
            if has_bias != 0:
                acc = acc + bias[col]
            y[row, col] = acc

    @cute.jit
    def _launch(xc, wc, bc, yc, total_outputs, k_size, n_size, weight_is_nk, has_bias):
        threads = 256
        blocks = (total_outputs + threads - 1) // threads
        _decode_gemv_kernel(
            xc,
            wc,
            bc,
            yc,
            total_outputs,
            k_size,
            n_size,
            weight_is_nk,
            has_bias,
        ).launch(grid=[blocks, 1, 1], block=[threads, 1, 1])

    return cute, from_dlpack, _launch


def _validate_decode_gemv_inputs(
    x: Any,
    weight: Any,
    bias: Any | None,
    weight_layout: str,
) -> tuple[int, int, int, int, int]:
    import torch

    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("x and weight must be CUDA tensors")
    if bias is not None and not bias.is_cuda:
        raise ValueError("bias must be a CUDA tensor when provided")
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise ValueError("CuTe GEMV supports float32 x and weight tensors only")
    if bias is not None and bias.dtype != torch.float32:
        raise ValueError("CuTe GEMV supports float32 bias tensors only")
    if x.dim() != 2:
        raise ValueError("x must be shaped [M, K]")
    if weight.dim() != 2:
        raise ValueError("weight must be 2D")
    if weight_layout == "KN":
        k_size, n_size = int(weight.size(0)), int(weight.size(1))
        weight_is_nk = 0
    elif weight_layout == "NK":
        n_size, k_size = int(weight.size(0)), int(weight.size(1))
        weight_is_nk = 1
    else:
        raise ValueError("weight_layout must be 'KN' or 'NK'")
    if int(x.size(1)) != k_size:
        raise ValueError("x K dimension must match weight K dimension")
    if bias is not None and (bias.dim() != 1 or int(bias.size(0)) != n_size):
        raise ValueError("bias must be shaped [N]")
    if (
        not x.is_contiguous()
        or not weight.is_contiguous()
        or (bias is not None and not bias.is_contiguous())
    ):
        raise ValueError("x, weight, and bias must be contiguous")
    if x.device != weight.device or (bias is not None and x.device != bias.device):
        raise ValueError("x, weight, and bias must be on the same CUDA device")

    m_size = int(x.size(0))
    total_outputs = m_size * n_size
    if m_size <= 0 or k_size <= 0 or n_size <= 0 or total_outputs <= 0:
        raise ValueError("M, K, and N must be positive")
    return m_size, k_size, n_size, total_outputs, weight_is_nk


def decode_gemv_cute(
    x: Any,
    weight: Any,
    bias: Any | None = None,
    *,
    weight_layout: str = "KN",
) -> Any:
    import torch

    m_size, k_size, n_size, total_outputs, weight_is_nk = _validate_decode_gemv_inputs(
        x,
        weight,
        bias,
        weight_layout,
    )
    cute_mod, from_dlpack, _launch = _load_kernels()

    y = torch.empty((m_size, n_size), device=x.device, dtype=torch.float32)
    bias_tensor = bias if bias is not None else y
    has_bias = 1 if bias is not None else 0

    xc = from_dlpack(x)
    if weight_is_nk:
        wc = from_dlpack(weight)
    else:
        wc = from_dlpack(weight.contiguous())
    bc = from_dlpack(bias_tensor)
    yc = from_dlpack(y)

    _launch(xc, wc, bc, yc, total_outputs, k_size, n_size, weight_is_nk, has_bias)
    torch.cuda.synchronize()
    return y
