from __future__ import annotations

from functools import lru_cache
from typing import Any

from cuda.bindings import driver as cuda


@lru_cache(maxsize=1)
def _load_kernels() -> Any:
    from cutlass import cute
    from cutlass._mlir.dialects import nvvm
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cutlass_dsl import T

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
            # Four shorter sums reduce FP32 error and dependency-chain length.
            acc0 = 0.0
            acc1 = 0.0
            acc2 = 0.0
            acc3 = 0.0
            for kk in range(0, k_size, 4):
                if weight_is_nk != 0:
                    acc0 = acc0 + x[row, kk] * weight[col, kk]
                    if kk + 1 < k_size:
                        acc1 = acc1 + x[row, kk + 1] * weight[col, kk + 1]
                    if kk + 2 < k_size:
                        acc2 = acc2 + x[row, kk + 2] * weight[col, kk + 2]
                    if kk + 3 < k_size:
                        acc3 = acc3 + x[row, kk + 3] * weight[col, kk + 3]
                else:
                    acc0 = acc0 + x[row, kk] * weight[kk, col]
                    if kk + 1 < k_size:
                        acc1 = acc1 + x[row, kk + 1] * weight[kk + 1, col]
                    if kk + 2 < k_size:
                        acc2 = acc2 + x[row, kk + 2] * weight[kk + 2, col]
                    if kk + 3 < k_size:
                        acc3 = acc3 + x[row, kk + 3] * weight[kk + 3, col]
            acc = (acc0 + acc1) + (acc2 + acc3)
            if has_bias != 0:
                acc = acc + bias[col]
            y[row, col] = acc

    @cute.jit
    def _launch(
        xc,
        wc,
        bc,
        yc,
        total_outputs,
        k_size,
        n_size,
        weight_is_nk,
        has_bias,
        stream: cuda.CUstream,
    ):
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
        ).launch(grid=[blocks, 1, 1], block=[threads, 1, 1], stream=stream)

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
    _, from_dlpack, _launch = _load_kernels()

    y = torch.empty((m_size, n_size), device=x.device, dtype=torch.float32)
    bias_tensor = bias if bias is not None else y
    has_bias = 1 if bias is not None else 0

    xc = from_dlpack(x)
    wc = from_dlpack(weight)
    bc = from_dlpack(bias_tensor)
    yc = from_dlpack(y)

    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    _launch(
        xc, wc, bc, yc, total_outputs, k_size, n_size, weight_is_nk, has_bias, stream
    )
    return y
