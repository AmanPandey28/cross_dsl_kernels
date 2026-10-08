"""Teaching INT4 format: NK storage, low nibble first, code = q + 8."""

from __future__ import annotations

import torch

GROUP_SIZES = (32, 64, 128, 256)


def _check_group_size(group_size: int) -> None:
    if type(group_size) is not int or group_size not in GROUP_SIZES:
        raise ValueError(f"group_size must be one of {GROUP_SIZES}")


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack signed [-8, 7] integers [N, K] into uint8 [N, ceil(K/2)]."""
    if q.ndim != 2 or min(q.shape) < 1 or q.dtype != torch.int8:
        raise ValueError("q must be a nonempty int8 matrix [N, K]")
    if bool(((q < -8) | (q > 7)).any()):
        raise ValueError("INT4 values must lie in [-8, 7]")
    codes = q.to(torch.int16) + 8
    if q.shape[1] % 2:
        codes = torch.nn.functional.pad(codes, (0, 1), value=8)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).contiguous()


def unpack_int4(packed: torch.Tensor, k: int) -> torch.Tensor:
    """Decode both nibbles independently; ignore the padded high nibble."""
    if type(k) is not int or k < 1:
        raise ValueError("K must be a positive integer")
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("packed must be a uint8 matrix")
    if packed.shape[0] < 1 or packed.shape[1] != (k + 1) // 2:
        raise ValueError("packed shape must be [N, ceil(K/2)]")
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
    return (codes[:, :k].to(torch.int16) - 8).to(torch.int8).contiguous()


def quantize_int4(weight: torch.Tensor, group_size: int = 128):
    """Round-to-nearest symmetric quantization, not AWQ/GPTQ calibration.

    Setup only: this function checks values and can synchronize a CUDA tensor.
    Scales are rounded to FP16 BEFORE choosing q. Zero groups use scale one.
    """
    _check_group_size(group_size)
    if weight.ndim != 2 or min(weight.shape) < 1 or not weight.is_floating_point():
        raise ValueError("weight must be a nonempty floating-point [N, K] matrix")
    if weight.requires_grad:
        raise ValueError("quantization is forward-only; detach weight explicitly")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("weight must be finite")
    n, k = weight.shape
    groups = (k + group_size - 1) // group_size
    padded = torch.nn.functional.pad(weight.float(), (0, groups * group_size - k))
    blocks = padded.reshape(n, groups, group_size)
    peak = blocks.abs().amax(dim=-1)
    scale = torch.where(peak == 0, 1.0, (peak / 7).clamp_min(2**-24)).half()
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("group scales cannot be represented in FP16")
    q = (blocks / scale.float().unsqueeze(-1)).round().clamp(-7, 7).to(torch.int8)
    if not bool(torch.isfinite((q.float() * scale.float().unsqueeze(-1)).half()).all()):
        raise ValueError("dequantized weights cannot be represented in FP16")
    return pack_int4(q.flatten(1)[:, :k]), scale.contiguous()


def dequantize_int4(packed, scales, k: int, group_size: int = 128):
    """Materialize [N, K] FP16 weights, including the specified FP16 rounding."""
    _check_group_size(group_size)
    q = unpack_int4(packed, k)
    expected = (q.shape[0], (k + group_size - 1) // group_size)
    if scales.dtype != torch.float16 or tuple(scales.shape) != expected:
        raise ValueError("scales must be FP16 [N, ceil(K/group_size)]")
    if scales.device != packed.device:
        raise ValueError("packed and scales must be on the same device")
    group = torch.arange(k, device=packed.device) // group_size
    return (q.float() * scales[:, group].float()).half().contiguous()


def validate_w4a16(
    x, packed, scales, bias=None, *, group_size=128, out=None, cuda=True
):
    """Metadata-only launch checks; finite inputs/scales are a caller precondition."""
    _check_group_size(group_size)
    if x.ndim != 2 or x.dtype != torch.float16:
        raise ValueError("x must be FP16 [M, K]")
    m, k = x.shape
    if not 1 <= m <= 16 or k < 1:
        raise ValueError("decode requires 1 <= M <= 16 and K > 0")
    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError("packed must be uint8 [N, ceil(K/2)]")
    n = packed.shape[0]
    if n < 1 or packed.shape[1] != (k + 1) // 2:
        raise ValueError("packed shape must be [N, ceil(K/2)]")
    if scales.dtype != torch.float16 or tuple(scales.shape) != (
        n,
        (k + group_size - 1) // group_size,
    ):
        raise ValueError("scales must be FP16 [N, ceil(K/group_size)]")
    if bias is not None and (bias.dtype != torch.float16 or tuple(bias.shape) != (n,)):
        raise ValueError("bias must be FP16 [N] or None")
    tensors = (x, packed, scales) + (() if bias is None else (bias,))
    if out is not None:
        if out.dtype != torch.float16 or tuple(out.shape) != (m, n):
            raise ValueError("out must be FP16 [M, N]")
        tensors += (out,)
    if cuda and not x.is_cuda:
        raise ValueError("W4A16 kernels require CUDA tensors")
    if any(t.device != x.device or not t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must be contiguous and on the same device")
    if any(t.requires_grad for t in tensors):
        raise ValueError("W4A16 kernels are forward-only; detach tensors explicitly")
    if max(m, k, n, *(t.numel() for t in tensors)) >= 2**31:
        raise ValueError("this prototype requires signed 32-bit indexing")
    if out is not None:
        lo, hi = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
        if any(
            lo < t.data_ptr() + t.numel() * t.element_size() and t.data_ptr() < hi
            for t in tensors[:-1]
        ):
            raise ValueError("out must not overlap an input")
    return m, k, n


def w4a16_reference(x, packed, scales, bias=None, *, group_size=128):
    """Independent FP64 dot of FP16 inputs and dequantized FP16 weights."""
    validate_w4a16(x, packed, scales, bias, group_size=group_size, cuda=False)
    weight = dequantize_int4(packed, scales, x.shape[1], group_size)
    y = x.double() @ weight.double().t()
    if bias is not None:
        y += bias.double()
    return y.half()
