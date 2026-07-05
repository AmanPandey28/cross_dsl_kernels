from __future__ import annotations

import unittest

try:
    import torch
except Exception:  # pragma: no cover - unittest skip path
    torch = None

from crossdsl_kernels.references.gemv import decode_linear_reference
from crossdsl_kernels.references.rmsnorm import fused_residual_rmsnorm_reference
from crossdsl_kernels.references.rope_kv import rope_gqa_paged_kv_append_reference


@unittest.skipIf(torch is None, "torch is not importable")
class ReferenceTests(unittest.TestCase):
    def test_rmsnorm_reference_matches_manual_fp32(self) -> None:
        gen = torch.Generator(device="cpu").manual_seed(7)
        x = torch.randn(3, 8, generator=gen, dtype=torch.float32)
        residual = torch.randn(3, 8, generator=gen, dtype=torch.float32)
        weight = torch.randn(8, generator=gen, dtype=torch.float32)
        y, r = fused_residual_rmsnorm_reference(x, residual, weight, 1e-5)
        manual_r = x + residual
        manual_y = manual_r * torch.rsqrt((manual_r * manual_r).mean(dim=-1, keepdim=True) + 1e-5) * weight
        self.assertTrue(torch.allclose(r, manual_r))
        self.assertTrue(torch.allclose(y, manual_y))

    def test_decode_linear_reference_matches_matmul(self) -> None:
        gen = torch.Generator(device="cpu").manual_seed(11)
        x = torch.randn(3, 5, generator=gen)
        w = torch.randn(5, 7, generator=gen)
        b = torch.randn(7, generator=gen)
        out = decode_linear_reference(x, w, b, out_dtype=torch.float32)
        self.assertTrue(torch.allclose(out, x @ w + b))

    def test_decode_linear_reference_accepts_nk_physical_weight(self) -> None:
        gen = torch.Generator(device="cpu").manual_seed(12)
        x = torch.randn(2, 5, generator=gen)
        w_kn = torch.randn(5, 7, generator=gen)
        w_nk = w_kn.t().contiguous()
        out = decode_linear_reference(x, w_nk, weight_layout="NK", out_dtype=torch.float32)
        self.assertTrue(torch.allclose(out, x @ w_kn))

    def test_decode_linear_reference_rejects_bad_shapes(self) -> None:
        x = torch.randn(2, 5)
        w = torch.randn(6, 7)
        with self.assertRaisesRegex(ValueError, "K dimension"):
            decode_linear_reference(x, w)

    def test_rope_paged_append_mutates_only_expected_slots(self) -> None:
        tokens, hq, hkv, dim = 3, 4, 2, 8
        gen = torch.Generator(device="cpu").manual_seed(13)
        q = torch.randn(tokens, hq, dim, generator=gen)
        k = torch.randn(tokens, hkv, dim, generator=gen)
        v = torch.randn(tokens, hkv, dim, generator=gen)
        positions = torch.tensor([0, 7, 8], dtype=torch.long)
        page_table = torch.tensor([[2, 0], [1, 3]], dtype=torch.long)
        sequence_ids = torch.tensor([0, 0, 1], dtype=torch.long)
        theta = torch.arange(16, dtype=torch.float32).unsqueeze(1) / 100.0
        freqs = torch.arange(dim // 2, dtype=torch.float32).unsqueeze(0) + 1.0
        cos = torch.cos(theta * freqs)
        sin = torch.sin(theta * freqs)
        sentinel = -999.0
        k_cache = torch.full((4, 8, hkv, dim), sentinel)
        v_cache = torch.full((4, 8, hkv, dim), sentinel)

        q_out = rope_gqa_paged_kv_append_reference(
            q,
            k,
            v,
            cos,
            sin,
            positions,
            page_table,
            sequence_ids,
            k_cache,
            v_cache,
            page_size=8,
            rope_dim=dim,
            interleaved=False,
            kv_layout="NHD",
        )

        self.assertEqual(q_out.shape, q.shape)
        self.assertFalse(torch.all(k_cache == sentinel))
        self.assertFalse(torch.all(v_cache == sentinel))
        self.assertTrue(torch.all(k_cache[1] == sentinel))
        self.assertTrue(torch.allclose(v_cache[2, 0], v[0]))
        self.assertTrue(torch.allclose(v_cache[2, 7], v[1]))
        self.assertTrue(torch.allclose(v_cache[3, 0], v[2]))
        self.assertTrue(torch.allclose(v_cache[1], torch.full_like(v_cache[1], sentinel)))


if __name__ == "__main__":
    unittest.main()
