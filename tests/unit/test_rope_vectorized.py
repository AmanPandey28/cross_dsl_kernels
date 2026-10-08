from __future__ import annotations

import unittest

import torch

from crossdsl_kernels.references.rope_kv import (
    rope_gqa_paged_kv_append_reference,
    rope_gqa_paged_kv_append_vectorized,
    validate_paged_kv_indices,
)


class VectorizedRopeTests(unittest.TestCase):
    def test_matches_loop_reference_and_preserves_unwritten_slots(self) -> None:
        gen = torch.Generator().manual_seed(41)
        for layout in ("NHD", "HND"):
            for interleaved in (False, True):
                with self.subTest(layout=layout, interleaved=interleaved):
                    q = torch.randn(5, 6, 10, generator=gen)
                    k = torch.randn(5, 2, 10, generator=gen)
                    v = torch.randn(5, 2, 10, generator=gen)
                    positions = torch.tensor([0, 7, 8, 15, 16])
                    sequences = torch.tensor([0, 0, 0, 1, 1])
                    table = torch.tensor([[4, 1, 5], [2, 0, 3]])
                    theta = torch.randn(24, 4, generator=gen)
                    shape = (7, 8, 2, 10) if layout == "NHD" else (7, 2, 8, 10)
                    ref_k = torch.full(shape, -999.0)
                    ref_v = torch.full(shape, -999.0)
                    out_k, out_v = ref_k.clone(), ref_v.clone()
                    args = (
                        q,
                        k,
                        v,
                        theta.cos(),
                        theta.sin(),
                        positions,
                        table,
                        sequences,
                    )
                    kwargs = {
                        "page_size": 8,
                        "rope_dim": 8,
                        "interleaved": interleaved,
                        "kv_layout": layout,
                    }
                    validate_paged_kv_indices(
                        positions,
                        table,
                        sequences,
                        page_size=8,
                        num_cache_pages=7,
                        max_position=24,
                    )
                    ref_q = rope_gqa_paged_kv_append_reference(
                        *args, ref_k, ref_v, **kwargs
                    )
                    out_q = rope_gqa_paged_kv_append_vectorized(
                        *args, out_k, out_v, **kwargs
                    )
                    self.assertTrue(torch.equal(out_q, ref_q))
                    self.assertTrue(torch.equal(out_k, ref_k))
                    self.assertTrue(torch.equal(out_v, ref_v))
                    self.assertTrue(torch.equal(out_q[..., 8:], q[..., 8:]))

    def test_rejects_invalid_or_duplicate_destinations(self) -> None:
        cases = [
            ([0], [[0]], [1], "sequence"),
            ([-1], [[0]], [0], "position"),
            ([16], [[0]], [0], "position"),
            ([0], [[3]], [0], "physical page"),
            ([0, 0], [[0]], [0, 0], "distinct"),
            ([0, 0], [[0], [0]], [0, 1], "distinct"),
        ]
        for positions, table, sequences, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                validate_paged_kv_indices(
                    torch.tensor(positions),
                    torch.tensor(table),
                    torch.tensor(sequences),
                    page_size=8,
                    num_cache_pages=3,
                    max_position=16,
                )


if __name__ == "__main__":
    unittest.main()
