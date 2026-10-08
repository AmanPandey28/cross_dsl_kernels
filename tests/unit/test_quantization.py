import unittest

import torch

from crossdsl_kernels.quantization import (
    dequantize_int4,
    pack_int4,
    quantize_int4,
    unpack_int4,
    validate_w4a16,
    w4a16_reference,
)


class QuantizationTests(unittest.TestCase):
    def test_every_nibble_and_odd_tail_roundtrip(self):
        q = torch.arange(-8, 8, dtype=torch.int8).reshape(2, 8)
        for k in range(1, 9):
            packed = pack_int4(q[:, :k])
            self.assertTrue(torch.equal(unpack_int4(packed, k), q[:, :k]))
            if k % 2:
                self.assertTrue(
                    torch.equal(
                        packed[:, -1] >> 4, torch.full((2,), 8, dtype=torch.uint8)
                    )
                )
        self.assertEqual(
            pack_int4(torch.tensor([[-8, 7, 0]], dtype=torch.int8)).tolist(),
            [[240, 136]],
        )

    def test_group_boundary_and_hand_computed_dot(self):
        q = torch.zeros(1, 33, dtype=torch.int8)
        q[0, 0], q[0, 31], q[0, 32] = -8, 7, 3
        scales = torch.tensor([[0.5, 2]], dtype=torch.float16)
        packed = pack_int4(q)
        weight = dequantize_int4(packed, scales, 33, 32)
        self.assertEqual(weight[0, [0, 31, 32]].tolist(), [-4, 3.5, 6])
        y = w4a16_reference(
            torch.ones(1, 33, dtype=torch.float16),
            packed,
            scales,
            torch.tensor([1], dtype=torch.float16),
            group_size=32,
        )
        self.assertEqual(y.item(), 6.5)

    def test_zero_groups_and_quantization_error_bound(self):
        generator = torch.Generator().manual_seed(26)
        for group in (32, 64, 128, 256):
            for k in (1, 31, 32, 33, 127, 128, 129, 257):
                w = torch.randn(3, k, generator=generator).half()
                w[0].zero_()
                packed, scales = quantize_int4(w, group)
                actual = dequantize_int4(packed, scales, k, group)
                self.assertTrue(torch.equal(actual[0], w[0]))
                per_value_scale = scales[:, torch.arange(k) // group].float()
                bound = 0.5 * per_value_scale + actual.float().abs() * 0.001 + 1e-6
                self.assertTrue(
                    bool(((actual.float() - w.float()).abs() <= bound).all())
                )

    def test_fp16_dequantization_rounding_is_part_of_contract(self):
        q = torch.tensor([[3]], dtype=torch.int8)
        scales = torch.tensor([[0.1]], dtype=torch.float16)
        value = dequantize_int4(pack_int4(q), scales, 1, 32)
        self.assertEqual(value.item(), (scales.float() * 3).half().item())

    def test_bad_values_and_groups_are_rejected(self):
        with self.assertRaises(ValueError):
            pack_int4(torch.tensor([[8]], dtype=torch.int8))
        for group in (0, 31, 33, True):
            with self.assertRaises(ValueError):
                quantize_int4(torch.ones(1, 2), group)
        for value in (float("nan"), float("inf"), 1e30):
            with self.assertRaises(ValueError):
                quantize_int4(torch.tensor([[value]]))

    def test_metadata_contract_and_alias_guard(self):
        x = torch.ones(2, 33, dtype=torch.float16)
        packed, scales = quantize_int4(torch.ones(5, 33))
        self.assertEqual(validate_w4a16(x, packed, scales, cuda=False), (2, 33, 5))
        with self.assertRaisesRegex(ValueError, "CUDA"):
            validate_w4a16(x, packed, scales)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_w4a16(
                x, packed, scales, out=x.flatten()[:10].view(2, 5), cuda=False
            )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_w4a16(x.t().contiguous().t(), packed, scales, cuda=False)
        with self.assertRaisesRegex(ValueError, "forward-only"):
            validate_w4a16(x.requires_grad_(), packed, scales, cuda=False)


if __name__ == "__main__":
    unittest.main()
