from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))

from rmsnorm_bf16 import check_rms_output, counterbalanced_orders, make_rms_case
from summarize_rmsnorm_bf16 import paired_interval

from crossdsl_kernels.references.rmsnorm import fused_residual_rmsnorm_reference


class Bf16RmsnormTests(unittest.TestCase):
    def test_reference_keeps_unrounded_sum_for_normalization(self):
        case = make_rms_case(2, 513, 71, device="cpu")
        x, residual, weight, eps = case["args"]
        originals = tuple(t.clone() for t in (x, residual, weight))
        y, r_out = fused_residual_rmsnorm_reference(*case["args"])
        r = x.float() + residual.float()
        expected = (
            r * torch.rsqrt(r.square().mean(-1, keepdim=True) + eps) * weight.float()
        ).bfloat16()
        rounded = r.bfloat16().float()
        wrong = (
            rounded
            * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + eps)
            * weight.float()
        ).bfloat16()
        self.assertTrue(torch.equal(y, expected))
        self.assertFalse(torch.equal(y, wrong))
        self.assertTrue(torch.equal(r_out, r.bfloat16()))
        self.assertTrue(
            all(torch.equal(t, old) for t, old in zip((x, residual, weight), originals))
        )

    def test_guard_rejects_wrong_semantics_and_nonfinite_outputs(self):
        case = make_rms_case(2, 513, 71, device="cpu")
        self.assertEqual(check_rms_output(case["expected"], case)["status"], "PASS")
        y, r = case["expected"]
        self.assertEqual(check_rms_output((y + 1, r), case)["status"], "FAIL")
        self.assertEqual(check_rms_output((y, r + 1), case)["status"], "FAIL")
        self.assertEqual(
            check_rms_output((y * float("nan"), r), case)["status"], "FAIL"
        )

    def test_zero_norm_and_tail_views(self):
        case = make_rms_case(3, 513, 1, device="cpu", pattern="zero", offset=1)
        self.assertTrue(case["args"][0].is_contiguous())
        self.assertEqual(case["args"][0].storage_offset(), 1)
        self.assertEqual(check_rms_output(case["expected"], case)["rel_l2_err"], 0)
        self.assertEqual(case["logical_bytes"], 2 * (4 * 3 * 513 + 513))

    def test_orders_reverse_each_pair_and_are_reproducible(self):
        backends = ["cuda_fast", "triton_fast", "pytorch", "torch_compile"]
        orders = counterbalanced_orders(backends, 20, 19)
        self.assertEqual(orders, counterbalanced_orders(backends, 20, 19))
        for index in range(0, 20, 2):
            self.assertEqual(sorted(orders[index]), sorted(backends))
            self.assertEqual(orders[index + 1], list(reversed(orders[index])))

    def test_block_interval_preserves_pair_alignment(self):
        result = paired_interval([2, 4, 8, 16], [1, 2, 4, 8])
        self.assertEqual(result["paired_median_speedup"], 2)
        self.assertEqual(result["block_bootstrap_95_interval"], [2, 2])
        with self.assertRaises(ValueError):
            paired_interval([2, 4], [1])


if __name__ == "__main__":
    unittest.main()
