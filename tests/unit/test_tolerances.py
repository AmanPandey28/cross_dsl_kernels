from __future__ import annotations

import unittest

from crossdsl_kernels.tolerances import fp32_gemv_status, fp32_gemv_tolerance


class GemvToleranceTest(unittest.TestCase):
    def test_abs_tolerance_keeps_existing_1024_policy(self) -> None:
        tolerance = fp32_gemv_tolerance(1024)
        self.assertAlmostEqual(tolerance["max_abs_err"], 2.0e-4)
        self.assertAlmostEqual(tolerance["rel_l2_err"], 1.0e-5)

    def test_abs_tolerance_scales_with_reduction_length(self) -> None:
        tolerance = fp32_gemv_tolerance(4096)
        self.assertAlmostEqual(tolerance["max_abs_err"], 4.0e-4)

    def test_status_requires_abs_and_matrix_relative_error(self) -> None:
        self.assertEqual(fp32_gemv_status(4096, 3.1e-4, 7.0e-7)[0], "PASS")
        self.assertEqual(fp32_gemv_status(4096, 4.1e-4, 7.0e-7)[0], "FAIL")
        self.assertEqual(fp32_gemv_status(4096, 3.1e-4, 1.1e-5)[0], "FAIL")


if __name__ == "__main__":
    unittest.main()
