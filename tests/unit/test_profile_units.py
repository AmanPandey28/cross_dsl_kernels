from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from summarize_comparison import metric_value


class ProfilerUnitTests(unittest.TestCase):
    def test_ncu_duration_units_normalize_to_microseconds(self):
        for value, unit, expected in (
            ("4.59", "ms", 4590.0),
            ("990.34", "us", 990.34),
            ("1,000", "ns", 1.0),
            ("0.001", "s", 1000.0),
            ("4.59", "msecond", 4590.0),
        ):
            result = metric_value("duration_us", value, unit)
            self.assertAlmostEqual(result["value"], expected)
            self.assertEqual(result["unit"], "us")
            self.assertEqual(result["raw_unit"], unit)

    def test_unknown_duration_units_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown NCU duration unit"):
            metric_value("duration_us", "2", "cycles")

    def test_other_metrics_retain_units(self):
        self.assertEqual(
            metric_value("dram_gbps", "263.74", "Gbyte/second"),
            {"value": 263.74, "unit": "Gbyte/second"},
        )


if __name__ == "__main__":
    unittest.main()
