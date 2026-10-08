from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from compare import profile_torch, select_cases


class ComparisonDriverTests(unittest.TestCase):
    def test_case_filters_reject_unknown_names(self):
        catalog = BENCHMARKS / "shapes/comparison.json"
        with self.assertRaisesRegex(ValueError, "unknown cases"):
            select_cases(catalog, "all", "not_a_case", False)
        self.assertEqual(len(select_cases(catalog, "gemv", "", False)), 13)

    def test_trace_metadata_accepts_relative_and_external_paths(self):
        profiler = MagicMock()
        profiler.__enter__.return_value = profiler
        profiler.events.return_value = []
        profiler.key_averages.return_value.table.return_value = "table"
        profiler.export_chrome_trace.side_effect = lambda p: Path(p).write_text("{}")
        module = SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CPU=0, CUDA=1),
            profile=lambda **kwargs: profiler,
            record_function=lambda label: MagicMock(),
        )
        torch = SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: None),
            autograd=SimpleNamespace(DeviceType=SimpleNamespace(CUDA=1)),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "external", Path("results") / "test-export"]
            with patch.dict(sys.modules, {"torch.profiler": module}):
                for path in paths:
                    try:
                        metadata = profile_torch(torch, lambda: None, path, "test", 1)
                        self.assertTrue(Path(metadata["trace"]).is_absolute())
                        self.assertEqual(
                            json.loads(Path(metadata["trace"]).read_text()), {}
                        )
                    finally:
                        path.with_suffix(".trace.json").unlink(missing_ok=True)
                        path.with_suffix(".txt").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
