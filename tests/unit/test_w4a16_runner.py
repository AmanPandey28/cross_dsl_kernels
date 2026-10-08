import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module("w4a16_runner", "benchmarks/w4a16_benchmark.py")
profiler = load_module("w4a16_profiler", "scripts/profile_w4a16.py")


class W4A16RunnerTests(unittest.TestCase):
    def test_gate_distinguishes_rounding_from_error(self):
        reference = torch.tensor([0.0, 10.0, -100.0])
        self.assertEqual(runner.check_output(reference, reference)["status"], "PASS")
        self.assertEqual(
            runner.check_output(reference + 1, reference)["status"], "FAIL"
        )
        near_zero = torch.tensor([0.003, 10.0, -100.0])
        self.assertEqual(runner.check_output(near_zero, reference)["status"], "FAIL")
        for bad in (float("nan"), float("inf")):
            output = torch.tensor([bad, 10.0, -100.0])
            check = runner.check_output(output, reference)
            self.assertEqual(check["status"], "FAIL")
            self.assertFalse(check["finite"])

    def test_profile_jobs_are_bounded_and_use_the_same_driver(self):
        jobs = profiler.commands("python", "EXP-20261005-028", Path("profiles"))
        self.assertEqual(len(jobs), 15)
        collectors = [cmd for _, cmd in jobs if "benchmarks/w4a16_benchmark.py" in cmd]
        self.assertEqual(len(collectors), 8)
        for cmd in collectors:
            self.assertEqual(cmd[cmd.index("--graph-calls") + 1], "0")
            self.assertIn("--output", cmd)
        ncu = [cmd for _, cmd in jobs if cmd[0] == "ncu" and "--export" in cmd]
        self.assertEqual(len(ncu), 6)
        for cmd in ncu:
            self.assertEqual(cmd[cmd.index("--launch-count") + 1], "1")
            self.assertEqual(cmd[cmd.index("--cases") + 1], "hidden_m16")


if __name__ == "__main__":
    unittest.main()
