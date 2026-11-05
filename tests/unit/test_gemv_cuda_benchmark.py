from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gemv_cuda_benchmark as driver


class GemvCudaBenchmarkTests(unittest.TestCase):
    def test_required_cases_and_tails(self):
        cases = driver.projection_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual({s["m"] for s in cases if s["name"].startswith("hidden_m")}, {1, 2, 4, 8, 16})
        self.assertEqual({s["layout"] for s in cases if s["m"] == 5}, {"KN", "NK"})
        with self.assertRaisesRegex(ValueError, "unknown cases"):
            driver.select_cases("missing")
        self.assertEqual(len(driver.select_cases("hidden_m1,hidden_m16")), 2)

    def test_backend_mapping_uses_separate_module(self):
        case = {"args": ("x", "w", None), "kwargs": {"weight_layout": "NK"}}
        module = MagicMock()
        with patch.object(driver, "simt_module", return_value=module):
            for backend, entry in driver.SIMT_ENTRY_POINTS.items():
                driver.prepare(backend, case)()
                getattr(module, entry).assert_called_once_with("x", "w", None, "NK")
        self.assertEqual(driver.SIMT_SOURCES, ("gemv_simt_ext.cpp", "gemv_simt.cu", "gemm_tiled.cu"))
        self.assertEqual(driver.SIMT_ENTRY_POINTS, {
            "simt_warp": "decode_gemv_simt_warp", "simt_rows": "decode_gemv_simt_rows",
            "tiled": "decode_gemm_tiled"})
        for name in driver.SIMT_SOURCES:
            source = ROOT / "csrc/gemv" / name
            self.assertTrue(source.is_file())
            self.assertNotIn("clean", source.read_text())
        with self.assertRaisesRegex(ValueError, "unknown backend"):
            driver.prepare("missing", case)

    def test_counterbalanced_orders(self):
        backends = list(driver.BACKENDS)
        first = driver.backend_orders(backends, 30, 19)
        self.assertEqual(first, driver.backend_orders(backends, 30, 19))
        for i in range(0, 30, 2):
            self.assertEqual(set(first[i]), set(backends))
            self.assertEqual(first[i + 1], first[i][::-1])

    def test_fp64_oracle_layout_equivalence_and_original_gate(self):
        import torch
        from comparison_cases import check_output
        from crossdsl_kernels.tolerances import fp32_gemv_tolerance

        x = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
        w = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
        bias = torch.tensor([0.5, 1.0, -2.0])
        expected = torch.tensor([[9.5, 9.0, -11.0], [11.5, 18.0, -23.0]])
        for layout, weight in (("KN", w), ("NK", w.t().contiguous())):
            case = {"args": (x, weight, bias), "kwargs": {"weight_layout": layout},
                    "tolerance": fp32_gemv_tolerance(2)}
            case["expected"] = driver.independent_expected(case)
            self.assertTrue(torch.equal(case["expected"][0], expected))
            self.assertEqual(check_output("gemv", expected, case)["status"], "PASS")
            self.assertEqual(check_output("gemv", expected + 0.01, case)["status"], "FAIL")
            self.assertEqual(check_output("gemv", expected * float("nan"), case)["status"], "FAIL")

    def test_sample_timer_primes_before_measurement(self):
        torch = MagicMock()
        start, end = MagicMock(), MagicMock()
        torch.cuda.Event.side_effect = [start, end]
        start.elapsed_time.return_value = 6.0
        timer = driver.make_timer(torch)
        self.assertEqual(start.record.call_count, 1)
        self.assertEqual(end.synchronize.call_count, 1)
        run = MagicMock()
        self.assertEqual(timer(run, 3)[0], 2.0)
        run.assert_called_once()
        self.assertEqual(start.record.call_count, 2)
        self.assertEqual(torch.cuda.Event.call_count, 2)

    def test_profile_jobs_select_new_and_historical_kernels(self):
        import profile_gemv_simt

        jobs = profile_gemv_simt.commands("python", "EXP-20261105-105", Path("profiles"))
        self.assertEqual(len(jobs), 17)
        for _, command in jobs:
            if "--experiment" in command:
                self.assertIn("benchmarks/gemv_cuda_benchmark.py", command)
        self.assertIn("--cuda-graph-trace=node", jobs[1][1])
        self.assertEqual(jobs[1][1][jobs[1][1].index("--graph-calls") + 1], "2")
        for name, command in jobs[3::2]:
            self.assertEqual(command[command.index("--launch-count") + 1], "1")
            self.assertEqual(command[command.index("--launch-skip") + 1], "6")
            self.assertIn("--diagnostic", command)
            self.assertEqual(command[command.index("--graph-calls") + 1], "0")
        self.assertTrue(any(name == "hidden_nk_m16--tiled" for name, _ in jobs))
        for job, symbol in (("hidden_m16--simt_warp", "gemv_warp_kernel"),
                            ("hidden_m16--simt_rows", "gemv_rows_kn_kernel"),
                            ("hidden_nk_m16--simt_rows", "gemv_rows_nk_kernel")):
            command = dict(jobs)[job]
            pattern = command[command.index("--kernel-name") + 1].removeprefix("regex:")
            self.assertIsNotNone(re.search(pattern, f"void {symbol}(...)"))
        old_rows = dict(jobs)["hidden_m16--cuda_fast"]
        pattern = old_rows[old_rows.index("--kernel-name") + 1].removeprefix("regex:")
        for name in ("void decode_gemv_rows_kernel<false>(...)",
                     "void decode_gemv_rows_kernel<(bool)0>(...)"):
            self.assertIsNotNone(re.search(pattern, name))
        counters = profile_gemv_simt.commands("python", "EXP-20261105-106", Path("profiles"), True)
        self.assertEqual(len(counters), 14)
        self.assertIn("dram__bytes_op_read.sum", counters[0][1][counters[0][1].index("--metrics") + 1])

    def test_analysis_rejects_missing_or_changed_samples(self):
        from summarize_gemv_simt import analyze

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            data = {"rows": [{"case": "one", "backend": "tiled", "status": "PASS",
                              "timing": {"event": {"median_ms": 9, "samples_ms": [1, 3]}}}], "repeat": 2}
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "median"):
                analyze(path)
            data["rows"][0]["timing"]["event"]["median_ms"] = 2
            data["repeat"] = 30
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "sample count"):
                analyze(path)

    def test_counter_byte_units_are_not_assumed(self):
        from summarize_gemv_simt import counter_value, profiles

        value = counter_value("dram_bytes", "1.25", "Gbyte")
        self.assertEqual(value["value"], 1.25e9)
        self.assertEqual(value["unit"], "byte")
        self.assertEqual(value["raw_unit"], "Gbyte")
        self.assertEqual(counter_value("duration_us", "2", "ms")["value"], 2000)
        with self.assertRaisesRegex(ValueError, "unknown DRAM"):
            counter_value("dram_bytes", "1", "unknown")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no captured kernel"):
                profiles(Path(directory))

    def test_historical_backend_aliases_preserve_raw_rows(self):
        from summarize_gemv_simt import analyze

        data = {"experiment_id": "archived", "status": "PASS", "seed": 1, "warmup": 1,
                "repeat": 1, "graph_calls": 1, "timing_policy": "test", "rows": [
                    {"case": "one", "backend": backend, "status": "PASS",
                     "timing": {"event": {"median_ms": value, "samples_ms": [value]}}}
                    for backend, value in (("cuda", 2.0), ("clean_warp", 1.0))]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            raw = json.dumps(data)
            path.write_text(raw)
            summary = analyze(path)
            self.assertEqual(path.read_text(), raw)
            self.assertEqual(summary["rows"], data["rows"])
            self.assertEqual(summary["comparisons"][0]["candidate"], "simt_warp")
            self.assertEqual(summary["comparisons"][0]["baseline_over_candidate"], 2.0)
            data["rows"].append({**data["rows"][1], "backend": "simt_warp"})
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "duplicate backend"):
                analyze(path)


if __name__ == "__main__":
    unittest.main()
