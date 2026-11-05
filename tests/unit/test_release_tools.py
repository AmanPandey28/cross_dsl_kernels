from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import check_evidence
import comparison_cases
import profile_runner


class ReleaseToolTests(unittest.TestCase):
    def test_saved_sources_and_numerical_evidence(self):
        self.assertEqual(check_evidence.validate()["sources"], 40)

    def test_historical_sources_remain_separate_from_current_code(self):
        root = Path(__file__).resolve().parents[2]
        manifest = json.loads((root / "results/manifest.json").read_text())
        source = "csrc/rmsnorm/rmsnorm_kernel.cu"
        historical = manifest["source_snapshots"]["bf16_20261009"]["production_source_sha256"][source]
        self.assertNotEqual(manifest["production_source_sha256"][source], historical)
        data = json.loads((root / "results/bf16/full.json").read_text())
        self.assertEqual(data["source_sha256"][source], historical)

    def test_projection_profiler_uses_public_runner(self):
        import profile_gemv_clean

        self.assertIs(profile_gemv_clean.run_commands, profile_runner.run_commands)

    def test_cuda_rows_reject_missing_samples_or_failed_graphs(self):
        from check_cuda_evidence import check_rows

        data = {"status": "PASS", "repeat": 2, "inputs_unchanged": True,
                "rows": [{"status": "PASS", "correctness": {"status": "PASS"},
                          "graph_correctness": {"status": "FAIL"},
                          "timing": {"event": {"samples_ms": [1, 2]}}}]}
        with self.assertRaisesRegex(ValueError, "graph"):
            check_rows(data, 1)
        data["rows"][0]["graph_correctness"]["status"] = "PASS"
        data["rows"][0]["timing"]["event"]["samples_ms"] = [1]
        with self.assertRaisesRegex(ValueError, "sample"):
            check_rows(data, 1)

    def test_collection_hashes_use_historical_snapshot(self):
        old = {"production_source_sha256": {"csrc/example.cu": "old"}}
        check_evidence.check_snapshot({"source_sha256": {"csrc/example.cu": "old"}}, old, "run")
        with self.assertRaisesRegex(ValueError, "disagrees with snapshot"):
            check_evidence.check_snapshot({"source_sha256": {"csrc/example.cu": "new"}}, old, "run")

    def test_bf16_profile_preview_uses_standalone_runner(self):
        import profile_rmsnorm_bf16

        self.assertIs(profile_rmsnorm_bf16.run_commands, profile_runner.run_commands)
        jobs = profile_rmsnorm_bf16.commands("python", "EXP-20261103-101", Path("profiles"))
        self.assertEqual(len(jobs), 7)
        self.assertIn("--profile-iterations", jobs[0][1])
        for _, command in jobs[3::2]:
            self.assertIn("--launch-count", command)
            self.assertEqual(command[command.index("--clock-control") + 1], "none")

    def test_sample_validation_rejects_wrong_median_and_nonfinite(self):
        for value in (
            {"samples_ms": [1, 3], "median_ms": 1},
            {"samples_ms": [float("nan")], "median_ms": 0},
        ):
            with self.assertRaises(ValueError):
                check_evidence.check_samples(value)
        check_evidence.check_samples({"samples_ms": [1, 3], "median_ms": 2})

    def test_fresh_output_and_id_required(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            profile_runner.validate_run("EXP-20261008-101", output)
            with self.assertRaisesRegex(ValueError, "experiment ID"):
                profile_runner.validate_run("invalid", output)
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                profile_runner.validate_run("EXP-20261008-101", output)

    def test_cuda_loader_is_self_contained_and_preserves_flags(self):
        try:
            with patch("torch.utils.cpp_extension.load") as load:
                for op in ("rmsnorm", "gemv", "rope_kv"):
                    comparison_cases.cuda_module(op)
                    args = load.call_args.kwargs
                    self.assertTrue(all(Path(p).is_file() for p in args["sources"]))
                    self.assertEqual(args["extra_cuda_cflags"], ["-O3"])
                    self.assertEqual(args["extra_cflags"], ["-O3"])
                    self.assertEqual(args["extra_ldflags"],
                                     ["-lcublas", "-lcublasLt"] if op == "gemv" else [])
                with self.assertRaisesRegex(ValueError, "unknown CUDA workload"):
                    comparison_cases.cuda_module("invalid")
        finally:
            comparison_cases.cuda_module.cache_clear()

    def test_failed_job_stops_and_saves_log(self):
        process = MagicMock()
        process.__enter__.return_value = process
        process.communicate.return_value = ("failed correctly", None)
        process.returncode = 1
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            with patch.object(profile_runner.subprocess, "Popen", return_value=process) as run:
                result = profile_runner.run_commands("EXP-20261008-101", output,
                                                    [("first", ["one"]), ("second", ["two"])])
            self.assertEqual(result, 1)
            run.assert_called_once()
            self.assertEqual((output / "first.log").read_text(), "failed correctly")
            self.assertEqual(len(json.loads((output / "manifest.json").read_text())["jobs"]), 1)

    def test_missing_tool_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            with patch.object(profile_runner.subprocess, "Popen", side_effect=FileNotFoundError("missing tool")):
                result = profile_runner.run_commands("EXP-20261008-101", output,
                                                    [("missing", ["missing-tool"])])
            self.assertEqual(result, 127)
            self.assertIn("missing tool", (output / "missing.log").read_text())

    def test_timeout_kills_only_own_process_group_and_preserves_log(self):
        process = MagicMock()
        process.__enter__.return_value = process
        process.pid = 1234
        process.communicate.side_effect = [subprocess.TimeoutExpired(["ncu"], 600),
                                           ("partial log", None)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            with (
                patch.object(profile_runner.subprocess, "Popen", return_value=process) as run,
                patch.object(profile_runner.os, "killpg") as kill,
            ):
                result = profile_runner.run_commands("EXP-20261008-101", output, [("ncu", ["ncu"])])
            self.assertEqual(result, 124)
            self.assertTrue(run.call_args.kwargs["start_new_session"])
            kill.assert_called_once_with(1234, profile_runner.signal.SIGKILL)
            self.assertIn("partial log", (output / "ncu.log").read_text())


if __name__ == "__main__":
    unittest.main()
