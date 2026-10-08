from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from compare import verify_replay
from comparison_cases import check_output

from crossdsl_kernels.contracts import Evidence, KernelContract, validate_cache_storage

spec = importlib.util.spec_from_file_location(
    "cute_launch_under_test", ROOT / "src/crossdsl_kernels/cute/launch.py"
)
assert spec is not None and spec.loader is not None
launch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch_module)
prepare_launch = launch_module.prepare_launch


class KernelContractTests(unittest.TestCase):
    def test_existing_contract_types_remain_available(self):
        contract = KernelContract("rmsnorm", "row_major", "fp32", "fp32", "none")
        self.assertEqual(contract.evidence, Evidence.DESIGNED)

    def test_nonfinite_result_fails_and_remains_json_serializable(self):
        expected = torch.ones(2)
        case = {
            "expected": (expected,),
            "tolerance": {"max_abs_err": 1e-6, "rel_l2_err": 1e-5},
        }
        result = check_output("gemv", torch.full_like(expected, float("nan")), case)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["errors"][0]["finite"])
        json.dumps(result, allow_nan=False)

    def test_prepared_cute_drops_constexpr_only_from_runtime_call(self):
        compiled = MagicMock()
        cute = SimpleNamespace(compile=MagicMock(return_value=compiled))
        cuda = SimpleNamespace(CUstream=lambda stream: stream)
        modules = {
            "cutlass": SimpleNamespace(cute=cute),
            "cuda.bindings": SimpleNamespace(driver=cuda),
            "cutlass.cute.runtime": SimpleNamespace(
                from_dlpack=lambda tensor: "adapter"
            ),
        }
        tensor = SimpleNamespace(device="cuda:0")
        stream = SimpleNamespace(cuda_stream=17)
        with (
            patch.dict(sys.modules, modules),
            patch("torch.cuda.current_stream", return_value=stream),
        ):
            run = prepare_launch(
                "launcher", (tensor,), (4096, 1e-5, 16), constexpr_scalars=(0,)
            )
            run()
        cute.compile.assert_called_once_with("launcher", "adapter", 4096, 1e-5, 16, 17)
        compiled.assert_called_once_with("adapter", 1e-5, 16, 17)

    def test_overlapping_contiguous_views_are_rejected(self):
        storage = torch.zeros(32)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_cache_storage((), storage[:16], storage[8:24])
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_cache_storage((storage[4:8],), storage[:16], torch.zeros(16))

    def test_disjoint_views_of_same_storage_are_allowed(self):
        storage = torch.zeros(32)
        validate_cache_storage((), storage[:16], storage[16:])

    def test_noop_graph_cannot_pass_using_previous_cache_writes(self):
        expected = tuple(torch.ones(2) for _ in range(3))
        output, kc, vc = (t.clone() for t in expected)
        case = {
            "args": (kc, vc),
            "expected": expected,
            "tolerance": {"max_abs_err": 1e-6, "rel_l2_err": 1e-5},
        }
        runtime = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        graph = SimpleNamespace(replay=lambda: None)
        self.assertEqual(
            verify_replay(runtime, graph, output, case, "rope_kv")["status"], "FAIL"
        )

        def replay():
            for actual, target in zip((output, kc, vc), expected):
                actual.copy_(target)

        graph.replay = replay
        self.assertEqual(
            verify_replay(runtime, graph, output, case, "rope_kv")["status"], "PASS"
        )


if __name__ == "__main__":
    unittest.main()
