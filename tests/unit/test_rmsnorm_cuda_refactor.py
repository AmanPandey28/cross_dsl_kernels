from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import rmsnorm_cuda_compare as comparison


class RmsnormCudaRefactorTests(unittest.TestCase):
    def test_comparison_covers_large_tail_and_fallback(self):
        self.assertEqual(comparison.SHAPES, ((256, 4096), (4096, 4096), (3, 513), (2, 12289)))

    def test_before_loader_preserves_flags_and_isolates_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = root / "before"
            before.mkdir()
            for name in ("rmsnorm_ext.cpp", "rmsnorm_kernel.cu"):
                (before / name).write_text("// preserved source\n")
            with patch.object(comparison, "ROOT", root), patch.dict("os.environ", {}, clear=True):
                with patch("torch.utils.cpp_extension.load") as load:
                    comparison.load_before(None, before)
            options = load.call_args.kwargs
            self.assertEqual(options["extra_cflags"], ["-O3"])
            self.assertEqual(options["extra_cuda_cflags"], ["-O3"])
            self.assertIn("sm120", options["build_directory"])
            self.assertEqual(options["sources"], [str(before / "rmsnorm_ext.cpp"), str(before / "rmsnorm_kernel.cu")])

    def test_source_keeps_bf16_reading_order_and_both_barriers(self):
        source = (ROOT / "csrc/rmsnorm/rmsnorm_kernel.cu").read_text()
        symbols = ("void rmsnorm_bf16_recompute_kernel", "void rmsnorm_bf16_register_kernel",
                   "launch_bf16_rmsnorm(", "void fused_residual_rmsnorm_f32_staged_kernel")
        positions = [source.index(symbol) for symbol in symbols]
        self.assertEqual(positions, sorted(positions))
        fast_reduction = source.split("if constexpr (Fast) {", 1)[1].split("} else {", 1)[0]
        self.assertEqual(fast_reduction.count("__syncthreads();"), 2)
        self.assertIn("if (!fast || hidden > 8192)", source)
        for capacity in (256, 512, 1024, 2048, 4096, 8192):
            self.assertIn(f"rmsnorm_bf16_register_kernel<{capacity}>", source)


if __name__ == "__main__":
    unittest.main()
