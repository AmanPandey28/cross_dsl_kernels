#!/usr/bin/env python3
"""Preview or collect a bounded profile bundle for the parallel CUDA kernels."""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from profile_runner import run_commands


COUNTER_METRICS = (
    "gpu__time_duration.sum", "dram__bytes.sum", "dram__bytes_op_read.sum", "dram__bytes_op_write.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
)


def commands(python, experiment, output, counters_only=False):
    base = [python, "benchmarks/gemv_cuda_benchmark.py", "--experiment", experiment,
            "--warmup", "5", "--repeat", "2", "--graph-calls", "0", "--diagnostic"]
    systems_base = list(base)
    systems_base[systems_base.index("--graph-calls") + 1] = "2"
    selected = ["--cases", "hidden_m1,hidden_m16,mlp_up_m16"]
    jobs = [
        ("torch", base + selected + ["--profile-iterations", "3", "--output", str(output / "torch")]),
        ("nsys", ["nsys", "profile", "--trace=cuda,nvtx,cublas", "--sample=none", "--cpuctxsw=none",
                  "--cuda-graph-trace=node", "--output", str(output / "gemv-clean")]
                 + systems_base + selected + ["--output", str(output / "nsys-results")]),
        ("nsys-stats", ["nsys", "stats", "--report", "cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum",
                        "--format", "csv", "--output", str(output / "gemv-clean-stats"),
                        str(output / "gemv-clean.nsys-rep")]),
    ]
    if counters_only:
        jobs = []
    for case, backend, kernel in (
        ("hidden_m16", "cuda", ".*decode_gemv_f32_warp_kernel.*"),
        ("hidden_m16", "cuda_fast", ".*decode_gemv_rows_kernel.*"),
        ("hidden_m16", "clean_warp", ".*gemv_warp_clean_kernel.*"),
        ("hidden_m16", "clean_rows", ".*gemv_rows_kn_clean_kernel.*"),
        ("hidden_m16", "tiled", ".*gemm_tiled_kernel.*"),
        ("hidden_nk_m16", "clean_rows", ".*gemv_rows_nk_clean_kernel.*"),
        ("hidden_nk_m16", "tiled", ".*gemm_tiled_kernel.*"),
    ):
        name = f"{case}--{backend}"
        # Each process runs exactly one layout/case; bool demangling varies by tool version.
        report = output / f"{name}.ncu-rep"
        collect = ["ncu", "--target-processes", "all", "--kernel-name-base", "demangled",
                   "--kernel-name", f"regex:{kernel}", "--launch-skip", "6", "--launch-count", "1",
                   "--clock-control", "none"]
        if counters_only:
            collect += ["--metrics", ",".join(COUNTER_METRICS)]
        else:
            for section in ("SpeedOfLight", "MemoryWorkloadAnalysis", "LaunchStats", "Occupancy",
                            "SchedulerStats", "WarpStateStats"):
                collect += ["--section", section]
        collect += ["--export", str(report)] + base + ["--cases", case, "--backends", backend,
                                                       "--output", str(output / f"{name}-results")]
        jobs += [(name, collect), (f"{name}-csv", ["ncu", "--import", str(report), "--page", "raw", "--csv", "--print-units", "base"])]
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--counters-only", action="store_true")
    args = parser.parse_args()
    jobs = commands(args.python, args.experiment, args.output, args.counters_only)
    for _, command in jobs:
        print(shlex.join(command), flush=True)
    return run_commands(args.experiment, args.output, jobs) if args.run else 0


if __name__ == "__main__":
    raise SystemExit(main())
