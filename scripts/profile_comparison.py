#!/usr/bin/env python3
"""Collect a bounded Torch, Systems, and Compute evidence bundle."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from profile_runner import run_commands

ROOT = Path(__file__).resolve().parents[1]
CASES = "rmsnorm_256x4096,hidden_m1,hidden_m16,rope_16t_hnd_partial"
NCU_CASES = [
    ("rmsnorm_4096x4096", "cuda", ".*fused_residual_rmsnorm.*"),
    ("rmsnorm_4096x4096", "triton", ".*fused_residual_rmsnorm.*"),
    ("hidden_m1", "cuda", ".*decode_gemv.*"),
    ("hidden_m16", "cublas", ".*gemm.*"),
    ("rope_16t_hnd_partial", "cuda", ".*append_kv.*"),
]


def commands(
    python: str, experiment: str, output: Path, optimized: bool = False
) -> list[tuple[str, list[str]]]:
    base = [
        python,
        "benchmarks/compare.py",
        "--experiment",
        experiment,
        "--warmup",
        "3",
        "--repeat",
        "3",
        "--graph-calls",
        "0",
        "--no-calibrate",
    ]
    ncu_cases = NCU_CASES
    if optimized:
        base += [
            "--backends",
            "pytorch,torch_compile,cuda,triton,cute,cublas,cublaslt,cuda_fast,triton_fast,cute_fast",
        ]
        ncu_cases = (
            [
                ("rmsnorm_4096x4096", backend, ".*rmsnorm.*")
                for backend in ("cuda_fast", "triton_fast", "cute_fast")
            ]
            + [
                ("hidden_m16", backend, ".*rows.*")
                for backend in ("cuda_fast", "triton_fast", "cute_fast")
            ]
            + [
                ("rope_16t_hnd_partial", backend, ".*(rope_append|fused_kernel).*")
                for backend in ("cuda_fast", "triton_fast", "cute_fast")
            ]
        )
    selected = ["--cases", CASES]
    jobs = [
        (
            "torch",
            base
            + selected
            + ["--output", str(output / "torch"), "--profile-iterations", "3"],
        ),
        (
            "nsys",
            [
                "nsys",
                "profile",
                "--trace=cuda,nvtx,osrt,cublas",
                "--sample=none",
                "--cpuctxsw=none",
                "--output",
                str(output / "comparison"),
            ]
            + base
            + selected
            + ["--output", str(output / "nsys-results")],
        ),
        (
            "nsys-stats",
            [
                "nsys",
                "stats",
                "--report",
                "cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum",
                "--format",
                "csv",
                "--output",
                str(output / "comparison-stats"),
                str(output / "comparison.nsys-rep"),
            ],
        ),
    ]
    sections = [
        "SpeedOfLight",
        "MemoryWorkloadAnalysis",
        "LaunchStats",
        "Occupancy",
        "SchedulerStats",
        "WarpStateStats",
        "InstructionStats",
    ]
    for case, backend, kernel in ncu_cases:
        name = f"{case}--{backend}"
        report = output / f"{name}.ncu-rep"
        command = [
            "ncu",
            "--target-processes",
            "all",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            f"regex:{kernel}",
            "--launch-skip",
            "4",
            "--launch-count",
            "1",
        ]
        for section in sections:
            command.extend(["--section", section])
        command.extend(["--export", str(report)])
        command.extend(
            base
            + [
                "--cases",
                case,
                "--backends",
                backend,
                "--output",
                str(output / f"{name}-results"),
            ]
        )
        jobs.append((name, command))
        jobs.append(
            (f"{name}-csv", ["ncu", "--import", str(report), "--page", "raw", "--csv"])
        )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Collect reports; otherwise print commands only",
    )
    args = parser.parse_args()
    jobs = commands(args.python, args.experiment, args.output, args.optimized)
    if not args.run:
        for _, command in jobs:
            print(shlex.join(command))
        return
    try:
        raise SystemExit(run_commands(args.experiment, args.output, jobs))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
