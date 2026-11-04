#!/usr/bin/env python3
"""Collect bounded BF16 Torch, Nsight Systems and Nsight Compute reports."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from profile_runner import run_commands


def commands(python, experiment, output):
    base = [
        python,
        "benchmarks/rmsnorm_bf16.py",
        "--experiment",
        experiment,
        "--rounds",
        "2",
        "--warmup",
        "5",
        "--graph-calls",
        "0",
    ]
    selected = ["--cases", "bf16_256x4096,bf16_4096x4096"]
    jobs = [
        (
            "torch",
            base
            + selected
            + ["--profile-iterations", "3", "--output", str(output / "torch")],
        ),
        (
            "nsys",
            [
                "nsys",
                "profile",
                "--trace=cuda,nvtx,osrt",
                "--sample=none",
                "--cpuctxsw=none",
                "--output",
                str(output / "bf16"),
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
                str(output / "bf16-stats"),
                str(output / "bf16.nsys-rep"),
            ],
        ),
    ]
    for backend, kernel in (
        ("cuda_fast", ".*rmsnorm_bf16_register.*"),
        ("triton_fast", ".*fused_residual_rmsnorm.*"),
    ):
        name = f"bf16_4096x4096--{backend}"
        report = output / f"{name}.ncu-rep"
        collect = [
            "ncu",
            "--target-processes",
            "all",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            f"regex:{kernel}",
            "--launch-skip",
            "6",
            "--launch-count",
            "1",
            "--clock-control",
            "none",
        ]
        for section in (
            "SpeedOfLight",
            "MemoryWorkloadAnalysis",
            "LaunchStats",
            "Occupancy",
            "SchedulerStats",
            "WarpStateStats",
        ):
            collect += ["--section", section]
        collect += (
            ["--export", str(report)]
            + base
            + [
                "--cases",
                "bf16_4096x4096",
                "--backends",
                backend,
                "--output",
                str(output / f"{name}-results"),
            ]
        )
        jobs.append((name, collect))
        jobs.append(
            (f"{name}-csv", ["ncu", "--import", str(report), "--page", "raw", "--csv"])
        )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    jobs = commands(args.python, args.experiment, args.output)
    for _, command in jobs:
        print(shlex.join(command), flush=True)
    return run_commands(args.experiment, args.output, jobs) if args.run else 0


if __name__ == "__main__":
    raise SystemExit(main())
