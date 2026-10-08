#!/usr/bin/env python3
"""Preview or collect a small W4A16 profiler bundle with the shared job runner."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from profile_runner import run_commands


def commands(python: str, experiment: str, output: Path):
    base = [
        python,
        "benchmarks/w4a16_benchmark.py",
        "--experiment",
        experiment,
        "--warmup",
        "3",
        "--repeat",
        "3",
        "--graph-calls",
        "0",
    ]
    selected = ["--cases", "hidden_m1,hidden_m16"]
    jobs = [
        (
            "torch",
            base + selected + ["--mode", "torch", "--output", str(output / "torch")],
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
                str(output / "w4a16"),
            ]
            + base
            + selected
            + ["--mode", "workload", "--output", str(output / "nsys-results")],
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
                str(output / "w4a16-stats"),
                str(output / "w4a16.nsys-rep"),
            ],
        ),
    ]
    # Two ownership variants per language. Skip direct validation and three warmups.
    for backend in ("cuda", "cuda_rows", "triton", "triton_rows", "cute", "cute_rows"):
        name = f"hidden_m16--{backend}"
        report = output / f"{name}.ncu-rep"
        command = [
            "ncu",
            "--target-processes",
            "all",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            "regex:.*w4a16.*",
            "--launch-skip",
            "4",
            "--launch-count",
            "1",
        ]
        for section in (
            "SpeedOfLight",
            "MemoryWorkloadAnalysis",
            "LaunchStats",
            "Occupancy",
            "SchedulerStats",
        ):
            command.extend(["--section", section])
        command += (
            ["--export", str(report)]
            + base
            + [
                "--mode",
                "workload",
                "--cases",
                "hidden_m16",
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
    if args.run:
        try:
            return run_commands(args.experiment, args.output, jobs)
        except ValueError as error:
            parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
