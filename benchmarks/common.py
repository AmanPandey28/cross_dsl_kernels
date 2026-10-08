from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def command_output(*command: str) -> str:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=False
        )
        return (result.stdout or result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def gpu_state() -> str:
    return command_output(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,power.limit,"
        "power.draw,temperature.gpu,pstate,clocks.current.graphics,clocks.current.memory",
        "--format=csv",
    )


def provenance(torch: Any, experiment: str) -> dict[str, Any]:
    packages = {}
    for name in (
        "torch",
        "triton",
        "nvidia-cutlass-dsl",
        "cuda-python",
        "cuda-bindings",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = {}
    for directory in ("src", "csrc", "benchmarks"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.suffix in {".py", ".cpp", ".cu", ".json"}:
                sources[str(path.relative_to(ROOT))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return {
        "experiment_id": experiment,
        "timestamp": dt.datetime.now(
            ZoneInfo("America/Indiana/Indianapolis")
        ).isoformat(timespec="seconds"),
        "command": shlex.join([sys.executable, *sys.argv]),
        "git_commit": command_output("git", "rev-parse", "HEAD"),
        "git_status": command_output("git", "status", "--short"),
        "source_sha256": sources,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "cuda_toolkit": command_output("nvcc", "--version"),
        "nsys": command_output("nsys", "--version"),
        "ncu": command_output("ncu", "--version"),
        "compiler": command_output("c++", "--version"),
        "build_flags": {"cxx": ["-O3"], "cuda": ["-O3"], "arch": "12.0"},
        "cuda_available": torch.cuda.is_available(),
        "gpu_before": gpu_state(),
        "power_policy": "Observed only; clocks and power limits are not controlled.",
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CONDA_DEFAULT_ENV",
                "CUDA_VISIBLE_DEVICES",
                "CUDA_HOME",
                "MAX_JOBS",
            )
        },
    }


def summarize(samples_ms: list[float]) -> dict[str, Any]:
    ordered = sorted(samples_ms)
    count = len(ordered)
    return {
        "median_ms": statistics.median(ordered),
        "p10_ms": ordered[round(0.1 * (count - 1))],
        "p90_ms": ordered[round(0.9 * (count - 1))],
        "min_ms": ordered[0],
        "mean_ms": statistics.mean(ordered),
        "samples_ms": samples_ms,
    }


def measure(
    torch: Any, run: Callable[[], Any], warmup: int, repeat: int
) -> dict[str, Any]:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    event_samples, wall_samples = [], []
    for _ in range(repeat):
        begin = time.perf_counter()
        start.record()
        run()
        end.record()
        end.synchronize()
        wall_samples.append((time.perf_counter() - begin) * 1000)
        event_samples.append(float(start.elapsed_time(end)))
    return {
        "cuda_event": summarize(event_samples),
        "synchronized_wall": summarize(wall_samples),
    }


def capture(torch: Any, run: Callable[[], Any], calls: int) -> tuple[Any, Any]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            output = run()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def measure_graph(torch: Any, graph: Any, repeat: int, calls: int) -> dict[str, Any]:
    measured = measure(torch, graph.replay, warmup=3, repeat=repeat)
    for summary in measured.values():
        for key, value in summary.items():
            summary[key] = (
                [x / calls for x in value] if isinstance(value, list) else value / calls
            )
    return measured
