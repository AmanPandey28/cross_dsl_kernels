#!/usr/bin/env python3
"""Compare fixed FP32 workloads using shared inputs, oracles, and timing."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common import capture, gpu_state, measure, measure_graph, provenance

BACKENDS = ("pytorch", "torch_compile", "cuda", "triton", "cute", "cublas", "cublaslt")
VARIANTS = ("cuda_fast", "triton_fast", "cute_fast")


def verify_replay(
    torch: Any, graph: Any, output: Any, case: dict[str, Any], op: str
) -> dict[str, Any]:
    from comparison_cases import check_output

    outputs = output if isinstance(output, (tuple, list)) else (output,)
    for tensor in outputs:
        tensor.fill_(12345.0)
    if op == "rope_kv":
        for cache in case["args"][-2:]:
            cache.fill_(-999.0)
    graph.replay()
    torch.cuda.synchronize()
    return check_output(op, output, case)


def select_cases(
    catalog: Path, op: str, names: str, smoke: bool
) -> list[dict[str, Any]]:
    cases = json.loads(catalog.read_text())["cases"]
    requested = set(names.split(",")) if names else set()
    available = {case["name"] for case in cases}
    if requested - available:
        raise ValueError(f"unknown cases: {sorted(requested - available)}")
    selected = [
        case
        for case in cases
        if (op == "all" or case["op"] == op)
        and (not requested or case["name"] in requested)
        and (not smoke or case.get("smoke", False))
    ]
    if not selected:
        raise ValueError("no cases match the requested filters")
    return selected


def write_results(directory: Path, result: dict[str, Any]) -> None:
    (directory / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    fields = [
        "op",
        "case",
        "backend",
        "status",
        "setup_and_first_call_ms",
        "event_ms",
        "wall_ms",
        "graph_event_ms",
        "graph_wall_ms",
        "logical_gbps",
        "effective_tflops",
    ]
    table = [
        "# Comparison Results",
        "",
        f"Experiment: `{result['experiment_id']}`",
        "",
        "Times are medians in milliseconds. JSON retains every sample, tolerance, and error.",
        "",
        "| Operator | Case | Backend | Status | Event | Wall | Graph/call |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    with (directory / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            flat = {key: row.get(key, "") for key in fields}
            for key, timing, surface in (
                ("event_ms", "timing", "cuda_event"),
                ("wall_ms", "timing", "synchronized_wall"),
                ("graph_event_ms", "graph_timing", "cuda_event"),
                ("graph_wall_ms", "graph_timing", "synchronized_wall"),
            ):
                flat[key] = row.get(timing, {}).get(surface, {}).get("median_ms", "")
            writer.writerow(flat)
            times = [
                f"{flat[key]:.6f}" if isinstance(flat[key], float) else "-"
                for key in ("event_ms", "wall_ms", "graph_event_ms")
            ]
            table.append(
                f"| {row['op']} | {row['case']} | {row['backend']} | {row['status']} | {' | '.join(times)} |"
            )
    (directory / "results.md").write_text("\n".join(table) + "\n")


def calibrate(torch: Any, warmup: int, repeat: int, graph_calls: int) -> dict[str, Any]:
    # 512 MiB of copy traffic per call prevents a tiny cache-resident ceiling.
    elements = 64 * 1024 * 1024
    x = torch.ones(elements, device="cuda")
    y = torch.empty_like(x)
    run = lambda: y.copy_(x)
    timings = measure(torch, run, warmup, repeat)
    logical_bytes = elements * 8
    result = {
        "op": "pytorch_copy",
        "elements": elements,
        "dtype": "float32",
        "logical_bytes": logical_bytes,
        "timing": timings,
        "cuda_event_gbps": logical_bytes / timings["cuda_event"]["median_ms"] / 1e6,
        "correct": bool(torch.equal(x, y)),
    }
    if graph_calls:
        graph, _ = capture(torch, run, graph_calls)
        graph_timing = measure_graph(torch, graph, repeat, graph_calls)
        result["graph_timing"] = graph_timing
        result["graph_cuda_event_gbps"] = (
            logical_bytes / graph_timing["cuda_event"]["median_ms"] / 1e6
        )
    return result


def profile_torch(
    torch: Any, run: Any, path: Path, label: str, iterations: int
) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(iterations):
            with record_function(label):
                run()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path.with_suffix(".trace.json")))
    events = [
        event
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    memory = sum(max(0, event.self_cpu_memory_usage) for event in prof.events())
    table = prof.key_averages().table(sort_by="self_device_time_total", row_limit=30)
    path.with_suffix(".txt").write_text(table + "\n")
    return {
        "iterations": iterations,
        "cuda_activity_count": len(events),
        "cuda_activities_per_call": len(events) / iterations,
        "positive_self_cpu_memory_bytes": memory,
        "trace": str(path.with_suffix(".trace.json").resolve()),
    }


def run_suite(
    args: argparse.Namespace, specs: list[dict[str, Any]], backends: list[str]
) -> int:
    import torch
    from comparison_cases import check_output, make_case, prepare_backend

    result = provenance(torch, args.experiment)
    result.update(
        rows=[],
        status="RUNNING",
        catalog=str(args.catalog),
        seed=args.seed,
        warmup=args.warmup,
        repeat=args.repeat,
        graph_calls=args.graph_calls,
        dtype="float32",
        timing_policy="CUDA events and synchronized wall time; batched CUDA Graph replay",
        profiled_timings_are_diagnostic=bool(args.profile_iterations),
    )
    if not torch.cuda.is_available():
        result.update(
            status="SKIP", detail="No CUDA device is visible to this process."
        )
        write_results(args.output, result)
        print(result["detail"], file=sys.stderr)
        return 2
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError(
            "This catalog is scoped to SM120; use a separate experiment for another architecture."
        )
    result.update(
        gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    result["tf32"] = False
    if not args.no_calibrate:
        result["bandwidth_calibration"] = calibrate(
            torch, args.warmup, args.repeat, args.graph_calls
        )
    write_results(args.output, result)

    for index, spec in enumerate(specs):
        case = make_case(spec, args.seed + index)
        supported = [
            backend
            for backend in backends
            if spec["op"] == "gemv" or backend not in {"cublas", "cublaslt"}
        ]
        random.Random(args.seed + index).shuffle(supported)
        for backend in supported:
            row = {
                "op": spec["op"],
                "case": spec["name"],
                "backend": backend,
                "shape": spec,
                "seed": args.seed + index,
                "status": "ERROR",
                "logical_bytes": case["logical_bytes"],
                "flops": case["flops"],
            }
            label = f"crossdsl::{spec['name']}::{backend}"
            try:
                if spec["op"] == "rope_kv":
                    for cache in case["args"][-2:]:
                        cache.fill_(-999.0)
                torch.cuda.synchronize()
                begin = time.perf_counter()
                run = prepare_backend(spec["op"], backend, case)
                output = run()
                torch.cuda.synchronize()
                row["setup_and_first_call_ms"] = (time.perf_counter() - begin) * 1000
                row["correctness"] = check_output(spec["op"], output, case)
                row["status"] = row["correctness"]["status"]
                if row["status"] == "PASS":
                    torch.cuda.nvtx.range_push(label)
                    try:
                        row["timing"] = measure(torch, run, args.warmup, args.repeat)
                        if args.graph_calls:
                            graph, graph_output = capture(torch, run, args.graph_calls)
                            row["graph_correctness"] = verify_replay(
                                torch, graph, graph_output, case, spec["op"]
                            )
                            row["status"] = row["graph_correctness"]["status"]
                            row["graph_timing"] = measure_graph(
                                torch, graph, args.repeat, args.graph_calls
                            )
                            del graph, graph_output
                    finally:
                        torch.cuda.nvtx.range_pop()
                    event_ms = row["timing"]["cuda_event"]["median_ms"]
                    row["logical_gbps"] = case["logical_bytes"] / event_ms / 1e6
                    row["effective_tflops"] = case["flops"] / event_ms / 1e9
                    if "graph_timing" in row:
                        graph_ms = row["graph_timing"]["cuda_event"]["median_ms"]
                        row["graph_logical_gbps"] = (
                            case["logical_bytes"] / graph_ms / 1e6
                        )
                        row["graph_effective_tflops"] = case["flops"] / graph_ms / 1e9
                        ceiling = result.get("bandwidth_calibration", {}).get(
                            "graph_cuda_event_gbps"
                        )
                        if ceiling and spec["op"] == "rmsnorm":
                            row["percent_copy_ceiling"] = (
                                100 * row["graph_logical_gbps"] / ceiling
                            )
                    if args.profile_iterations:
                        row["torch_profile"] = profile_torch(
                            torch,
                            run,
                            args.output / f"{spec['name']}--{backend}",
                            label,
                            args.profile_iterations,
                        )
                del run, output
            except Exception as error:  # noqa: BLE001 - save failures from every backend
                row.update(status="ERROR", error=f"{type(error).__name__}: {error}")
                row["traceback"] = traceback.format_exc()
            result["rows"].append(row)
            write_results(args.output, result)
            print(f"{spec['name']} {backend}: {row['status']}", flush=True)
        del case
        torch.cuda.synchronize()
    result["gpu_after"] = gpu_state()
    result["status"] = (
        "PASS" if all(row["status"] == "PASS" for row in result["rows"]) else "FAIL"
    )
    write_results(args.output, result)
    return 0 if result["status"] == "PASS" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--catalog", type=Path, default=ROOT / "benchmarks/shapes/comparison.json"
    )
    parser.add_argument(
        "--op", choices=("all", "rmsnorm", "gemv", "rope_kv"), default="all"
    )
    parser.add_argument("--cases", default="")
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--graph-calls", type=int, default=20)
    parser.add_argument("--profile-iterations", type=int, default=0)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"EXP-\d{8}-\d{3}", args.experiment):
        parser.error("--experiment must be an immutable EXP-YYYYMMDD-NNN ID")
    if (
        args.warmup < 1
        or args.repeat < 2
        or args.graph_calls < 0
        or args.profile_iterations < 0
    ):
        parser.error(
            "use warmup >= 1, repeat >= 2, graph-calls >= 0, profile-iterations >= 0"
        )
    backends = args.backends.split(",")
    if (
        not backends
        or set(backends) - set(BACKENDS + VARIANTS)
        or len(backends) != len(set(backends))
    ):
        parser.error(f"choose unique backends from {BACKENDS + VARIANTS}")
    try:
        specs = select_cases(args.catalog, args.op, args.cases, args.smoke)
    except ValueError as error:
        parser.error(str(error))
    if args.list:
        print(json.dumps(specs, indent=2))
        return
    if args.output.exists():
        parser.error(
            "output directory already exists; choose a fresh path to preserve evidence"
        )
    args.output.mkdir(parents=True)
    with Path("/tmp/crossdsl-gpu.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(
                "GPU lock is held by another experiment; retry when it finishes"
            )
        raise SystemExit(run_suite(args, specs, backends))


if __name__ == "__main__":
    main()
