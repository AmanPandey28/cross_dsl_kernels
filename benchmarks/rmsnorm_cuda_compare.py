#!/usr/bin/env python3
"""Same-process CUDA RMSNorm before/after and recompute/retain comparison."""
from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common import capture, gpu_state, measure, provenance, summarize
from rmsnorm_bf16 import check_rms_output, counterbalanced_orders, make_rms_case
from summarize_rmsnorm_bf16 import paired_interval

SHAPES = ((256, 4096), (4096, 4096), (3, 513), (2, 12289))


def load_before(torch, directory):
    from torch.utils.cpp_extension import load

    sources = [directory / name for name in ("rmsnorm_ext.cpp", "rmsnorm_kernel.cu")]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()[:12]
    name = f"crossdsl_rmsnorm_before_{digest}"
    build = ROOT / "results/build/torch_extensions/rmsnorm_compare/sm120" / name
    build.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", "4")
    return load(name=name, sources=[str(path) for path in sources], build_directory=str(build),
                extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], verbose=False)


def fp32_checks(torch, modules):
    from comparison_cases import check_output, make_case

    checks = []
    stream = torch.cuda.Stream()
    for hidden in (1, 4, 513, 1024, 2048, 4096, 8192, 8196):
        for offset in (0, 1):
            case = make_case({"op": "rmsnorm", "rows": 3, "hidden": hidden}, 811)
            args = []
            for tensor in case["args"][:3]:
                storage = torch.empty(tensor.numel() + offset, device="cuda")
                view = storage[offset:].view(tensor.shape)
                view.copy_(tensor)
                args.append(view)
            args.append(case["args"][3])
            originals = [tensor.clone() for tensor in args[:3]]
            stream.wait_stream(torch.cuda.current_stream())
            for version, module in modules.items():
                for fast in (False, True):
                    with torch.cuda.stream(stream):
                        output = module.fused_residual_rmsnorm(*args, fast=fast)
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    details = check_output("rmsnorm", output, case)
                    unchanged = all(torch.equal(a, b) for a, b in zip(args, originals))
                    checks.append({"hidden": hidden, "offset": offset, "version": version,
                                   "fast": fast, **details, "inputs_unchanged": unchanged})
    return checks


def run(args):
    import torch
    from comparison_cases import cuda_module

    result = provenance(torch, args.experiment)
    result.update(rows=[], fp32_checks=[], rounds=args.rounds, warmup=args.warmup,
                  graph_calls=args.graph_calls, seed=args.seed, orders={}, telemetry=[],
                  timing_policy="Existing fresh-event helper; adjacent shuffled/reversed rounds; modes separate",
                  allocation_policy="Identical allocating CUDA extension APIs; graph pools separate",
                  before_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in args.before.glob("*.c*")})
    if not torch.cuda.is_available():
        result.update(status="SKIP", reason="CUDA unavailable in this process")
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        return 2
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Keep this SM120 comparison separate from other architectures")
    modules = {"before": load_before(torch, args.before), "after": cuda_module("rmsnorm")}
    result["module_paths"] = {name: module.__file__ for name, module in modules.items()}
    result["fp32_checks"] = fp32_checks(torch, modules)
    for index, (rows, hidden) in enumerate(SHAPES):
        name = f"bf16_{rows}x{hidden}"
        case = make_rms_case(rows, hidden, args.seed + index)
        originals = [tensor.clone() for tensor in case["args"][:3]]
        runs, graphs, entries, outputs = {}, {}, {}, {}
        for version, module in modules.items():
            for fast in (False, True):
                backend = f"{version}_{'register' if fast else 'recompute'}"
                execute = functools.partial(module.fused_residual_rmsnorm, *case["args"], fast=fast)
                output = execute()
                torch.cuda.synchronize()
                outputs[backend] = tuple(tensor.clone() for tensor in output)
                entry = {"case": name, "backend": backend, "fast": fast,
                         "actual_path": "register" if fast and hidden <= 8192 else "recompute",
                         "correctness": check_rms_output(output, case),
                         "samples": {key: [] for key in ("event", "wall", "graph_event", "graph_wall")}}
                for _ in range(args.warmup):
                    execute()
                graph, output = capture(torch, execute, args.graph_calls)
                for tensor in output:
                    tensor.fill_(12345)
                graph.replay()
                torch.cuda.synchronize()
                entry["graph_correctness"] = check_rms_output(output, case)
                runs[backend], graphs[backend], entries[backend] = execute, graph, entry
                result["rows"].append(entry)
        result.setdefault("bitwise_before_after", {})[name] = {
            path: all(torch.equal(a, b) for a, b in zip(outputs[f"before_{path}"], outputs[f"after_{path}"]))
            for path in ("register", "recompute")}
        orders = counterbalanced_orders(list(runs), args.rounds, args.seed + index)
        result["orders"][name] = orders
        for order in orders:
            telemetry = {"case": name, "before": gpu_state()}
            for backend in order:
                for mode, execute, divisor in (("event", runs[backend], 1),
                                                ("graph_event", graphs[backend].replay, args.graph_calls)):
                    timing = measure(torch, execute, warmup=0, repeat=1)
                    entries[backend]["samples"][mode].append(timing["cuda_event"]["median_ms"] / divisor)
                    wall = "wall" if mode == "event" else "graph_wall"
                    entries[backend]["samples"][wall].append(timing["synchronized_wall"]["median_ms"] / divisor)
            telemetry["after"] = gpu_state()
            result["telemetry"].append(telemetry)
        for entry in entries.values():
            entry["timing"] = {key: summarize(samples) for key, samples in entry.pop("samples").items()}
            entry["status"] = "PASS" if all(entry[key]["status"] == "PASS"
                                                   for key in ("correctness", "graph_correctness")) else "FAIL"
        result["inputs_unchanged"] = result.get("inputs_unchanged", True) and all(
            torch.equal(a, b) for a, b in zip(case["args"], originals))
        print(f"checked and measured {name}", flush=True)
        del graphs, runs, entries, outputs, output, graph
        torch.cuda.synchronize()
    result["comparisons"] = []
    for name in result["orders"]:
        entries = {row["backend"]: row for row in result["rows"] if row["case"] == name}
        pairs = (("before_register", "after_register"), ("before_recompute", "after_recompute"),
                 ("after_recompute", "after_register"))
        for baseline, candidate in pairs:
            for mode in ("event", "graph_event"):
                a, b = entries[baseline], entries[candidate]
                result["comparisons"].append({"case": name, "baseline": baseline, "candidate": candidate,
                    "mode": mode, "median_ratio": a["timing"][mode]["median_ms"] / b["timing"][mode]["median_ms"],
                    **paired_interval(a["timing"][mode]["samples_ms"], b["timing"][mode]["samples_ms"])})
    result["gpu_after"] = gpu_state()
    valid = all(row["status"] == "PASS" and row.get("inputs_unchanged", True)
                for row in result["rows"] + result["fp32_checks"])
    identical = all(all(paths.values()) for paths in result["bitwise_before_after"].values())
    result["status"] = "PASS" if valid and identical and result["inputs_unchanged"] else "FAIL"
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['status']}: {len(result['rows'])} BF16 rows; {len(result['fp32_checks'])} FP32 checks", flush=True)
    return 0 if result["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--graph-calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    if not re.fullmatch(r"EXP-\d{8}-\d{3}", args.experiment):
        parser.error("use a recorded EXP-YYYYMMDD-NNN ID")
    if args.rounds < 2 or args.rounds % 2 or args.warmup < 1 or args.graph_calls < 1:
        parser.error("use even rounds >= 2, warmup >= 1, graph-calls >= 1")
    if not all((args.before / name).is_file() for name in ("rmsnorm_ext.cpp", "rmsnorm_kernel.cu")):
        parser.error("--before must contain the preserved binding and CUDA source")
    if args.output.exists():
        parser.error("preserve previous results; choose a fresh output directory")
    with Path("/tmp/crossdsl-gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True)
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
