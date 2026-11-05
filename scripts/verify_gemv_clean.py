#!/usr/bin/env python3
"""Bounded correctness checks for the parallel CUDA projection extension."""
from __future__ import annotations

import argparse
import fcntl
import json
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from common import capture, provenance
from gemv_cuda_benchmark import CLEAN_ENTRY_POINTS, clean_module, prepare, projection_cases, verify_changed_replay


def tiny_cases():
    return [{"name": f"tiny_{m}x{k}x{n}_{layout}_{bias}", "op": "gemv",
             "m": m, "k": k, "n": n, "layout": layout, "bias": bias}
            for m, k, n in ((1, 1, 1), (2, 7, 31), (3, 2, 33), (1, 33, 17))
            for layout in ("KN", "NK") for bias in (False, True)]


def validate(args):
    import torch
    from comparison_cases import check_output, make_case

    result = provenance(torch, args.experiment)
    result.update(checks=[], status="RUNNING", small_only=args.small_only)

    def write():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def record(name, run):
        try:
            details = run()
        except Exception:
            details = {"status": "FAIL", "traceback": traceback.format_exc()}
        result["checks"].append({"name": name, **details})
        write()
        print(f"{name}: {details['status']}", flush=True)

    if not torch.cuda.is_available():
        result.update(status="SKIP", reason="CUDA unavailable to this process")
        write()
        return 2
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    write()
    module = clean_module()
    result["extension"] = module.__file__
    specs = projection_cases() + tiny_cases()
    if args.small_only:
        specs = [spec for spec in specs if spec["k"] <= 257 and spec["n"] <= 67]
    stream = torch.cuda.Stream()
    for index, spec in enumerate(specs):
        case = make_case(spec, 61000 + index)
        for backend in CLEAN_ENTRY_POINTS:

            def check(case=case, backend=backend):
                inputs = [t.clone() for t in case["args"] if t is not None]
                run = prepare(backend, case)
                direct = check_output("gemv", run(), case)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    output = run()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                side_stream = check_output("gemv", output, case)
                unchanged = all(torch.equal(t, old) for t, old in zip(
                    (t for t in case["args"] if t is not None), inputs))
                return {"status": "PASS" if unchanged and direct["status"] == side_stream["status"] == "PASS" else "FAIL",
                        "direct": direct, "side_stream": side_stream, "inputs_unchanged": unchanged}

            record(f"direct_stream::{spec['name']}::{backend}", check)
    for layout in ("KN", "NK"):
        for bias in (False, True):
            case = make_case({"op": "gemv", "m": 5, "k": 257, "n": 67,
                              "layout": layout, "bias": bias}, 71000)
            for backend in CLEAN_ENTRY_POINTS:

                def replay(case=case, backend=backend):
                    graph, output = capture(torch, prepare(backend, case), 2)
                    return verify_changed_replay(torch, graph, output, case)

                record(f"graph::{layout}::bias{bias}::{backend}", replay)

    x = torch.randn(3, 7, device="cuda")
    weight = torch.randn(7, 5, device="cuda")
    bias = torch.randn(5, device="cuda")
    invalid = {
        "x_dtype": (x.half(), weight, bias, "KN"),
        "weight_dtype": (x, weight.half(), bias, "KN"),
        "bias_dtype": (x, weight, bias.half(), "KN"),
        "x_rank": (x.flatten(), weight, bias, "KN"),
        "weight_rank": (x, weight.flatten(), bias, "KN"),
        "k_mismatch": (x, weight[:6].contiguous(), bias, "KN"),
        "layout": (x, weight, bias, "invalid"),
        "x_strided": (torch.randn(3, 14, device="cuda")[:, ::2], weight, bias, "KN"),
        "weight_strided": (x, torch.randn(7, 10, device="cuda")[:, ::2], bias, "KN"),
        "bias_strided": (x, weight, torch.randn(10, device="cuda")[::2], "KN"),
        "bias_shape": (x, weight, bias[:4], "KN"),
        "empty_bias": (x, weight, bias[:0], "KN"),
        "zero_m": (x[:0], weight, bias, "KN"),
        "zero_k": (x[:, :0], weight[:0], bias, "KN"),
        "zero_n": (x, weight[:, :0], None, "KN"),
        "x_cpu": (x.cpu(), weight, bias, "KN"),
        "weight_cpu": (x, weight.cpu(), bias, "KN"),
        "bias_cpu": (x, weight, bias.cpu(), "KN"),
    }
    for backend, entry in CLEAN_ENTRY_POINTS.items():
        for label, inputs in invalid.items():

            def reject(entry=entry, inputs=inputs):
                try:
                    getattr(module, entry)(*inputs)
                except (ValueError, RuntimeError) as error:
                    return {"status": "PASS", "rejected": str(error)}
                return {"status": "FAIL", "reason": "invalid inputs accepted"}

            record(f"reject::{label}::{backend}", reject)
    result["multi_gpu"] = "NOT RUN: only one local GPU; CUDA/CPU device mismatch rejected, two-CUDA-device mismatch not exercised"
    if torch.cuda.device_count() > 1:
        for entry in CLEAN_ENTRY_POINTS.values():
            def reject_device(entry=entry):
                try:
                    getattr(module, entry)(x, weight.to("cuda:1"), None, "KN")
                except RuntimeError as error:
                    return {"status": "PASS", "rejected": str(error)}
                return {"status": "FAIL"}
            record(f"reject::two_cuda_devices::{entry}", reject_device)
        result["multi_gpu"] = "Device mismatch rejection exercised; not multi-GPU execution certification"
    result["status"] = "PASS" if all(item["status"] == "PASS" for item in result["checks"]) else "FAIL"
    write()
    return 0 if result["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"EXP-\d{8}-\d{3}", args.experiment):
        parser.error("use EXP-YYYYMMDD-NNN")
    if args.output.exists():
        parser.error("choose a fresh output path")
    with Path("/tmp/crossdsl-gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        return validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
