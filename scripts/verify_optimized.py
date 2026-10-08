#!/usr/bin/env python3
"""Bounded public-API, tail, stream and invalid-input regression checks."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))


def public_run(op, backend, case):
    from comparison_cases import prepare_backend

    if backend == "cuda_fast":
        return prepare_backend(op, backend, case)()
    if backend == "triton_fast":
        return prepare_backend(op, backend, case)()
    if op == "rmsnorm":
        from crossdsl_kernels.cute.rmsnorm import fused_residual_rmsnorm_cute

        return fused_residual_rmsnorm_cute(*case["args"], fast=True)
    if op == "gemv":
        from crossdsl_kernels.cute.gemv_rows import decode_gemv_cute_rows as kernel
    else:
        from crossdsl_kernels.cute.rope_kv_fused import (
            rope_gqa_paged_kv_append_cute_fused as kernel,
        )
    return kernel(*case["args"], **case["kwargs"])


def main():
    import torch
    from common import provenance
    from comparison_cases import check_output, make_case

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output path")
    result = provenance(torch, args.experiment)
    result["checks"] = []
    catalog = json.loads((ROOT / "benchmarks/shapes/comparison.json").read_text())[
        "cases"
    ]
    specs = [
        next(s for s in catalog if s["name"] == name)
        for name in ("rmsnorm_3x513_tail", "rope_7t_tail", "rope_16t_hnd_partial")
    ]
    specs += [
        {
            "name": f"gemv_m5_{layout}",
            "op": "gemv",
            "m": 5,
            "k": 257,
            "n": 67,
            "layout": layout,
            "bias": True,
        }
        for layout in ("KN", "NK")
    ]
    stream = torch.cuda.Stream()

    def save(name, run):
        try:
            details = run()
            status = details.get("status", "PASS")
        except Exception:  # noqa: BLE001 - preserve each independent regression failure
            status, details = "FAIL", {"traceback": traceback.format_exc()}
        result["checks"].append({"name": name, "status": status, "details": details})
        print(f"{name}: {status}", flush=True)

    with Path("/tmp/crossdsl-gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, spec in enumerate(specs):
            for backend in ("cuda_fast", "triton_fast", "cute_fast"):

                def check(spec=spec, backend=backend, index=index):
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        case = make_case(spec, 500 + index)
                        output = public_run(spec["op"], backend, case)
                    torch.cuda.current_stream().wait_stream(stream)
                    return check_output(spec["op"], output, case)

                save(f"stream::{spec['name']}::{backend}", check)

        rms = make_case(specs[0], 600)
        rope = make_case(specs[1], 601)

        def empty_bias():
            from comparison_cases import cuda_module

            case = make_case(
                {
                    "op": "gemv",
                    "m": 5,
                    "k": 257,
                    "n": 67,
                    "layout": "KN",
                    "bias": False,
                },
                602,
            )
            output = cuda_module("gemv").decode_gemv_rows(
                *case["args"][:2], torch.empty(0, dtype=torch.int64), "KN"
            )
            return check_output("gemv", output, case)

        save("empty_bias::cuda_fast", empty_bias)
        for backend in ("cuda_fast", "triton_fast", "cute_fast"):
            bad_cases = []
            for eps in (-1.0, float("nan"), float("inf")):
                bad_cases.append(
                    ("rmsnorm", f"eps={eps}", {**rms, "args": (*rms["args"][:3], eps)})
                )
            bad_cases.append(
                (
                    "rope_kv",
                    "cache_alias",
                    {**rope, "args": (*rope["args"][:-1], rope["args"][-2])},
                )
            )
            bad_cases.append(
                (
                    "rope_kv",
                    "input_cache_alias",
                    {
                        **rope,
                        "args": (
                            rope["args"][-2]
                            .flatten()[: rope["args"][0].numel()]
                            .view_as(rope["args"][0]),
                            *rope["args"][1:],
                        ),
                    },
                )
            )
            bad_cases.append(
                (
                    "rope_kv",
                    "gqa_ratio",
                    {
                        **rope,
                        "args": (
                            torch.zeros(7, 5, 66, device="cuda"),
                            *rope["args"][1:],
                        ),
                    },
                )
            )
            for op, label, case in bad_cases:

                def reject(op=op, backend=backend, case=case):
                    try:
                        public_run(op, backend, case)
                    except (ValueError, RuntimeError) as error:
                        return {"status": "PASS", "rejected": str(error)}
                    raise AssertionError("invalid input was accepted")

                save(f"reject::{label}::{backend}", reject)
        result["multi_gpu"] = (
            "SKIP: laptop has one GPU; device guards inspected but cross-device execution untested"
        )
        result["status"] = (
            "PASS" if all(c["status"] == "PASS" for c in result["checks"]) else "FAIL"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
