#!/usr/bin/env python3
"""BF16 rounding, tail, stream, graph replay and FP32 regression checks."""

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


def main():
    import torch
    from common import capture, gpu_state, provenance
    from comparison_cases import check_output, make_case, prepare_backend
    from rmsnorm_bf16 import check_rms_output, make_rms_case, rms_backend

    from crossdsl_kernels.references.rmsnorm import fused_residual_rmsnorm_reference

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sanitizer", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = provenance(torch, args.experiment)
    result["checks"] = []
    if not torch.cuda.is_available():
        result.update(status="SKIP", reason="CUDA unavailable in this process")
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return 2

    def save(name, run):
        try:
            details = run()
        except Exception:  # noqa: BLE001 - retain independent GPU regression failures
            details = {"status": "FAIL", "traceback": traceback.format_exc()}
        result["checks"].append({"name": name, **details})
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"{name}: {details['status']}", flush=True)

    sizes = (1, 31, 255, 256, 257, 512, 513, 1024, 2048, 4096, 8192, 8193, 12289, 65536)
    specs = [(3, hidden, "random", int(hidden % 2 == 1)) for hidden in sizes]
    specs += [
        (2, 4096, pattern, 0)
        for pattern in ("zero", "cancel", "small", "large", "rounding")
    ]
    if args.sanitizer:
        specs = [specs[i] for i in (0, 6, 9, 11, 14, 18)]
    with Path("/tmp/crossdsl-gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream = torch.cuda.Stream()
        for index, (rows, hidden, pattern, offset) in enumerate(specs):
            case = make_rms_case(
                rows, hidden, 9000 + index, pattern=pattern, offset=offset
            )
            for backend in ("cuda", "cuda_fast", "triton"):

                def check(case=case, backend=backend):
                    originals = tuple(t.clone() for t in case["args"][:3])
                    run = rms_backend(backend, case)
                    # Compile once outside the side-stream correctness test.
                    run()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        output = run()
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    details = check_rms_output(output, case)
                    details["inputs_unchanged"] = all(
                        torch.equal(t, old) for t, old in zip(case["args"], originals)
                    )
                    r = case["args"][0].float() + case["args"][1].float()
                    ideal = r.double() * torch.rsqrt(
                        r.double().square().mean(-1, keepdim=True) + 1e-5
                    )
                    ideal *= case["args"][2].double()
                    wider_case = {**case, "expected": (ideal.bfloat16(), r.bfloat16())}
                    details["fp64_stats_oracle"] = check_rms_output(output, wider_case)
                    if (
                        not details["inputs_unchanged"]
                        or details["fp64_stats_oracle"]["status"] != "PASS"
                    ):
                        details["status"] = "FAIL"
                    return details

                save(
                    f"stream::{backend}::{rows}x{hidden}::{pattern}::offset{offset}",
                    check,
                )
        for rows, hidden in ((1, 4096), (3, 513)):
            for backend in ("cuda", "cuda_fast", "triton"):

                def graph_check(rows=rows, hidden=hidden, backend=backend):
                    case = make_rms_case(rows, hidden, 10101)
                    graph, output = capture(torch, rms_backend(backend, case), 2)
                    for tensor in output:
                        tensor.fill_(12345)
                    graph.replay()
                    torch.cuda.synchronize()
                    poison = check_rms_output(output, case)
                    case["args"][0].mul_(0.5)
                    case["expected"] = fused_residual_rmsnorm_reference(*case["args"])
                    for tensor in output:
                        tensor.fill_(12345)
                    graph.replay()
                    torch.cuda.synchronize()
                    changed = check_rms_output(output, case)
                    return {
                        "status": "PASS"
                        if poison["status"] == changed["status"] == "PASS"
                        else "FAIL",
                        "poisoned_replay": poison,
                        "changed_input_replay": changed,
                    }

                save(f"graph::{backend}::{rows}x{hidden}", graph_check)
        for backend in ("cuda", "cuda_fast", "triton"):
            for label in ("mixed_dtype", "fp16", "strided", "zero_rows", "bad_eps"):

                def reject(backend=backend, label=label):
                    case = make_rms_case(2, 513, 77)
                    x, residual, weight, eps = case["args"]
                    if label == "mixed_dtype":
                        weight = weight.float()
                    elif label == "fp16":
                        x, residual, weight = (t.half() for t in (x, residual, weight))
                    elif label == "strided":
                        x = torch.zeros(2, 1026, device="cuda", dtype=torch.bfloat16)[
                            :, ::2
                        ]
                    elif label == "zero_rows":
                        x, residual = x[:0], residual[:0]
                    else:
                        eps = float("nan")
                    case["args"] = (x, residual, weight, eps)
                    try:
                        rms_backend(backend, case)()
                    except (ValueError, RuntimeError) as error:
                        return {"status": "PASS", "rejected": str(error)}
                    return {"status": "FAIL", "reason": "invalid input accepted"}

                save(f"reject::{backend}::{label}", reject)
        for hidden in (513, 4096):
            for backend in ("cuda", "cuda_fast", "triton", "triton_fast"):

                def fp32_check(hidden=hidden, backend=backend):
                    case = make_case(
                        {"op": "rmsnorm", "rows": 3, "hidden": hidden}, 123
                    )
                    return check_output(
                        "rmsnorm", prepare_backend("rmsnorm", backend, case)(), case
                    )

                save(f"fp32_regression::{backend}::{hidden}", fp32_check)
    result["gpu_after"] = gpu_state()
    result["status"] = (
        "PASS" if all(c["status"] == "PASS" for c in result["checks"]) else "FAIL"
    )
    result["limits"] = (
        "Forward-only, one SM120 GPU; no gradients, mixed weight dtype or CuTe BF16 claim"
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
