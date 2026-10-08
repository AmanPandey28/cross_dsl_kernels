#!/usr/bin/env python3
"""Adversarial packed-code and fixed-buffer graph checks, suitable for memcheck."""

import argparse
import fcntl
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks"), str(ROOT)]

from common import capture, provenance
from profile_runner import validate_run
from w4a16_benchmark import check_output, prepare_backend


def verify(experiment, output):
    import torch
    from crossdsl_kernels.quantization import pack_int4, w4a16_reference

    torch.set_num_threads(4)
    result = provenance(torch, experiment)
    result.update(status="RUNNING", rows=[])
    if not torch.cuda.is_available():
        result.update(status="SKIP", reason="CUDA-hidden; no GPU evidence")
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        return 2
    for m, k, n, group in (
        (3, 33, 7, 32),
        (7, 129, 9, 64),
        (2, 257, 5, 256),
        (16, 4096, 17, 128),
    ):
        # All sixteen codes, exact binary scales and cancellation-heavy inputs.
        q = ((torch.arange(n * k).reshape(n, k) % 16) - 8).to(torch.int8)
        x = ((torch.arange(m * k).reshape(m, k) % 7) - 3).half().cuda()
        packed = pack_int4(q).cuda()
        scales = torch.full(
            (n, (k + group - 1) // group), 0.125, dtype=torch.float16, device="cuda"
        )
        bias = torch.linspace(-0.25, 0.25, n, dtype=torch.float16, device="cuda")
        inputs = (x, packed, scales, bias)
        originals = tuple(t.clone() for t in inputs)

        def reference():
            return w4a16_reference(*(t.cpu() for t in inputs), group_size=group).cuda()

        for backend in (
            "cuda",
            "cuda_rows",
            "triton",
            "triton_rows",
            "cute",
            "cute_rows",
        ):
            row = {"shape": [m, k, n], "group_size": group, "backend": backend}
            try:
                for target, original in zip(inputs, originals):
                    target.copy_(original)
                case = {
                    "x": x,
                    "packed": packed,
                    "scales": scales,
                    "bias": bias,
                    "spec": {"group_size": group},
                }
                run, y = prepare_backend(case, backend)
                expected = reference()
                run()
                row["direct"] = check_output(y, expected)
                stream = torch.cuda.Stream()
                y.fill_(float("nan"))
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                torch.cuda.current_stream().wait_stream(stream)
                row["side_stream"] = check_output(y, expected)
                row["inputs_unchanged"] = all(
                    torch.equal(a, b) for a, b in zip(inputs, originals)
                )
                graph, graph_y = capture(torch, run, 3)
                # Change contents, not addresses. A cached output cannot pass this.
                x.neg_()
                packed.bitwise_xor_(255)
                scales.mul_(0.5)
                bias.add_(0.25)
                expected = reference()
                graph_y.fill_(float("nan"))
                graph.replay()
                row["changed_input_graph"] = check_output(graph_y, expected)
                row["status"] = (
                    "PASS"
                    if row["inputs_unchanged"]
                    and all(
                        row[key]["status"] == "PASS"
                        for key in ("direct", "side_stream", "changed_input_graph")
                    )
                    else "FAIL"
                )
                del graph, graph_y, run, y
            except Exception:
                row.update(status="ERROR", traceback=traceback.format_exc())
            result["rows"].append(row)
            (output / "results.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
            print(f"{m}x{k}x{n} {backend}: {row['status']}", flush=True)
    result["status"] = (
        "PASS" if all(row["status"] == "PASS" for row in result["rows"]) else "FAIL"
    )
    (output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return 0 if result["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate_run(args.experiment, args.output)
    except ValueError as error:
        parser.error(str(error))
    with Path("/tmp/crossdsl-gpu.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another experiment holds the GPU lock")
        args.output.mkdir(parents=True)
        return verify(args.experiment, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
