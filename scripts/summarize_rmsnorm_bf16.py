#!/usr/bin/env python3
"""Recompute paired BF16/vendor comparisons and unit-aware profile summaries."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from summarize_comparison import METRICS, metric_value


def paired_interval(baseline, candidate):
    ratios = [a / b for a, b in zip(baseline, candidate, strict=True)]
    # Adjacent counterbalanced rounds form a block; do not pretend order pairs are independent.
    blocks = [ratios[i : i + 2] for i in range(0, len(ratios), 2)]
    rng = random.Random(19)
    bootstrap = sorted(
        statistics.median([value for _ in blocks for value in rng.choice(blocks)])
        for _ in range(2000)
    )
    return {
        "paired_median_speedup": statistics.median(ratios),
        "paired_speedup_samples": ratios,
        "block_bootstrap_95_interval": [bootstrap[49], bootstrap[1949]],
        "interval_limit": "Conditional on this process; excludes systematic clock, thermal and harness error",
    }


def analyze(path):
    data = json.loads(path.read_text())
    grouped = {}
    for row in data["rows"]:
        grouped.setdefault(row["case"], {})[row["backend"]] = row
        for summary in row["timing"].values():
            assert statistics.median(summary["samples_ms"]) == summary["median_ms"]
    comparisons = []
    for case, rows in grouped.items():
        for candidate in ("cuda_fast", "triton_fast"):
            if candidate not in rows:
                continue
            for baseline in ("pytorch", "torch_compile", "cublas", "cublaslt"):
                if baseline not in rows:
                    continue
                for mode in ("event", "graph_event"):
                    a, b = rows[baseline], rows[candidate]
                    if mode not in a["timing"]:
                        continue
                    valid = a["status"] == b["status"] == "PASS"
                    comparisons.append(
                        {
                            "case": case,
                            "candidate": candidate,
                            "baseline": baseline,
                            "mode": mode,
                            "valid": valid,
                            "median_speedup": a["timing"][mode]["median_ms"]
                            / b["timing"][mode]["median_ms"],
                            **paired_interval(
                                a["timing"][mode]["samples_ms"],
                                b["timing"][mode]["samples_ms"],
                            ),
                        }
                    )
    return {
        "experiment_id": data["experiment_id"],
        "source": str(path),
        "status": data["status"],
        "rows": data["rows"],
        "comparisons": comparisons,
        "failed_cases": [r for r in data["rows"] if r["status"] != "PASS"],
        "method": {
            key: data[key]
            for key in (
                "seed",
                "rounds",
                "graph_calls",
                "warmup",
                "timing_policy",
                "allocation_policy",
            )
        },
    }


def profiles(directory):
    ncu = []
    for path in sorted(directory.glob("*-csv.log")):
        with path.open() as handle:
            reader = csv.DictReader(handle)
            units = next(reader)
            for row in reader:
                ncu.append(
                    {
                        "file": str(path),
                        "kernel": row["Kernel Name"],
                        "metrics": {
                            name: metric_value(name, row[key], units[key])
                            if key in row
                            else None
                            for name, key in METRICS.items()
                        },
                    }
                )
    launches = []
    data = json.loads((directory / "torch/results.json").read_text())
    for row in data["rows"]:
        trace = directory / "torch" / Path(row["torch_profile"]["trace"]).name
        events = json.loads(trace.read_text())["traceEvents"]
        counts = Counter(e["name"] for e in events if e.get("cat") == "kernel")
        launches.append(
            {
                "case": row["case"],
                "backend": row["backend"],
                "trace": str(trace),
                "kernel_names": dict(counts),
                "kernels_per_call": sum(counts.values())
                / row["torch_profile"]["iterations"],
            }
        )
    return {
        "ncu": ncu,
        "torch_launches": launches,
        "note": "NCU replay/cache policy and clocks differ from warm unprofiled loops; SOL uses a theoretical denominator, not copy bandwidth",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve old analysis; choose a fresh directory")
    args.output.mkdir(parents=True)
    result = {"experiments": [analyze(path) for path in args.results]}
    if args.profiles:
        result["profiles"] = profiles(args.profiles)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Paired Results",
        "",
        "Latencies are medians in microseconds. Raw JSON retains all samples, paired orders and telemetry.",
        "Ordinary events include GPU idle time while Python submits work; graph replay amortizes that host overhead.",
        "Clocks are unlocked. These are operator measurements, not inference latency or whole-model quality.",
        "",
        "| Experiment | Case | Backend | Gate | Event us | Graph us |",
        "|---|---|---|---|---:|---:|",
    ]
    for experiment in result["experiments"]:
        for row in experiment["rows"]:
            times = [
                f"{row['timing'][m]['median_ms'] * 1000:.3f}"
                if m in row["timing"]
                else "-"
                for m in ("event", "graph_event")
            ]
            lines.append(
                f"| {experiment['experiment_id']} | {row['case']} | {row['backend']} | {row['status']} | {' | '.join(times)} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "Compare only PASS rows with matched semantics. Failed vendor rows are diagnostic, not valid speedup baselines.",
        "Each JSON comparison has a paired-round speedup and block-bootstrap interval; those intervals do not cover systematic laptop drift.",
        "A fused custom kernel need not beat compiled PyTorch; inspect each shape, baseline and timing mode separately.",
    ]
    if args.profiles:
        lines += [
            "",
            "## Profiles",
            "",
            "Torch trace kernel categories, rather than all CUDA activities, count launches.",
        ]
        for row in result["profiles"]["torch_launches"]:
            lines.append(
                f"- {row['case']} {row['backend']}: {row['kernels_per_call']:g} kernels/call."
            )
        for row in result["profiles"]["ncu"]:
            values = row["metrics"]
            lines.append(
                f"- {row['kernel']}: {values['dram_gbps']['value']:.2f} {values['dram_gbps']['unit']} DRAM; "
                f"{values['dram_sol_pct']['value']:.2f}% sustained theoretical peak; "
                f"{values['registers']['value']:g} registers/thread; {values['occupancy_pct']['value']:.2f}% achieved occupancy."
            )
        lines.append(
            "Missing raw metrics remain null; counter times are diagnostic and units are explicitly normalized."
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Saved recomputed evidence in {args.output}")


if __name__ == "__main__":
    main()
