#!/usr/bin/env python3
"""Recompute SIMT projection comparisons and counter evidence without a GPU."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from summarize_comparison import METRICS, metric_value

# Collection-time identifiers remain untouched in the archived input rows.
BACKEND_ALIASES = {"clean_warp": "simt_warp", "clean_rows": "simt_rows"}

EXTRA_METRICS = {
    "dram_bytes": "dram__bytes.sum",
    "dram_read_bytes": "dram__bytes_op_read.sum",
    "dram_write_bytes": "dram__bytes_op_write.sum",
    "shared_load_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "shared_store_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "active_warps": "sm__warps_active.avg.per_cycle_active",
}


def counter_value(name, raw, unit):
    value = metric_value(name, raw, unit)
    if name in {"dram_bytes", "dram_read_bytes", "dram_write_bytes"}:
        factors = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}
        if unit not in factors:
            raise ValueError(f"unknown DRAM byte unit: {unit}")
        return {"value": value["value"] * factors[unit], "unit": "byte",
                "raw_value": value["value"], "raw_unit": unit}
    return value


def analyze(path):
    data = json.loads(path.read_text())
    rows = {}
    for row in data["rows"]:
        if row["status"] != "PASS":
            raise ValueError(f"failed saved row: {path}: {row['case']} {row['backend']}")
        for summary in row["timing"].values():
            if statistics.median(summary["samples_ms"]) != summary["median_ms"]:
                raise ValueError("saved median does not match samples")
            if len(summary["samples_ms"]) != data["repeat"]:
                raise ValueError("sample count does not match repeats")
        backend = BACKEND_ALIASES.get(row["backend"], row["backend"])
        case_rows = rows.setdefault(row["case"], {})
        if backend in case_rows:
            raise ValueError(f"duplicate backend after alias resolution: {row['case']} {backend}")
        case_rows[backend] = row
    comparisons = []
    for case, backends in rows.items():
        for baseline, candidate in (("cuda", "simt_warp"), ("cuda_fast", "simt_rows"),
                                    ("cuda_fast", "tiled"), ("simt_rows", "tiled"),
                                    ("cublas", "simt_rows"), ("cublas", "tiled")):
            if baseline not in backends or candidate not in backends:
                continue
            for mode in ("event", "graph_event"):
                if mode not in backends[baseline]["timing"]:
                    continue
                old = backends[baseline]["timing"][mode]["median_ms"]
                new = backends[candidate]["timing"][mode]["median_ms"]
                comparisons.append({"case": case, "baseline": baseline, "candidate": candidate,
                                    "mode": mode, "baseline_over_candidate": old / new,
                                    "candidate_latency_change_pct": (new / old - 1) * 100,
                                    "refactor_regression_warning": baseline in {"cuda", "cuda_fast"}
                                    and candidate in {"simt_warp", "simt_rows"} and new > old * 1.1})
    return {"experiment_id": data["experiment_id"], "source": str(path), "status": data["status"],
            "rows": data["rows"], "comparisons": comparisons,
            "method": {key: data[key] for key in ("seed", "warmup", "repeat", "graph_calls", "timing_policy")}}


def profiles(directory):
    result = []
    for path in sorted(directory.glob("*-csv.log")):
        with path.open() as handle:
            reader = csv.DictReader(handle)
            units = next(reader)
            rows = list(reader)
        if len(rows) != 1:
            raise ValueError(f"expected one selected kernel: {path}")
        row = rows[0]
        result.append({"file": str(path), "kernel": row["Kernel Name"],
                       "metrics": {name: counter_value(name, row[key], units[key]) if key in row and row[key] else None
                                   for name, key in (METRICS | EXTRA_METRICS).items()}})
    if not result:
        raise ValueError(f"no captured kernel CSVs found: {directory}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output directory")
    summary = {"experiments": [analyze(path) for path in args.results]}
    if args.profiles:
        summary["profiles"] = profiles(args.profiles)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = ["# Projection Evidence", "", "Graph-event microseconds/call; unchanged numerical gates.",
             "", "| Experiment | Case | Backend | p10 | Median | p90 |", "|---|---|---|---:|---:|---:|"]
    for experiment in summary["experiments"]:
        for row in sorted(experiment["rows"], key=lambda r: (r["case"], r["backend"])):
            value = row["timing"]["graph_event"]
            times = " | ".join(f"{value[key] * 1000:.3f}" for key in ("p10_ms", "median_ms", "p90_ms"))
            backend = BACKEND_ALIASES.get(row["backend"], row["backend"])
            lines.append(f"| {experiment['experiment_id']} | {row['case']} | {backend} | {times} |")
    lines += ["", "Refactor warnings (>10% median regression) remain in JSON; do not discard losing cases.",
              "NCU values, when present, are diagnostics, not graph replay samples."]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Saved {len(summary['experiments'])} collections; {len(summary.get('profiles', []))} selected profiles")


if __name__ == "__main__":
    main()
