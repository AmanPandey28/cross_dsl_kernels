#!/usr/bin/env python3
"""Generate comparison tables, plots, and profiler extracts from saved evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

BACKENDS = (
    "pytorch",
    "torch_compile",
    "cuda",
    "triton",
    "cute",
    "cublas",
    "cublaslt",
    "cuda_fast",
    "triton_fast",
    "cute_fast",
)
COLORS = (
    "#6b7280",
    "#202020",
    "#00856a",
    "#c34836",
    "#a57516",
    "#286bc1",
    "#875f98",
    "#159bdd",
    "#d94886",
    "#749f25",
)
METRICS = {
    "duration_us": "gpu__time_duration.sum",
    "dram_gbps": "dram__bytes.sum.per_second",
    "dram_sol_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "sm_sol_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "registers": "launch__registers_per_thread",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "waves": "launch__waves_per_multiprocessor",
    "eligible_warps": "smsp__warps_eligible.avg.per_cycle_active",
    "issue_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "long_scoreboard_per_issue": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "tensor_elapsed_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
}


def latency(row: dict) -> float:
    return row["graph_timing"]["cuda_event"]["median_ms"] * 1000


def metric_value(name: str, raw: str, unit: str) -> dict:
    value = float(raw.replace(",", ""))
    if name == "duration_us":
        factors = {
            "ns": 0.001,
            "us": 1.0,
            "ms": 1000.0,
            "s": 1e6,
            "nsecond": 0.001,
            "usecond": 1.0,
            "msecond": 1000.0,
            "second": 1e6,
        }
        if unit not in factors:
            raise ValueError(f"unknown NCU duration unit: {unit}")
        return {
            "value": value * factors[unit],
            "unit": "us",
            "raw_value": value,
            "raw_unit": unit,
        }
    return {"value": value, "unit": unit}


def profile_extracts(profiles: Path, output: Path) -> None:
    studies = []
    for path in sorted(profiles.glob("*-csv.log")):
        with path.open() as handle:
            reader = csv.DictReader(handle)
            units = next(reader)
            for row in reader:
                studies.append(
                    {
                        "case": path.name.removesuffix("-csv.log"),
                        "kernel": row["Kernel Name"],
                        "metrics": {
                            name: metric_value(name, row[key], units[key])
                            for name, key in METRICS.items()
                        },
                    }
                )
    launches = []
    data = json.loads((profiles / "torch/results.json").read_text())
    for row in data["rows"]:
        trace = json.loads(
            (profiles / "torch" / Path(row["torch_profile"]["trace"]).name).read_text()
        )
        counts = Counter(event.get("cat") for event in trace["traceEvents"])
        iterations = row["torch_profile"]["iterations"]
        launches.append(
            {
                "case": row["case"],
                "backend": row["backend"],
                "kernels_per_call": counts["kernel"] / iterations,
                "copies_per_call": counts["gpu_memcpy"] / iterations,
            }
        )
    (output / "profile_summary.json").write_text(
        json.dumps(
            {
                "source": str(profiles),
                "ncu": studies,
                "torch_launches": launches,
                "metrics": METRICS,
                "note": "NCU is diagnostic; durations may reverse unprofiled rankings.",
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Profiler Extract",
        "",
        f"Source: `{profiles}`",
        "",
        "| NCU case | Duration us | DRAM GB/s | DRAM SOL % | L2 hit % | Registers | Occupancy % | Issue % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for study in studies:
        values = [
            study["metrics"][key]["value"]
            for key in (
                "duration_us",
                "dram_gbps",
                "dram_sol_pct",
                "l2_hit_pct",
                "registers",
                "occupancy_pct",
                "issue_pct",
            )
        ]
        lines.append(f"| {study['case']} | {' | '.join(f'{v:.2f}' for v in values)} |")
    lines += [
        "",
        "| Torch case | Backend | Kernel launches/call | Device copies/call |",
        "|---|---|---:|---:|",
    ]
    for row in launches:
        lines.append(
            f"| {row['case']} | {row['backend']} | {row['kernels_per_call']:g} | {row['copies_per_call']:g} |"
        )
    lines += [
        "",
        "Counts come from trace categories, not aggregate CUDA activity counts.",
        "Metric names and units, including stalls and tensor utilization, remain in JSON.",
    ]
    (output / "profile_summary.md").write_text("\n".join(lines) + "\n")


def summarize(data: dict, output: Path) -> None:
    rows = data["rows"]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["case"], {})[row["backend"]] = row
    lines = [
        "# Results Summary",
        "",
        f"Source: `{data['experiment_id']}`; {len(rows)} rows passed.",
        "",
        f"FP32, SM120, TF32 disabled. {data['warmup']} warmups, {data['repeat']} trials; times below are microseconds",
        f"per call in a {data['graph_calls']}-call CUDA Graph batch. Raw JSON also includes ordinary CUDA-event",
        "and synchronized wall times, setup plus first call, p10/p90 and every sample.",
        "",
    ]
    for op in ("rmsnorm", "gemv", "rope_kv"):
        selected = [row for row in rows if row["op"] == op]
        cases = list(dict.fromkeys(row["case"] for row in selected))
        winners = Counter(
            min(grouped[case], key=lambda b: latency(grouped[case][b]))
            for case in cases
        )
        lines += [
            f"## {op}",
            "",
            f"Lowest observed medians over {len(cases)} cases: {dict(winners)}.",
            "",
            f"![{op} latency]({op}-latency.png)",
            "",
            "| Case | " + " | ".join(BACKENDS) + " |",
            "|---|" + "---:|" * len(BACKENDS),
        ]
        fields = (
            "case",
            "backend",
            "graph_us",
            "graph_p10_us",
            "graph_p90_us",
            "logical_gbps",
            "tflops",
        )
        with (output / f"{op}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in selected:
                graph = row["graph_timing"]["cuda_event"]
                writer.writerow(
                    {
                        "case": row["case"],
                        "backend": row["backend"],
                        "graph_us": latency(row),
                        "graph_p10_us": graph["p10_ms"] * 1000,
                        "graph_p90_us": graph["p90_ms"] * 1000,
                        "logical_gbps": row["graph_logical_gbps"],
                        "tflops": row["graph_effective_tflops"],
                    }
                )
        for case in cases:
            cells = [
                f"{latency(grouped[case][b]):.3f}" if b in grouped[case] else "-"
                for b in BACKENDS
            ]
            lines.append(f"| {case} | {' | '.join(cells)} |")
        lines += [
            "",
            "Close rankings are observed medians, not statistically established winners.",
            "",
        ]
    ceiling = data["bandwidth_calibration"]["graph_cuda_event_gbps"]
    rms = grouped["rmsnorm_4096x4096"]["cuda"]
    lines += [
        "## Bandwidth and Limits",
        "",
        f"The same-process 512 MiB-traffic FP32 copy measured **{ceiling:.2f} GB/s**.",
        f"CUDA RMSNorm 4096x4096: **{rms['graph_logical_gbps']:.2f} logical GB/s**,",
        f"**{rms['percent_copy_ceiling']:.2f}%** of the copy rate.",
        "The unique-payload model `4*(4*rows*hidden+hidden)` excludes staged residual",
        "rereads and is not hardware DRAM traffic. NCU uses a different denominator.",
        "",
        "One native process sweep; clocks and power observed, not fixed. Backend order",
        "is seeded and randomized, but thermal state still varies over sequential rows.",
        "Prepared CuTe excludes DLPack/JIT overhead. Graph replay excludes CPU Lt",
        "descriptor/heuristic work. These are fixed-buffer microbenchmarks, not model latency.",
        "Repeat in independent processes before accepting a dispatch policy. Custom paths",
        "remain FP32 contiguous prototypes; B200 and current container parity are unmeasured.",
    ]
    (output / "results_summary.md").write_text("\n".join(lines) + "\n")


def plots(rows: list[dict], output: Path, experiment: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for op in ("rmsnorm", "gemv", "rope_kv"):
        selected = [row for row in rows if row["op"] == op]
        cases = list(dict.fromkeys(row["case"] for row in selected))
        fig, ax = plt.subplots(figsize=(11, 5))
        for backend, color in zip(BACKENDS, COLORS, strict=True):
            matching = {
                row["case"]: row for row in selected if row["backend"] == backend
            }
            if matching:
                ax.plot(
                    [cases.index(case) for case in matching],
                    [latency(row) for row in matching.values()],
                    marker="o",
                    markersize=4,
                    label=backend,
                    color=color,
                )
        ax.set_xticks(range(len(cases)), cases, rotation=35, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel("Graph replay per call (microseconds, median)")
        ax.set_title(f"{op}: FP32, SM120 | {experiment}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"{op}-latency.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.result.read_text())
    if data["status"] != "PASS" or any(
        "graph_timing" not in row for row in data["rows"]
    ):
        parser.error("an accepted sweep with graph measurements is required")
    if args.output.exists():
        parser.error("choose a fresh analysis directory")
    args.output.mkdir(parents=True)
    summarize(data, args.output)
    plots(data["rows"], args.output, data["experiment_id"])
    if args.profiles:
        profile_extracts(args.profiles, args.output)


if __name__ == "__main__":
    main()
