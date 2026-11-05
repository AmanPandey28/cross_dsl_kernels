"""Validate additive CUDA collections and recompute their profiler summaries."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

COUNTS = {
    "results/cuda/gemv/full.json": 72,
    "results/cuda/gemv/repeat.json": 72,
    "results/checks/cuda-gemv.json": 150,
    "results/checks/cuda-gemv-memcheck.json": 126,
    "results/checks/cuda-gemv-racecheck.json": 126,
    "results/cuda/rmsnorm/full.json": 28,
    "results/cuda/rmsnorm/paired.json": 16,
    "results/cuda/rmsnorm/paired-repeat.json": 16,
    "results/checks/cuda-rmsnorm.json": 86,
    "results/checks/cuda-rmsnorm-memcheck.json": 47,
    "results/checks/cuda-release-gemv.json": 150,
    "results/checks/cuda-release-rmsnorm.json": 86,
    "results/checks/cuda-simt.json": 150,
    "results/checks/cuda-simt-release.json": 150,
}


def check_rows(data, count):
    rows = data.get("rows", data.get("checks"))
    if data["status"] != "PASS" or len(rows) != count:
        raise ValueError("unexpected CUDA coverage")
    if not data.get("inputs_unchanged", True):
        raise ValueError("CUDA inputs changed")
    for row in rows:
        if row["status"] != "PASS":
            raise ValueError("failed CUDA row")
        for key in ("correctness", "graph_correctness"):
            if key in row and row[key]["status"] != "PASS":
                raise ValueError(f"failed CUDA {key}")
        for summary in row.get("timing", {}).values():
            if len(summary["samples_ms"]) != data.get("rounds", data.get("repeat")):
                raise ValueError("missing CUDA timing sample")


def profile_summary(root):
    from summarize_gemv_simt import profiles as projection_profiles
    from summarize_rmsnorm_bf16 import profiles as normalization_profiles

    def launches(directory):
        result = []
        data = json.loads((directory / "torch/results.json").read_text())
        for row in data["rows"]:
            path = directory / "torch" / row["torch_profile"]["trace"]
            events = json.loads(path.read_text())["traceEvents"]
            counts = Counter(event["name"] for event in events if event.get("cat") == "kernel")
            result.append({"case": row["case"], "backend": row["backend"],
                           "kernel_names": dict(counts), "kernels_per_call": sum(counts.values()) / row["torch_profile"]["iterations"]})
        return result

    summary = {
        "gemv": {"ncu": projection_profiles(root / "results/cuda/gemv/profiles/full"),
                 "counters": projection_profiles(root / "results/cuda/gemv/profiles/counters"),
                 "torch_launches": launches(root / "results/cuda/gemv/profiles/full"),
                 "simt": {"ncu": projection_profiles(root / "results/cuda/gemv/profiles/simt"),
                          "torch_launches": launches(root / "results/cuda/gemv/profiles/simt")}},
        "rmsnorm": normalization_profiles(root / "results/cuda/rmsnorm/profiles"),
    }

    def relative(value):
        if isinstance(value, dict):
            return {key: relative(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relative(item) for item in value]
        if isinstance(value, str) and value.startswith(str(root) + "/"):
            return str(Path(value).relative_to(root))
        return value

    return relative(summary)


def instructions(path):
    result = {}
    current = None
    for line in path.read_text().splitlines():
        match = re.search(r"Function : (\S+)", line)
        if match:
            current = re.sub(r"_GLOBAL__N__[0-9a-f]{8}_", "_GLOBAL__N__UNIT_", match[1])
            result[current] = []
        elif current:
            result[current].extend(re.findall(r"/\* (0x[0-9a-f]+) \*/", line))
    return result


def resource_usage(path):
    result = {}
    current = None
    for line in path.read_text().splitlines():
        if line.startswith(" Function "):
            current = re.sub(r"_GLOBAL__N__[0-9a-f]{8}_", "_GLOBAL__N__UNIT_",
                             line.removeprefix(" Function ").removesuffix(":"))
        elif current and "REG:" in line:
            result[current] = line.strip()
    return result


def validate(root, loaded):
    counts = {}
    for name, count in COUNTS.items():
        check_rows(loaded[name], count)
        counts[name] = count
    manifest = json.loads((root / "results/manifest.json").read_text())
    aliases = manifest["interface_aliases"]
    for old, new in aliases["paths"].items():
        if (root / old).exists() or not (root / new).is_file():
            raise ValueError("projection source/interface path disagrees with aliases")
    for old, new in aliases["artifact_paths"].items():
        artifact = manifest["artifacts"][new]
        if artifact["previous_export_path"] != old or artifact["previous_export_sha256"] != artifact["export_sha256"]:
            raise ValueError("historical artifact bytes changed during relocation")
    current = manifest["production_source_sha256"]
    for name in ("results/checks/cuda-simt.json", "results/checks/cuda-simt-release.json"):
        recorded = loaded[name]["source_sha256"]
        if any(recorded.get(source) != digest for source, digest in current.items()):
            raise ValueError("current projection validation has incomplete or stale source provenance")
    for name in ("results/cuda/rmsnorm/paired.json", "results/cuda/rmsnorm/paired-repeat.json"):
        data = loaded[name]
        if len(data["fp32_checks"]) != 64 or any(row["status"] != "PASS" for row in data["fp32_checks"]):
            raise ValueError("failed additional FP32 coverage")
        if not all(all(paths.values()) for paths in data["bitwise_before_after"].values()):
            raise ValueError("original/current outputs differ")
        for source, expected in data["before_source_sha256"].items():
            path = root / "results/snapshots/rmsnorm_before" / source
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("original source snapshot changed")
    for name, text in (("cuda-gemv-memcheck.log", "ERROR SUMMARY: 0 errors"),
                       ("cuda-gemv-racecheck.log", "0 hazards displayed (0 errors, 0 warnings)"),
                       ("cuda-rmsnorm-memcheck.log", "ERROR SUMMARY: 0 errors")):
        if text not in (root / "results/checks" / name).read_text():
            raise ValueError(f"missing sanitizer summary: {name}")
    if profile_summary(root) != loaded["results/cuda/profiles.json"]:
        raise ValueError("CUDA profiles disagree with exported counters/events")
    directory = root / "results/cuda/rmsnorm/codegen"
    before, after = (instructions(directory / f"{version}-sass.log") for version in ("before", "after"))
    old_resources, new_resources = (resource_usage(directory / f"{version}-resources.log") for version in ("before", "after"))
    proof = loaded["results/cuda/rmsnorm/codegen/codegen.json"]
    if len(before) != 15 or before != after or old_resources != new_resources or proof["status"] != "PASS":
        raise ValueError("CUDA device instructions differ")
    for row in proof["functions"]:
        value = hashlib.sha256("".join(before[row["symbol"]]).encode()).hexdigest()
        if value != row["before_sha256"] or value != row["after_sha256"] or not row["resources_equal"]:
            raise ValueError("CUDA codegen proof disagrees with instruction encodings")
        if old_resources[row["symbol"]] != row["before_resources"] or new_resources[row["symbol"]] != row["after_resources"]:
            raise ValueError("CUDA codegen proof disagrees with resource reports")
    return counts
