"""Validate saved numerical evidence and production-source provenance on CPU."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_samples(value):
    if isinstance(value, dict):
        if "samples_ms" in value:
            samples = value["samples_ms"]
            if not samples or not all(math.isfinite(x) and x >= 0 for x in samples):
                raise ValueError("invalid timing sample")
            if not math.isclose(statistics.median(samples), value["median_ms"], rel_tol=1e-9):
                raise ValueError("saved median disagrees with samples")
        for item in value.values():
            check_samples(item)
    elif isinstance(value, list):
        for item in value:
            check_samples(item)


def validate(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "results/manifest.json").read_text())
    for name, expected in manifest["production_source_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"production source changed: {name}")
    loaded = {}
    for name, record in manifest["artifacts"].items():
        path = root / name
        if sha256(path) != record["export_sha256"]:
            raise ValueError(f"evidence changed: {name}")
        data = json.loads(path.read_text())
        check_samples(data)
        loaded[name] = data
    expected_counts = {
        "results/fp32/full.json": 226, "results/fp32/repeat.json": 72,
        "results/w4a16/full.json": 144, "results/w4a16/repeat.json": 36,
        "results/checks/fp32.json": 34, "results/checks/w4a16.json": 24,
        "results/checks/release-fp32.json": 68,
        "results/checks/release-w4a16.json": 36,
    }
    counts = {}
    for name, expected in expected_counts.items():
        data = loaded[name]
        rows = data.get("rows", data.get("checks"))
        if data["status"] != "PASS" or len(rows) != expected:
            raise ValueError(f"unexpected evidence coverage: {name}")
        if any(row["status"] != "PASS" for row in rows):
            raise ValueError(f"failed saved row: {name}")
        counts[name] = len(rows)
    for name, expected in (("results/profiles/fp32.json", 9),
                           ("results/profiles/w4a16.json", 6)):
        ncu = loaded[name]["ncu"]
        if len(ncu) != expected:
            raise ValueError(f"unexpected NCU coverage: {name}")
        if any(row["metrics"]["duration_us"]["unit"] != "us" for row in ncu):
            raise ValueError(f"unnormalized profiler duration: {name}")
    return {"sources": len(manifest["production_source_sha256"]), "rows": counts}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
