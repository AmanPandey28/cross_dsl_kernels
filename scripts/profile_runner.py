"""Run bounded profiler jobs, preserving failed and incomplete collections."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_run(experiment: str, output: Path) -> None:
    if not re.fullmatch(r"EXP-\d{8}-\d{3}", experiment):
        raise ValueError("provide an EXP-YYYYMMDD-NNN experiment ID")
    if output.exists():
        raise ValueError("output already exists; choose a fresh experiment path")


def run_commands(experiment: str, output: Path, jobs: list[tuple[str, list[str]]]) -> int:
    validate_run(experiment, output)
    output.mkdir(parents=True)
    manifest = {
        "experiment_id": experiment,
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "jobs": [],
    }
    for name, command in jobs:
        print(f"Running {name}", flush=True)
        try:
            with subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, start_new_session=True,
            ) as process:
                try:
                    log, _ = process.communicate(timeout=600)
                    code = process.returncode
                except subprocess.TimeoutExpired:
                    # Kill the profiler's workload too, using this job's own group.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    log, _ = process.communicate()
                    code = 124
                    log += "\nCollection exceeded the 600-second limit.\n"
        except FileNotFoundError as error:
            code, log = 127, str(error)
        path = output / f"{name}.log"
        path.write_text(log)
        manifest["jobs"].append({
            "name": name, "command": shlex.join(command),
            "returncode": code, "log": str(path),
        })
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if code:
            print(f"Failed ({code}); inspect {path}", file=sys.stderr)
            return code
    print(f"Saved evidence in {output}")
    return 0
