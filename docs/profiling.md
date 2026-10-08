# Profiling

Use separate correctness, benchmark and profiler collections. Profilers add
overhead, replay kernels and can change cache/clock behavior. Their durations
explain a mechanism; the unattached benchmark supplies the performance result.

## Collect A Bounded Bundle

Both wrappers print the exact commands by default. Add `--run` only when the
GPU, profiler executables and compute-counter permission are ready.

```bash
python scripts/profile_comparison.py --experiment EXP-20261008-201 \
  --output results/collected/EXP-20261008-201/fp32 --optimized
python scripts/profile_w4a16.py --experiment EXP-20261008-202 \
  --output results/collected/EXP-20261008-202/w4a16
```

The FP32 optimized bundle runs Torch Profiler, Nsight Systems, and nine bounded
Nsight Compute selections. W4A16 selects six custom kernels at M16, plus Torch
and Systems views at M1/M16. Each NCU collection targets one matching launch;
this is not exhaustive coverage of every shape or every launch in split-K.
The runner stops on failure, preserves logs/manifests, and enforces a 600-second
limit per subprocess group. Reports are not overwritten.

## Open The Reports

Generated paths appear in the collection manifest and printed commands.

```bash
# FP32 timeline:
nsys-ui results/collected/EXP-20261008-201/fp32/comparison.nsys-rep

# FP32 kernel counters (one example):
ncu-ui results/collected/EXP-20261008-201/fp32/rmsnorm_4096x4096--cuda_fast.ncu-rep

# W4A16 timeline and counters:
nsys-ui results/collected/EXP-20261008-202/w4a16/w4a16.nsys-rep
ncu-ui results/collected/EXP-20261008-202/w4a16/hidden_m16--cute_rows.ncu-rep
```

Torch Profiler produces `*.trace.json` Chrome traces and text operator tables.
Open a trace in a compatible trace viewer, for example Perfetto's UI, then
inspect CPU operator ranges, GPU launches, copies and idle gaps. These are
exported traces, not TensorBoard event logs; installing TensorBoard alone does
not turn this directory into a TensorBoard profiling run. Check a trace for
personal paths before sharing it with any external viewer.

## What To Read

| Tool | Question | Important caveat |
|---|---|---|
| Torch Profiler | Which Python/PyTorch operations launched kernels or copies? | CPU operator time and CUDA time overlap; do not sum them as serial latency |
| Nsight Systems | How do host ranges, launch gaps, streams and GPU work line up? | Synchronization and first-call compilation can dominate the visible timeline |
| Nsight Compute | Which kernel resources or dependencies constrain issue/throughput? | High occupancy alone does not imply high useful issue or speed |

For RMSNorm, compare kernels per call, registers/thread, occupancy, memory
throughput and reload behavior. Eager PyTorch launches seven kernels in the
profiled case; each custom path launches one, but compiled PyTorch also launches
one. Fusion is not evidence of a sevenfold latency gain.

For FP32 projection, inspect the main tile's eligible warps, issue-active rate,
long-scoreboard stalls, DRAM traffic and Tensor Core pipeline activity. Split-K
has an additional finish launch: a main-kernel-only NCU selection is not the
total projection latency. Occupancy and dependency chains must be considered
together. All nine saved candidate selections report zero tensor-pipeline
elapsed activity; do not describe these as Tensor Core results.

For W4A16, reduced bytes can coexist with poor arithmetic throughput. Compare
unpacking/conversion work, live accumulators and register usage between one-row
and four-row variants. The recorded Triton row-reuse variant rises from 40 to
72 registers/thread; the six custom selections again show zero tensor-pipeline
activity. Dense FP16 can exploit matrix hardware that these SIMT paths do not.

NCU CSV duration units are adaptive. Convert ms/us/ns/s explicitly and retain
the raw unit. [test_profile_units.py](../tests/unit/test_profile_units.py) guards
against interpreting milliseconds as microseconds. Logical payload bandwidth
from the benchmark and DRAM throughput from NCU use different byte models.

The saved [FP32](../results/profiles/fp32.json) and
[W4A16](../results/profiles/w4a16.json) summaries retain normalized counter
values and per-call launch counts. Large binary captures and full traces are
generated locally, not checked in. No project command changes NVIDIA counter
permissions or requires modifying the driver configuration.
