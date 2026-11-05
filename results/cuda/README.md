# CUDA Source And Measurement Evidence

Current projection sources and tools use `gemv_simt`, with `simt_warp` and
`simt_rows` backend identifiers. Historical collection labels and captured
symbols remain unchanged inside raw evidence. The [manifest](../manifest.json)
records interface and artifact-path aliases; relocated capture files retain
identical bytes. Trace index basenames and the derived profile summary follow
those paths. `results/checks/cuda-simt.json` and `cuda-simt-release.json` are
fresh 150-check validation collections. `gemv/profiles/simt/` contains current
counter and kernel-event exports. `gemv/interfaces.json` records the
identifier-only source comparison, not binary identity or improved performance.

These are additive portable exports of the November CUDA collections. Earlier
FP32/BF16/W4A16/profile exports remain byte-identical. The
[manifest](../manifest.json) separates current production hashes from historical
collection snapshots and records original/exported artifact hashes. Numerical
rows, samples, orders and telemetry are unchanged; private commands, nonpublic
paths and unrelated worktree metadata are omitted.

| Collection | Evidence |
|---|---|
| [GEMV full](gemv/full.json), [repeat](gemv/repeat.json) | 72 rows each; original/SIMT/tiled/cuBLAS paths |
| [GEMV checks](../checks/cuda-gemv.json) | 150 numerical/contract checks |
| GEMV memcheck/racecheck JSON/logs in `results/checks/` | 126 checks each; zero errors/warnings |
| [RMSNorm full](rmsnorm/full.json) | 28 rows; unchanged seven-shape BF16 harness |
| [RMSNorm paired](rmsnorm/paired.json), [repeat](rmsnorm/paired-repeat.json) | 16 BF16 rows and 64 extra FP32 checks per process |
| [RMSNorm checks](../checks/cuda-rmsnorm.json) | 86 current-build checks |
| RMSNorm memcheck JSON/log in `results/checks/` | 47 checks; zero memory errors |
| [RMSNorm codegen](rmsnorm/codegen/) | Raw instruction/resource listings and checked 15-kernel comparison |
| [Original source](../snapshots/rmsnorm_before/) | Exact preserved sources for paired comparison |
| [Profiles](profiles.json) | Recomputed from included NCU CSVs and kernel events |
| `results/checks/cuda-release-{gemv,rmsnorm}.json` | Fresh staged-checkout correctness, not performance claims |

The manifest's `historical_support_source_aliases` retains the collection-time
benchmark identity. Its current SIMT identifiers differ; `interface_aliases`
maps the interfaces without claiming that current names were used at collection
time. Historical supporting hashes stay historical.

`gemv/profiles/full/` has seven selected NCU CSVs, Systems summaries and 18 Torch
kernel-event studies; `gemv/profiles/counters/` adds seven byte/bank CSVs.
`rmsnorm/profiles/` has two NCU CSVs, Systems summaries and eight Torch studies.
Process identity and CPU/operator trace events are omitted. Complete profiler
binaries and compiled libraries stay local and can be regenerated.

NCU duration is diagnostic, not unprofiled graph latency. Logical bytes are
not DRAM bytes; sustained theoretical references are not measured-copy ceilings.
Adaptive units are retained in CSVs and normalized in the summary. Both wins
and losses are explained in [CUDA projection](../../docs/cuda_projection.md)
and [CUDA RMSNorm](../../docs/cuda_rmsnorm.md).

```bash
make evidence-check PYTHON=python
```

The checker verifies hashes, source snapshots, raw medians, coverage, numerical
gates, sanitizer summaries, instruction encodings and recomputed profiles.
Do not edit accepted exports in place to accommodate later source changes.
