# Saved Evidence

These files contain portable exports of accepted numerical results. Every row,
error, timing sample and profiler metric is preserved. Private shell commands,
absolute machine paths and unrelated dirty-worktree metadata are omitted or made
portable. They are therefore **structured exports, not byte-identical copies**
of the original artifacts. [manifest.json](manifest.json) records the original
artifact SHA256, exported SHA256 and exact production-source hashes.

| File | Measurement |
|---|---|
| [fp32/full.json](fp32/full.json) | EXP-20261005-012 full FP32 sweep |
| [fp32/repeat.json](fp32/repeat.json) | EXP-20261005-013 fresh-process repeat |
| [w4a16/full.json](w4a16/full.json) | EXP-20261005-027 full quantized sweep |
| [w4a16/repeat.json](w4a16/repeat.json) | EXP-20261005-029 quantized repeat |
| [profiles/fp32.json](profiles/fp32.json) | EXP-20261005-014 captures; EXP-016 unit-normalized extraction |
| [profiles/w4a16.json](profiles/w4a16.json) | EXP-20261005-028 selected counters and launch counts |
| [checks/fp32.json](checks/fp32.json) | EXP-20261005-015 bounded regressions |
| [checks/w4a16.json](checks/w4a16.json) | EXP-20261005-030 adversarial quantized regressions |
| [checks/release-fp32.json](checks/release-fp32.json) | EXP-20261008-002 independently staged release smoke; 68 passed rows |
| [checks/release-w4a16.json](checks/release-w4a16.json) | EXP-20261008-002 staged quantized smoke; 36 passed rows |
| [bf16/full.json](bf16/full.json) | EXP-20261009-002 seven-shape BF16 RMSNorm sweep; 28 rows |
| [bf16/repeat.json](bf16/repeat.json) | EXP-20261009-003 fresh-process BF16 repeat; 28 rows |
| [checks/bf16.json](checks/bf16.json) | EXP-20261009-001 rounding, streams, replay and FP32 checks; 86 passed |
| [checks/bf16-sanitizer.json](checks/bf16-sanitizer.json) | EXP-20261009-007 memcheck subset; 47 passed |
| [checks/bf16-memcheck.log](checks/bf16-memcheck.log) | EXP-20261009-007 zero reported memory errors |
| [profiles/bf16.json](profiles/bf16.json) | EXP-20261009-005 selected NCU counters and Torch launch counts |
| [profiles/bf16/](profiles/bf16/) | Raw metric CSVs, Systems summaries and kernel-only Torch event projections |
| [checks/release-bf16.json](checks/release-bf16.json) | EXP-20261103-001 freshly staged BF16 correctness checks; no performance claim |

```bash
python scripts/check_evidence.py
```

Current device kernels and package sources match `production_source_sha256`.
The manifest's separate `source_snapshots` retain historical collection hashes,
the prior FP32 release and BF16 collection hashes. Artifacts with recorded source
maps name their own snapshot; older profile summaries without such maps are
explicitly marked. Historical FP32 results are not attributed to the BF16-updated files. The FP32 arithmetic
paths are unchanged. Public supporting
loaders and profiler runners are self-contained replacements; their current
hashes can differ from the historical harness. Benchmark arithmetic, catalogs
and saved numerical values are not reinterpreted as measurements of changed
device code. New collections record their own current provenance.

Native binary builds, complete profiler captures and new runs are ignored by
Git. Generate them locally with the commands in [profiling](../docs/profiling.md).
Do not edit accepted samples in place; add a new collection and explain its scope.

The clean release was checked with 39 CPU unit tests in two local Python
environments, a Python wheel build, profiler command previews and 104 GPU smoke
rows. Smoke used only three trials and small graphs; any timing fields in those
two smoke exports are diagnostic, not additional performance claims.

See [BF16 RMSNorm](../docs/bf16_rmsnorm.md) for the mixed-precision contract,
timing limitations and exact regression coverage. BF16 profiler projections
omit CPU/operator events and process identity while retaining the kernel events
and numerical metrics needed to recompute the published extracts. Original
artifact hashes and projection descriptions remain in the manifest.
