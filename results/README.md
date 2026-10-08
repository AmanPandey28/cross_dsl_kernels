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

```bash
python scripts/check_evidence.py
```

The device kernels and package sources match these hashes. Public supporting
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
