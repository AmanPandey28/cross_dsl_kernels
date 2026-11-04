# BF16 Fused Residual RMSNorm

CUDA C++ and Triton accept contiguous, matching BF16 `x`, `residual` and
`weight` tensors on one device. `x` and `residual` are `[rows, hidden]`;
`weight` is `[hidden]`. Rows and hidden must be positive, and epsilon must be
finite and positive. Mixed dtypes, FP16 and strided inputs are rejected.
This extension does not add BF16 CuTe, BF16 dense GEMM, gradients or model integration.

## Numerical Contract

```python
r = x.float() + residual.float()
inv = torch.rsqrt(r.square().mean(dim=-1, keepdim=True) + eps)
y = (r * inv * weight.float()).bfloat16()
residual_out = r.bfloat16()
```

Both outputs are new allocations; inputs are unchanged. Normalization uses
the **unrounded FP32 sum**, not the BF16 `residual_out`. An in-place fused
normalizer that rounds the residual before reduction is not a matched baseline.
BF16 storage reduces payload size, but the reduction and scaling remain FP32.

## Implementation

- [CUDA binding](../csrc/rmsnorm/rmsnorm_ext.cpp): validates dtype, shape,
  device, contiguity and epsilon before dispatch.
- [CUDA kernels](../csrc/rmsnorm/rmsnorm_kernel.cu): one 256-thread CTA per row.
  Each thread sums squares, warp shuffles reduce within eight warps, and shared
  memory plus a block barrier combine the warp sums. `fast=True` retains FP32
  residual values in per-thread arrays for row capacities 256, 512, 1024, 2048,
  4096 and 8192. Invalid tail columns contribute zero but join the reduction.
  Above 8192, or with `fast=False`, a second pass recomputes the sum from the
  original inputs, never from rounded output. Scalar BF16 loads handle odd
  widths and contiguous offset views. Launches use the guarded device's current stream.
- [Triton](../src/crossdsl_kernels/triton/rmsnorm.py): one program per row,
  padded power-of-two vectors, masked loads/stores and FP32 reduction/scaling.
  BF16 hidden width is bounded at 65536. Default warps are 4 below a padded
  width of 2048, 8 below 8192, otherwise 16. The benchmark's `triton_fast`
  configuration uses the existing explicit override in `comparison_cases.py`;
  it is not a new default or a guarantee across shapes.
- [Reference](../src/crossdsl_kernels/references/rmsnorm.py): shared eager
  semantics, also used by `torch.compile`. Existing FP32 arithmetic paths are retained.

## Reproduce

Use the [recorded isolated environment](benchmarking.md). Each command needs a
fresh experiment/output path; CPU checks do not need a GPU.

```bash
make test evidence-check PYTHON=python
make bf16-check PYTHON=python EXP=EXP-20261103-101
make bf16-smoke PYTHON=python EXP=EXP-20261103-102
make bf16-benchmark PYTHON=python EXP=EXP-20261103-103
make bf16-profile-preview PYTHON=python EXP=EXP-20261103-104
# Execute the bounded preview only when ready to collect profiles:
python scripts/profile_rmsnorm_bf16.py --experiment EXP-20261103-104 \
  --output results/collected/EXP-20261103-104/bf16 --run
python scripts/summarize_rmsnorm_bf16.py \
  --results results/bf16/full.json results/bf16/repeat.json \
  --profiles results/profiles/bf16 --output results/analysis/bf16-recomputed
```

The benchmark catalog covers 1/16/256/4096 rows at hidden 4096, 16x8192,
the 3x513 odd tail and the 2x12289 CUDA fallback. Preparation and first call
are recorded separately. Twenty rounds use seeded shuffled backend order
followed by its reverse, with ten warmups and twenty captured calls per graph.
The public allocating APIs are compared, including eager/compiled outputs.

## Saved Results

[Full](../results/bf16/full.json) and [fresh-process repeat](../results/bf16/repeat.json)
each pass 28 backend-case rows, including direct and poisoned-output graph checks.
Both use the same seven shapes and four backends, with different seeds.

At **256x4096**, CUDA-event graph per-call medians in microseconds are:

| Backend | EXP-20261009-002 | EXP-20261009-003 |
|---|---:|---:|
| Eager PyTorch | 78.694 | 78.607 |
| Compiled PyTorch | 8.753 | 8.529 |
| CUDA fast | 14.942 | 14.702 |
| Triton fast | 8.258 | 7.667 |

CUDA improves on eager but **loses to compiled PyTorch**. Triton's advantage
over compiled PyTorch is modest in these two runs. Do not generalize from one
shape, compare different timing modes, or infer end-to-end inference speedups.
Ordinary CUDA-event and synchronized-wall distributions remain in the JSON.
Paired block-bootstrap intervals describe within-process sampling uncertainty,
not systematic clock drift or measurement bias.

**Timing limitation:** the existing caller creates fresh, lazily initialized
events for each `repeat=1` sample. Their overhead has not been independently
quantified. Ordinary event spans can include GPU idle gaps while Python submits
work; synchronized wall samples additionally include host overhead. Graph
per-call samples divide replay time by captured calls and use graph memory pools.
Laptop clocks were not locked, and the display shares the GPU. These historical
measurements are preserved, not silently corrected or recollected for publication.

## Correctness And Memory Checks

[The 86-check run](../results/checks/bf16.json) comprises 57 non-default-stream
checks (19 fixtures x CUDA baseline, CUDA fast and default Triton), six graph
checks, 15 rejection checks and eight FP32 regressions. Widths include
1, 31, 255, 256, 257, 512, 513, 1024, 2048, 4096, 8192, 8193, 12289 and 65536.
Fixtures exercise offset views, zero, cancellation, small/large values and
rounding-sensitive sums. Graph checks poison outputs, then change input values
at the same captured addresses and replay again, guarding against stale outputs.

The gate requires finite matching BF16 outputs, exact residual rounding,
relative L2 error <= 5e-4 and normalized maximum error <= 1. The denominator
is the larger neighboring BF16 spacing, floored at FP32's smallest positive
normal. This is **not a strict all-input one-ULP guarantee**, especially for
subnormal values. An additional oracle uses FP64 statistics after the FP32 add.
Overflow, extreme epsilon and all possible inputs are not certified.

[The memcheck subset](../results/checks/bf16-sanitizer.json) passes 47 checks;
the [saved log](../results/checks/bf16-memcheck.log) reports zero errors.
This is not proof of race freedom, all-input safety, or racecheck/synccheck coverage.

## Profiler Evidence

[Portable extracts](../results/profiles/bf16.json) include two selected NCU
kernels and eight Torch launch studies. Raw NCU metric CSVs, Systems summary
CSVs and kernel-only Torch event projections live in
[results/profiles/bf16/](../results/profiles/bf16/). Full binary reports and
CPU/operator trace events are not included; collect them locally for UI analysis.

At 4096x4096, NCU reports CUDA DRAM throughput of **237.99 decimal GB/s**
(82.71% sustained theoretical peak) and Triton **264.17 GB/s** (91.81%).
CUDA uses 44 registers/thread and 82.55% achieved occupancy; Triton uses
66 and 56.28%. Higher occupancy alone is therefore not a performance goal.
These are counter-based diagnostics under replay/cache policy and unlocked clocks,
not bandwidth divided by a measured copy ceiling or unprofiled benchmark latency.
Torch kernel categories show 12 eager kernels versus one custom/compiled kernel
per invocation in the selected traces; launch count is not itself a speedup.

## Provenance And Remaining Work

[The manifest](../results/manifest.json) retains the original artifact SHA256,
portable export SHA256, and separate historical collection, FP32 release and
BF16 source snapshots. Earlier results are not relabeled as runs of changed files.
Older profile summaries lack a collection source map and are explicitly marked
as such; a later release snapshot does not prove their collection source. The BF16 production
sources match their collection hashes; the standalone public loader and runner
are packaging replacements, not new measured device algorithms.

Next useful work is a bounded primed-event timing audit, matched fused-library
comparisons, numerical stress testing and pinned clean-room reproduction.
No BF16 CuTe extension or SM100/B200 performance result is claimed.
