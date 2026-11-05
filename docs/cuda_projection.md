# CUDA Decode Projection

Additional CUDA paths make ownership and reuse explicit without replacing the
original implementations or vendor baselines. This is FP32 SIMT computation,
not a Tensor Core GEMM. Read the [binding](../csrc/gemv/gemv_clean_ext.cpp),
[warp/four-row kernels](../csrc/gemv/gemv_clean.cu) and
[tiled kernel](../csrc/gemv/gemm_tiled.cu) alongside the benchmark.

## Contract And Mapping

`X[M,K] @ W[K,N] + bias[N] -> Y[M,N]`, with optional bias and KN or NK physical
weight storage. Inputs/output and accumulation are FP32; TF32 is not substituted.
Tensors must be contiguous on one CUDA device. Output is newly allocated and
inputs are unchanged. New entry points use `None`, not an empty bias sentinel.

| Backend | Entry | Ownership |
|---|---|---|
| `cuda` | Original `decode_gemv_warp` | Original warp baseline, not older v0 |
| `cuda_fast` | Original `decode_gemv_rows` | Original four-row path |
| `clean_warp` | `decode_gemv_warp_clean` | Eight warp-owned outputs/CTA |
| `clean_rows` | `decode_gemv_rows_clean` | Separate KN/NK mappings, four-row reuse |
| `tiled` | `decode_gemm_tiled` | 8x32 output tile; shared X/W; K tiles of 32 |
| `cublas` | Existing vendor wrapper | SGEMV at M1, SGEMM otherwise; bias if present |

`cublaslt` and `pytorch` are optional benchmark backends. The separate extension
uses a source-hashed module name; original module names and default dispatch
remain intact. The binding validates dtype/ranks/layout/devices/bias, bounds
byte-offset products and grid dimensions, guards the device and launches on
PyTorch's current stream. No transpose/prepack is hidden inside timed calls.

The warp baseline maps a flattened output to `(row,column)`. Lanes visit
`k=lane+32*i`; five shuffle stages reduce the sum, then lane zero applies bias
and stores. NK makes lane-wise weight reads adjacent; KN makes them strided
across K. Output-tail returns are whole-warp and precede no CTA barrier.

KN four-row ownership maps lanes across N and warps across K, reusing each
weight for four rows. Shared partials combine reduction pieces. NK instead
maps lanes along contiguous K and gives each warp an output column while
retaining four-row reuse. Layout changes the physical mapping, not the math.

The tiled GEMM stages 8x32 X and 32x32 W operands in shared memory and reuses
them across an 8x32 output tile. Per-thread FP32 tile partials shorten the
accumulation chain. The first barrier orders loads before consumers; the second
orders consumers before overwriting the next tile. Mask reads to zero and
store only valid outputs. Do not return partial blocks before those barriers.
Apply bias once. No matrix-hardware operation is implied by the GEMM label.

The unchanged FP64 oracle gates are `max_abs_err <= 2e-4*max(1,sqrt(K/1024))`
and relative L2 <= `1e-5`. Relative error near zero is diagnostic, not the sole gate.

## Results And Lessons

Two collections pass 72 rows each: 12 cases, six backends, five warmups,
30 shuffled/reversed paired samples/mode and 30 calls/graph. Graph medians, us:

| Case | Clean four-row, full/repeat | Tiled, full/repeat | cuBLAS, full/repeat |
|---|---:|---:|---:|
| M16, K=N=4096, KN | 2227.64 / 2556.47 | 1732.49 / 1316.79 | 292.57 / 273.16 |
| M16, K=4096, N=11008, KN | 5724.77 / 5702.50 | 4142.89 / 3942.21 | 1200.02 / 1694.79 |

Tiling helps these custom M16 comparisons but still loses to cuBLAS and at M1.
Clean NK M1 regresses about 7.7/7.4% against the earlier four-row path. No
universal speedup or default-dispatch promotion is claimed.

KN four-row profiling reports 48 registers/thread and 81.16% occupancy versus
the earlier path's 40 and 97.71%; lower occupancy can still give better timing.
Tiled KN issue-active is 23.80%. Shared-store bank-conflict counts are
32,632,387 for NK tiled versus 29,752 for KN. Padding is a next experiment,
not an implemented fix. Selected custom tensor-pipeline activity is zero.

M1 has near 0.5 useful FLOP/byte before cache/reuse effects. Small-M GEMM can
reuse weights across rows, but the actual ownership, shared traffic and scalar
instruction cost must exploit it. Higher theoretical intensity alone is not speed.

```bash
make gemv-list PYTHON=python EXP=EXP-20261105-204
make gemv-check PYTHON=python EXP=EXP-20261105-205
make gemv-benchmark PYTHON=python EXP=EXP-20261105-206
make gemv-profile-preview PYTHON=python EXP=EXP-20261105-207
```

Use fresh IDs/paths. Validation passes 150 checks; bounded memcheck/racecheck
each pass 126 with zero errors/warnings. Graphs are checked with poisoned output
and changed fixed-address inputs before restoring them. Two-device mismatch
execution and arbitrary extreme inputs remain untested.

The benchmark primes/reuses events, separates event/wall/graph samples and
excludes setup, references and JIT from steady-state measurements. Display,
clock, thermal and power variability remain. The profile runner previews jobs;
`--run` collects bounded captures. One case/layout per NCU process avoids bool
demangling ambiguity; one intended launch is selected after warmup. CSVs require
one kernel and explicit unit normalization. Missing counters are unknown, not
zero. See [CUDA evidence](../results/cuda/README.md) for samples and projections.
