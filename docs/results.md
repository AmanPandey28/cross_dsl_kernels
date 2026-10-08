# Results And Lessons

All measurements are local RTX 5050 Laptop GPU / SM120. Data, dtype, workload
and timing view matter more than a single cross-language ranking.
Saved samples are linked in [the evidence index](../results/README.md).

## Evidence Coverage

| Collection | Passed rows | Scope |
|---|---:|---|
| FP32 full, EXP-20261005-012 | 226/226 | 25 cases: RMSNorm, projection and RoPE/cache; applicable backends |
| FP32 fresh process, EXP-20261005-013 | 72/72 | Eight selected cases, new seed, 30 trials of 30-call graph batches |
| W4A16 full, EXP-20261005-027 | 144/144 | 16 cases x 9 backends; 20 trials of 10-call graph batches |
| W4A16 fresh process, EXP-20261005-029 | 36/36 | Four cases x 9 backends; 30 trials of 20-call graph batches |
| FP32 bounded contract/memcheck, EXP-20261005-015 | 34/34 | Tail, side-stream and invalid-input checks |
| W4A16 bounded contract/memcheck, EXP-20261005-030 | 24/24 | Full signed-code range, changed fixed-buffer inputs, tails and streams |

Original memcheck collections reported zero errors. The checked-in JSON contains
numerical checks; large original logs/captures remain local. Counts describe
backend-case combinations, not hundreds of distinct algorithm implementations.
Profiling covered 36 FP32 and 18 W4A16 Torch traces, one Systems timeline per
suite, and nine/six selected NCU custom kernels. That is representative coverage,
not a profile of every possible input or launch.

The independently staged release also passed 39 CPU unit tests and 104 bounded
GPU smoke rows on 2026-10-08. These validate the public loaders and scripts after
removing local scaffolding; they do not replace the historical full sweeps or
establish new speedups.

## FP32 Progression

Fresh-process graph medians, microseconds/call; ratio is **earlier custom /
candidate of the same language**, not versus a vendor library:

| Case | Language | Prior/candidate ratio |
|---|---|---:|
| Hidden projection, M16, K=N=4096 | CUDA | 5.85x |
| Hidden projection, M16, K=N=4096 | Triton | 15.31x |
| Hidden projection, M16, K=N=4096 | CuTe | 1.48x |
| MLP-up projection, M16, K=4096, N=11008 | CUDA | 6.77x |
| MLP-up projection, M16, K=4096, N=11008 | Triton | 16.27x |
| MLP-up projection, M16, K=4096, N=11008 | CuTe | 1.68x |
| RMSNorm, 4096x4096 | CuTe | 1.56x |
| HND partial RoPE/cache, 16 tokens | CUDA / Triton / CuTe | 1.67x / 1.45x / 2.13x |

The previous Triton hidden M16 path is 8761.00 us versus 572.35 us for the
candidate; MLP-up is 24739.68 us versus 1520.94 us. Large ratios indicate
substantial inefficiency in the earlier custom baseline, not superiority over
every library. cuBLAS, cuBLASLt and compiled PyTorch remain necessary comparators.
Full per-case library timings are retained rather than collapsed into one score.

CUDA large-row RMSNorm changes from a 1.44x gain in the sweep to only 1.05x in
the repeat. Triton's reduced-warp large-row path loses (0.97x repeat); tiled
NK M1 projection also loses in important CUDA/Triton cases. These candidates
should not be promoted into a universal dispatcher.

CuTe large RMSNorm achieves 253.77 logical GB/s in the repeat, about 81.17% of
the same-run 312.64 GB/s copy rate. The numerator counts unique payload bytes,
not all kernel reads or measured DRAM traffic; neither number is architectural
peak bandwidth. The small and large-row regimes require different reasoning.

## Quantized Projection

Fresh-process graph medians, microseconds/call:

| Case | CuTe one-row or best custom | Original dense FP16 baseline |
|---|---:|---:|
| Hidden M1 | 56.79 (CuTe one-row) | 123.20 |
| MLP-up M1 | 166.99 (CuTe one-row) | 836.43 |
| Hidden M16 | 601.34 (CuTe four-row) | 116.52 |
| MLP-up M16 | 1676.66 (CuTe four-row) | 675.91 |

This table intentionally includes losses. M1 can benefit from compressed weight
traffic, while M16 provides enough reuse for dense matrix hardware to dominate
the current SIMT implementations. Pre-dequantized, arithmetic-matched FP16 and
eager-dequantization baselines are also saved. There is **no optimized W4A16
library baseline** yet, so a dense win does not establish a competitive INT4 GEMM.

The full sweep's worst kernel relative L2 is about `6.756e-5` against the quantized
oracle. Separately, random-input output error relative to original dense weights
is about 11.4--11.8% for larger K. Correct decoding is not preserved model accuracy.
The custom quantizer is not calibrated and these are synthetic tensors, not an
accuracy evaluation on a transformer checkpoint.

Clocks materially change exact ratios: dense MLP-up M16 moves from 258.89 us in
the sweep to 675.91 us in the repeat. The directional M1 win/M16 loss repeats,
but choosing the largest observed ratio would overstate the evidence.

## Preserved Failure Lessons

- Full-K FP32 Triton accumulation failed a strict gate in a difficult regime.
  Split-K changed reduction order and shortened the chain; acceptance gates were
  retained. Do not hide numerical differences by loosening tolerance after a failure.
- A larger tile or fewer warps is not automatically better. Register pressure,
  layout and active warps interact, especially for large hidden dimensions.
- A successful graph replay can be a false positive if prior correct output is
  still present. Poison outputs and reset mutated state before checking replay.
- NCU duration units can change with workload length. Normalize units before
  presenting microseconds or comparing kernel times.
- Occupancy alone did not explain projection performance: profiled CUDA/CuTe/
  Triton main kernels had approximately 97.3/79.7/59.8% occupancy but only
  13.7/41.1/53.2% issue-active rates. Dependency stalls matter.
- Compressing weights is not the same as accelerating matrix computation. Scale
  access, unpacking and conversion overhead become visible when traffic shrinks.

## Limits

Unlocked laptop clocks, a shared display GPU, an observed 35 W limit and 8 GB
memory constrain reproducibility and shape size. Sequential backend order is
randomized but thermal drift remains. No end-to-end serving throughput, batching
policy, backward pass, multi-GPU correctness, long-running stability or model
quality is claimed. The current suite is largely contiguous, forward-only, with
bounded shapes and architecture checks. SM100/B200/B300 execution and current
container reproduction are unvalidated. No universal DSL ranking is supported.

## Lessons And Next Steps

First reason about **workload regime, layout and ownership**, then choose the
language constructs. Fusion removes launches and intermediate traffic; residency
trades loads for registers; row reuse trades duplicate work for larger live state;
split-K trades dependency depth for scratch traffic and an extra launch.
Every trade needs a numerical gate and an independent timing repeat.

The highest-value next work is an arithmetic/format-matched optimized W4A16
baseline and one bounded matrix-hardware candidate. Then add thermal/ordering
repeats, model-quality evaluation, stable dispatcher decisions, registered operator
integration, and clean-environment reproduction. A later B200 path needs its own
source/tuning/results validation, not copied SM120 timings.
