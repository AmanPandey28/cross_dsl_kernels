# CUDA Residual RMSNorm

BF16 and FP32 inputs produce two new outputs without input mutation:

```python
r = x.float() + residual.float()
inv_rms = torch.rsqrt(r.square().mean(-1, keepdim=True) + eps)
y = (r * inv_rms * weight.float()).to(x.dtype)
residual_out = r.to(x.dtype)
```

Normalization uses the **unrounded FP32 sum**, not the BF16 residual output.
BF16 validation requires exact residual rounding, output error within one larger
neighboring BF16 spacing and relative L2 <= `5e-4`, with a wider-statistics oracle.
These are bounded checks, not an all-input overflow or stability proof.

## Read The Code

| Order | Source | Purpose |
|---|---|---|
| 1 | [binding](../csrc/rmsnorm/rmsnorm_ext.cpp#L10) | Dtype, device, shape, contiguity and epsilon checks |
| 2 | [dtype dispatch](../csrc/rmsnorm/rmsnorm_kernel.cu#L364) | BF16 versus preserved FP32 paths |
| 3 | [BF16 launcher](../csrc/rmsnorm/rmsnorm_kernel.cu#L127) | Guard, output allocation, current stream and capacity |
| 4 | [baseline](../csrc/rmsnorm/rmsnorm_kernel.cu#L55) | Recompute from original inputs after reduction |
| 5 | [retained kernel](../csrc/rmsnorm/rmsnorm_kernel.cu#L86) | Keep unrounded values across reduction |
| 6 | [reduction](../csrc/rmsnorm/rmsnorm_kernel.cu#L16) | Warp sums, shared handoff and both barriers |
| 7 | [paired benchmark](../benchmarks/rmsnorm_cuda_compare.py) | Original/current and recompute/retain comparison |

One 256-thread CTA owns a row. At 256x4096 there are 256 CTAs, eight warps/CTA
and 16 values/thread at `threadIdx.x + 256*i`. A thread's iterations are strided,
but neighboring lanes read neighboring columns. Each warp summarizes 512 values.
Five shuffle stages reduce each warp; lane zero publishes its partial. The first
barrier precedes warp zero's reduction of eight partials; the second publishes
the final row sum to every warp. Invalid columns contribute zero and still join
all collectives. Each thread evaluates its inverse RMS using `hidden`, not capacity.

The baseline rereads X/residual, avoiding early BF16 rounding. Retention replaces
that pass with live FP32 local values. Unrolling trades scheduling opportunities
for live state/code size; local arrays are not guaranteed physical registers.
`fast=False` or hidden > 8192 recomputes; otherwise capacities are
256/512/1024/2048/4096/8192. FP32 scalar, aligned float4 and retained paths remain.
Vectorization requires width divisible by four and 16-byte-aligned pointers.

## Evidence And Limits

Graph medians, us/call; 20 samples, ten warmups, 20 captured calls:

| Shape | Original/current retained | Fresh-process repeat |
|---|---:|---:|
| 256x4096 | 14.769 / 14.895 | 15.004 / 14.732 |
| 4096x4096 | 637.924 / 654.734 | 657.357 / 657.329 |
| 3x513 | 3.360 / 3.164 | 3.162 / 3.123 |
| 2x12289, both recompute | 25.864 / 26.114 | 25.742 / 25.844 |

No graph-median regression exceeds 5% in either CUDA path/run. Ordinary events
are noisier. All 15 compiled device functions have matching instruction/control
words and resource reports, ignoring only NVCC's translation-unit symbol hash.
This is not whole-library identity or a universal performance guarantee.

Retention loses to recomputation at 4096 rows: 654.734 versus 617.679 us, then
657.329 versus 612.621 us. The unchanged four-backend harness newly measures
256x4096 graph medians of 79.489 us eager, 9.078 us compiled PyTorch, 15.394 us
CUDA and 8.242 us Triton. Default routing was not retuned. CUDA still loses to
the compiler baseline. Historical results stay in [the BF16 study](bf16_rmsnorm.md).

BF16 unique payload is `2*(4*R*H+H)` bytes, 8,396,800 bytes at 256x4096.
Approximate useful intensity is `5/8 = 0.625` FLOP/byte. Dividing by 14.7 us
gives 571 logical GB/s, not measured DRAM bandwidth. Baseline rereads and repeated
weight accesses may hit cache. The selected new large CUDA profile reports
219.67 decimal DRAM GB/s, 44 registers/thread, 82.66% occupancy and zero spill
bytes. Static shared storage is 32 bytes; the total includes 1024 driver bytes.
The unchanged Triton capture gives 341.40 GB/s with a different sustained
reference. Diagnostic profile rates do not replace unprofiled graph latency.

```bash
make bf16-check PYTHON=python EXP=EXP-20261105-201
make rmsnorm-compare PYTHON=python EXP=EXP-20261105-202
make bf16-profile-preview PYTHON=python EXP=EXP-20261105-203
```

Choose fresh IDs/paths. The paired driver reuses the historical fresh-event
helper; setup/JIT are outside timing, and event/wall/graph samples stay separate.
The driver changed between October and November from 595.91.07 to 595.99.02;
clocks remain unlocked. No system configuration was changed by these runs.
Current validation passes 86 checks and 47 memcheck checks with zero errors;
each paired process adds 64 FP32 path/stream checks. See
[CUDA evidence](../results/cuda/README.md) for samples, source snapshots and
profile projections. Full binary captures stay local. FP16, mixed dtypes,
gradients, whole-model accuracy, clean-room parity and B200 are not validated.
