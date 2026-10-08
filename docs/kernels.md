# Kernel Design

The suite holds the workload contract fixed while changing thread ownership,
reuse, fusion, or reduction structure. A language change alone is not an
optimization. PyTorch references prioritize clear mathematics; benchmark paths
avoid setup and conversion work in the steady-state region.

## Programming Models

CUDA C++ gives explicit blocks, warps, lanes, pointer arithmetic, shared memory
and synchronization. It makes ownership visible, but the programmer must prove
tail handling, race freedom and reduction correctness. C++ bindings validate
metadata and launch on the current PyTorch CUDA stream.

Triton expresses each program as tensor-valued index and mask operations.
The compiler distributes those values over threads. Tile dimensions, strides,
reduction axes, warp count and dot precision are still essential performance
decisions. Inspect generated code and profiling rather than inferring hardware
instructions from a high-level operator name.

CuTe DSL is a Python-hosted compiled GPU DSL, not Python running on the GPU.
Layouts and tiled ownership describe how tensor coordinates map to execution
and storage. These kernels also use explicit thread/lane indexing and reductions.
The host converts PyTorch tensors through DLPack and passes the current stream;
prepared launches avoid including repeated wrapper work in graph measurements.
Some older paths use low-level compiler interfaces, so the recorded CuTe version
matters. CuTe DSL here is distinct from a tuned CUTLASS GEMM library baseline.

## Fused Residual RMSNorm

For each row, `r = x + residual`, `inv = rsqrt(mean(r*r) + eps)`,
and `y = r * inv * weight`. Both `y` and `r` are returned.
FP32 data and accumulation, positive finite epsilon, contiguous storage and
non-overlapping outputs are the supported contract.

| Implementation | Files | Progression |
|---|---|---|
| CUDA | [kernel](../csrc/rmsnorm/rmsnorm_kernel.cu), [binding](../csrc/rmsnorm/rmsnorm_ext.cpp) | Shared/warp reduction, then keep row fragments resident |
| Triton | [rmsnorm.py](../src/crossdsl_kernels/triton/rmsnorm.py) | Power-of-two masked row reduction; bounded warp-count candidate |
| CuTe | [rmsnorm.py](../src/crossdsl_kernels/cute/rmsnorm.py) | Staged reduction, then register-resident fragments |
| PyTorch | [reference](../src/crossdsl_kernels/references/rmsnorm.py) | Eager and compiled execution in the shared harness |

The row reduction requires all contributing lanes to participate correctly.
Masked hidden-size tails contribute zero to the sum but normalization divides
by the actual hidden dimension. Holding a row can eliminate repeated global
loads; larger live fragments increase register pressure. On small workloads,
launch and wrapper costs can dominate the memory saving.

## Decode Projection

`y[m,n] = sum_k x[m,k] * W[k,n] + bias[n]`. KN stores the weight as `[K,N]`;
NK stores `[N,K]` and computes the same mathematical projection. Inputs and
accumulation are FP32, with TF32 disabled in the measured comparison.

| Implementation | Files | Progression |
|---|---|---|
| CUDA | [kernel](../csrc/gemv/gemv_kernel.cu), [binding](../csrc/gemv/gemv_ext.cpp) | Per-output reduction to layout-aware four-row reuse; vendor baselines in the same extension |
| Triton | [baseline](../src/crossdsl_kernels/triton/gemv.py), [candidate](../src/crossdsl_kernels/triton/gemv_rows.py) | Masked tiles to row reuse, tiled dot and split-K finish |
| CuTe | [baseline](../src/crossdsl_kernels/cute/gemv.py), [candidate](../src/crossdsl_kernels/cute/gemv_rows.py) | Serial output chains to layout-aware row reuse |
| PyTorch | [reference](../src/crossdsl_kernels/references/gemv.py) | Eager matmul and compiled baseline |

At M1, ideal arithmetic intensity is roughly 0.5 FLOP/byte for large FP32
weights. Multiple input rows provide opportunities to reuse each weight.
KN lanes should generally span neighboring output columns; NK lanes can span
contiguous K positions. A tile good for one layout can be poor for the other.
Split-K creates partial outputs and a second reduction launch. This reduces
long dependency chains but adds scratch storage, extra traffic and a different
summation order. The selected Triton FP32 dot path uses IEEE precision; measured
profiles do not show Tensor Core execution. Bias is applied exactly once after
the final reduction.

## RoPE And Paged KV Append

Each selected Q/K pair is rotated using the position's sine/cosine values.
Unrotated head dimensions are copied. K and V are written into translated
physical cache slots; rotated Q is returned. Query and KV head counts may
differ (GQA). Both adjacent-pair and split-half rotation conventions and NHD/HND
cache layouts are covered by the catalog.

| Implementation | Files |
|---|---|
| CUDA | [kernel](../csrc/rope_kv/rope_kv_kernel.cu), [binding](../csrc/rope_kv/rope_kv_ext.cpp) |
| Triton | [two-stage](../src/crossdsl_kernels/triton/rope_kv.py), [fused](../src/crossdsl_kernels/triton/rope_kv_fused.py) |
| CuTe | [two-stage](../src/crossdsl_kernels/cute/rope_kv.py), [fused](../src/crossdsl_kernels/cute/rope_kv_fused.py) |
| PyTorch | [reference and vectorized baseline](../src/crossdsl_kernels/references/rope_kv.py) |

For position `p`, the logical page is `p // page_size`, offset is
`p % page_size`, and the page table maps `(sequence, logical_page)` to a physical
page. HND and NHD have different linear strides. The cache is stateful: validating
only returned Q is insufficient. Tests check full cache contents including
sentinel pages. Valid indices and unique destination slots are required; duplicate
writes are not resolved atomically. Expensive value-dependent index validation is
setup, not repeated inside the timed launch.

## W4A16 Projection

`y = x @ dequantized_weight.T + bias`, with `x` and output in FP16,
FP32 products/accumulation, and FP16 group scales. Packed weights are NK:
two signed INT4 values per byte, low nibble first, stored as `code = q + 8`.
Odd K pads the unused high nibble with code 8 (zero). Group sizes are
32, 64, 128 and 256; partial final groups are masked.

| Implementation | Files |
|---|---|
| CUDA | [kernel](../csrc/w4a16/w4a16_kernel.cu), [binding](../csrc/w4a16/w4a16_ext.cpp) |
| Triton | [one-row and four-row variants](../src/crossdsl_kernels/triton/w4a16.py) |
| CuTe | [one-row and four-row variants](../src/crossdsl_kernels/cute/w4a16.py) |
| PyTorch | [packing, dequantization, FP64 oracle](../src/crossdsl_kernels/quantization.py), [backend wrapper](../src/crossdsl_kernels/w4a16.py) |

Weights are dequantized with the specified FP16 rounding **before** use in the
FP32 accumulation. The independent oracle multiplies FP16 operands in FP64 and
returns FP16. This distinction prevents a silently different arithmetic contract.
Three execution baselines are measured: original dense FP16 weights,
pre-dequantized FP16 weights with the same quantized values, and eager
dequantization followed by matmul. The second is the fairest arithmetic-matched
library comparator; the first changes approximation semantics.

The setup quantizer rounds scales to FP16, rounds/clamps weights to `[-7,7]`,
and uses scale one for all-zero groups. The decoder supports the full `[-8,7]`
domain, verified separately. There is no calibration, learned zero point or
model-quality guarantee. For large even K and group 128, packed weights plus
scales consume about 25.78% of dense FP16 weight bytes (3.88x smaller), not
exactly 4x. Unpacking instructions, scale loads, FP16 conversions and lower
compute throughput can outweigh that saving as M grows. These are SIMT kernels,
not INT4 Tensor Core GEMMs.
