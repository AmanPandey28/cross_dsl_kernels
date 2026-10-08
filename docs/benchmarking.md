# Setup And Measurement

## Environment

Use a source checkout on Linux with Python 3.11, a GPU-visible NVIDIA driver,
and an SM120-capable CUDA toolkit. The native C++ extensions need `nvcc`, a
compatible host C++ compiler and Ninja. This is an editable/source-checkout
workflow: a Python-only wheel does not bundle the native extension sources.

The saved measurements used:

| Component | Recorded version |
|---|---|
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU, compute capability 12.0 |
| Python | 3.11.14 |
| PyTorch | 2.10.0+cu130 |
| Triton | 3.6.0 |
| CuTe DSL (`nvidia-cutlass-dsl`) | 4.5.2 |
| CUDA bindings | 13.0.3 |
| CUDA toolkit | 13.1 |
| NVIDIA driver | 595.91.07 |

An isolated environment avoids changing system packages or a working Conda
environment. A setup recipe for the recorded Python dependencies is:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install nvidia-cutlass-dsl==4.5.2 cuda-bindings==13.0.3 ninja matplotlib
python -m pip install -e . --no-deps
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"
nvcc --version
nvidia-smi
make test evidence-check PYTHON=python
```

These installation commands require package-index access and an already suitable
host driver/toolkit. They do not install or replace either one. PyTorch's CUDA
runtime and the toolkit used to compile extensions are different components;
`nvidia-smi` reports the driver's supported CUDA level, not the active PyTorch
runtime. If a restricted shell cannot access device nodes, check GPU visibility
in the normal host environment before diagnosing a missing GPU.

## Fixed Catalogs And Backends

[comparison.json](../benchmarks/shapes/comparison.json) has seven RMSNorm,
thirteen FP32 projection and five RoPE/cache cases. Small/tail and larger decode
shapes are intentional. [w4a16.json](../benchmarks/shapes/w4a16.json) has sixteen
quantized cases including group sizes, odd K/N tails and M1/M8/M16.

The FP32 names `cuda`, `triton`, `cute` select retained earlier custom paths;
`cuda_fast`, `triton_fast`, `cute_fast` select optimization candidates.
"Fast" is a variant name, not a promise for every shape.
`pytorch` and `torch_compile` are execution baselines. `cublas` and `cublaslt`
only apply to projection; irrelevant combinations are skipped.
W4A16 has three PyTorch paths plus one-row/four-row variants for each language.
`torch_predequant` uses the same dequantized weights as the custom kernels.

```bash
# Bounded correctness-first runs:
make smoke w4a16-smoke PYTHON=python EXP=EXP-20261008-101

# Full local measurement, using a NEW ID:
make benchmark w4a16-benchmark PYTHON=python EXP=EXP-20261008-102

# A selected repeat with a new seed and process:
python benchmarks/compare.py --experiment EXP-20261008-103 \
  --output results/runs/EXP-20261008-103/fp32 \
  --cases hidden_m16,mlp_up_m16 \
  --backends pytorch,torch_compile,cuda,triton,cute,cublas,cublaslt,cuda_fast,triton_fast,cute_fast \
  --warmup 5 --repeat 30 --graph-calls 30 --seed 20261009
```

Do not benchmark concurrently with another GPU workload. A cooperative
`/tmp/crossdsl-gpu.lock` prevents these drivers from measuring simultaneously;
it cannot stop unrelated GPU programs. Each run records package versions,
compiler flags, source SHA256, seed, GPU telemetry and command. New output paths
are mandatory. The recorded suite does not fix clocks, change power limits or
run an open-ended tuning search.

## Correctness Before Timing

The shared FP32 driver uses FP64 expected arithmetic for RMSNorm/projection and
a separate CPU paged-cache reference for RoPE. Gates check finite values, maximum
absolute error and relative L2. GEMV's fixed absolute bound scales as
`2e-4 * max(1, sqrt(K/1024))`, with relative L2 at most `1e-5`.
Near-zero elementwise relative errors are diagnostic, not the acceptance gate.

W4A16 uses an independent FP64 CPU dot of FP16 activations and dequantized FP16
weights, returning FP16. Its bound is `abs(err) <= 0.002 + 0.002*abs(reference)`
and relative L2 at most `0.001`. Weight/output error against the original dense
weights is reported separately as **quantization quality**, not a kernel failure.
The original dense baseline has its own dense expected result.

Graph outputs are poisoned before replay; RoPE caches are reset too. This catches
a no-op replay that would otherwise reuse a correct output from a previous call.
Contract checks cover aliasing, unsupported shapes and invalid scalar metadata.
Value-dependent page indices must be checked before execution. A bounded
Compute Sanitizer check is not a proof for all inputs or race freedom.

## Three Timing Views

1. CUDA events around an ordinary launch measure an event interval, which can
   include launch/feed gaps. They are not identical to an isolated kernel duration.
2. Synchronized wall time includes Python/host overhead and waiting for completion.
3. CUDA Graph batches divide event/wall batch times by the captured call count.
   This reduces host-feed overhead and measures fixed-buffer steady state.

Compilation, conversion, allocation/setup and first-call latency are recorded
separately. Prepared CuTe launches exclude repeated DLPack/JIT wrapper work.
cuBLASLt graph replay excludes descriptor and heuristic setup; workspace is
bounded and the first successful permitted heuristic is used, not exhaustive
autotuning. Profiling runs disable graph timing and are diagnostic.

Keep all samples and show p10/p90, not just the fastest call. Randomized backend
order reduces one ordering bias but does not control mobile thermals. A second
process/seed is a useful repeat, not a confidence interval or independent GPU
replication. Roofline estimates must count scales, extra partial buffers and
repeated reads; "logical GB/s" is not a hardware DRAM counter or peak bandwidth.

## Supporting Commands

```bash
python scripts/verify_optimized.py --experiment EXP-20261008-104 \
  --output results/runs/EXP-20261008-104/fp32-contracts.json
python scripts/verify_w4a16.py --experiment EXP-20261008-105 \
  --output results/runs/EXP-20261008-105/w4a16-contracts
python scripts/summarize_comparison.py --result results/fp32/full.json \
  --output results/analysis/EXP-20261008-106
```

The summary command requires Matplotlib. For memory checks, prefix the two
verification commands with `compute-sanitizer --tool memcheck`. These checks
must run in a GPU-visible environment. CPU tests and saved-evidence validation
can run without a GPU. Container and remote B200 reproduction are not claimed
for this release.
