# Cross-DSL GPU Kernel Engineering: Transformer Inference Primitives

High-performance GPU implementations of three transformer-inference operators
across four DSLs: PyTorch (reference), CUDA C++, Triton, and CuTe DSL, plus
cuBLAS/cuBLASLt vendor baselines. Target: NVIDIA RTX 5050 Laptop GPU (Blackwell
SM120, 8GB).

## Operators

| Operator | Description | Shape Regime |
|---|---|---|
| Fused Residual RMSNorm | `r = x + residual; y = r * rsqrt(mean(r²)+eps) * weight` | Dense rows × 8–4096 |
| Decode GEMV / Small-M GEMM | `y = x @ W.T + b` with M ∈ [1,16], K,N ∈ [1024,11008] | Autoregressive decode |
| RoPE + GQA Paged-KV Append | Rotate Q/K, write K/V into paged cache with page-table translation | Attention prefill/decode |

## Backends

| Operator | PyTorch Ref | CUDA C++ | Triton | CuTe DSL | cuBLAS | cuBLASLt |
|---|---|---|---|---|---|---|
| RMSNorm | ✓ | ✓ | ✓ | ✓ | — | — |
| GEMV/GEMM | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| RoPE+KV | ✓ | ✓ | ✓ | ✓ | — | — |

✓ = implemented, GPU-tested, benchmarked. All backends pass correctness checks at
FP32 precision with documented tolerance policies.

## Performance Highlights (RTX 5050 SM120)

### RMSNorm
- CUDA C++ staged f32x4 kernel: 92.86% DRAM throughput at 256×4096 (NCU verified)
- Triton one-program-per-row: 1.72–1.86× vs eager PyTorch

### Decode GEMV
- CUDA warp-per-output (v1): competitive with cuBLAS for M=1 at large K
- cuBLASLt heuristics: bounded workspace (≤4 MiB), automatic layout selection
- Triton column-tile tuned: best overall for mixed M/K/N

### RoPE + GQA Paged-KV Append
- CUDA two-kernel pipeline: 33–247× vs PyTorch eager (Nsight Systems verified)
- HND and NHD cache layouts supported; interleaved and split-half RoPE conventions

### CuTe DSL
- RMSNorm GPU-tested and benchmarked (shared-memory tree reduction)
- GEMV one-thread-per-output (serial accumulation; passes at rel_l2 < 1.3e-6)
- RoPE+KV GPU-tested with separate NHD/HND kernel variants
- CUDA Graphs stream capture: **1,800× launch-overhead reduction** for RMSNorm

## Project Structure

```
├── src/crossdsl_kernels/          # Python package
│   ├── api.py                     # Public API
│   ├── contracts.py               # Type contracts and validation
│   ├── tolerances.py              # FP32 tolerance policies
│   ├── references/                # PyTorch eager reference implementations
│   ├── triton/                    # Triton kernels
│   └── cute/                      # CuTe DSL kernels
├── csrc/                          # CUDA C++ kernel sources
│   ├── rmsnorm/                   # staged f32x4 RMSNorm
│   ├── gemv/                      # decode GEMV v0 (block) + v1 (warp)
│   └── rope_kv/                   # RoPE + paged-KV append
├── tests/unit/                    # pytest suite
├── benchmarks/                    # CUDA-event-timed benchmarks
├── scripts/                       # Profiler harnesses (Nsight, Torch Profiler)
└── docker/                        # Reproducibility container
```

## Getting Started

### Prerequisites
- NVIDIA GPU with compute capability ≥ 7.0 (tested on SM120)
- CUDA Toolkit 13.0+
- Python 3.11+, PyTorch 2.10+, Triton 3.4+, nvidia-cutlass-dsl 4.5.2
- Conda environment recommended

### Build CUDA Extensions

```bash
cd csrc
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Run Tests

```bash
PYTHONPATH=src python -m pytest tests/unit/ -v
```

### Benchmark

```bash
PYTHONPATH=src python benchmarks/rmsnorm_benchmark.py --warmup 10 --repeat 100 --json results.json
```

### Profile

```bash
# Nsight Systems
nsys profile --trace=cuda,nvtx -o nsys_report \
  python scripts/profile_kernels.py --profiler none --workload all --iterations 10

# Torch Profiler
python scripts/profile_kernels.py --profiler torch --workload all --trace-dir traces/
```

## Key Design Decisions

1. **FP32 throughout**: All computations in float32 for maximum numerical accuracy and
   debugability. FP16/BF16 support deferred.
2. **Always-bias-safe**: GEMV contracts handle bias=None by passing a zero-filled buffer
   rather than branching in compute.
3. **Paged attention**: KV cache uses page-table translation with configurable page size,
   matching production inference engines.
4. **Back of the envelope first**: Every optimization was modeled analytically before
   writing a line of GPU code.
5. **Honest negative results**: cuBLASLt regression, CuTe GEMV serial accumulation error,
   and overhead-dominated latencies are all documented, not hidden.

## License

MIT
