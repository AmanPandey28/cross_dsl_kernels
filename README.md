# Cross-DSL Transformer Kernels

Four transformer workloads, implemented in **CUDA C++, Triton, and CuTe DSL**,
with PyTorch baselines and reproducible correctness, timing, and profiling.
The study asks which execution and memory-ownership choices improve each workload
on an **RTX 5050 Laptop GPU (SM120, 8 GB, observed 35 W power limit)**.

This is a forward-only kernel engineering study, not an inference server or a
claim that one language is universally faster. Baseline and improved custom
implementations are retained so the optimization steps can be inspected.

## Workloads

| Workload | What is computed | Main engineering question |
|---|---|---|
| Fused residual RMSNorm | Residual addition, row normalization and scale; returns normalized output and residual | Can row residency reduce reloads without excessive registers? |
| Decode GEMV / small-M GEMM | FP32 projection with optional bias, KN or NK weights, M=1..16 | How do layout, row reuse and split-K change the bottleneck? |
| RoPE + paged KV append | Rotary Q/K transform and K/V cache writes, GQA, NHD/HND layouts | Can fusion reduce launches while keeping page translation and mutation correct? |
| W4A16 projection | Packed group-scaled INT4 weights, FP16 activations/output, FP32 accumulation | When does compressed weight traffic outweigh unpacking and SIMT arithmetic? |

Every workload has all three custom implementations. PyTorch supplies numerical
references and execution baselines; FP32 projection also has bounded cuBLAS and
cuBLASLt baselines. W4A16 uses an explicit research packing format, not a drop-in
AWQ/GPTQ checkpoint format or an NVFP4 Tensor Core kernel.

## Results At A Glance

Saved full sweeps pass **226/226 FP32 backend-case rows** and **144/144 W4A16
rows**. Fresh-process repeats pass **72/72** and **36/36**, respectively.
Each benchmark row checks direct execution and poisoned-output graph replay.

| Optimization | Fresh-process observation | What it means |
|---|---|---|
| Triton row reuse + split-K FP32 projection | 15.31x hidden M16; 16.27x MLP-up M16 | Versus the earlier **custom Triton** kernels, not the best library |
| CuTe register-resident RMSNorm | 1.56x on 4096x4096 | Versus the earlier custom CuTe path; large mobile-clock variability remains |
| Fused RoPE/KV append | 1.67x CUDA, 1.45x Triton, 2.13x CuTe | Versus each language's two-stage custom path on the HND partial-rotation case |
| W4A16 M1 projection | CuTe hidden M1: 56.79 us; dense FP16: 123.20 us | Approximate weights; not an equal-model-quality or optimized-W4A16-library comparison |

Times above are **CUDA Graph per-call medians**, not end-to-end model latency.
Ordinary event timings, synchronized wall time, raw samples, numerical errors,
and losing variants are included in [results/](results/README.md).
At M16, every measured custom W4A16 variant loses to dense FP16. Lowering RMSNorm
warp count and tiling NK M1 projection also regress in important cases.
Read the [full results and limitations](docs/results.md) before quoting a number.

## Quick Start

Use Python 3.11 and a CUDA-capable PyTorch environment. The recorded environment
was Torch 2.10.0+cu130, Triton 3.6.0, CuTe DSL 4.5.2, cuda-bindings 13.0.3,
nvcc 13.1, and driver 595.91.07. CUDA C++ kernels are JIT-built from this checkout
using `nvcc`, a C++ compiler, and Ninja. No driver or system CUDA installation is
changed by this project.

```bash
# In an existing environment with the recorded dependencies:
python -m pip install -e . --no-deps
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
make test PYTHON=python
make evidence-check PYTHON=python
make list PYTHON=python
make smoke w4a16-smoke PYTHON=python EXP=EXP-20261008-101
```

For a new isolated environment, see [setup and measurement](docs/benchmarking.md).
Use a fresh experiment ID/output path for each collection. Runs refuse to
overwrite results and cooperate through a single-GPU lock. First-call compilation
can take minutes; compilation time is separate from steady-state measurements.

## Repository Map

```text
csrc/                       CUDA kernels and PyTorch C++ bindings, by workload
src/crossdsl_kernels/
  references/               PyTorch reference math and paged address helpers
  triton/                   Baseline and candidate Triton kernels
  cute/                     Baseline and candidate CuTe DSL kernels
  quantization.py           INT4 format, pack/unpack, numerical contract
  w4a16.py                  W4A16 backend selection
benchmarks/                 Shared inputs, oracles, timing, fixed shape catalogs
scripts/                    Profiling, regression checks and evidence validation
tests/unit/                 CPU checks for contracts, math and measurement tools
docs/                       Design, measurement, profiler interpretation and results
results/                    Portable numerical evidence with SHA256 provenance
```

Start with [kernel design](docs/kernels.md), then follow
[benchmarking](docs/benchmarking.md) and [profiling](docs/profiling.md).
The [lessons and next steps](docs/results.md#lessons-and-next-steps) explain
what succeeded, what failed, and what remains unproven.

Only local SM120 execution is validated. B200/SM100 and B300 are not tested here;
Blackwell-family branding does not make these launch configurations portable.
