\
# Docker Development and Reproduction Path

Docker is required for clean-room reproduction and future Modal packaging, but native execution remains the preferred local edit/profile loop.

## Responsibilities

- **Native host:** fastest CUDA/Triton/CuTe development; first Torch Profiler, Nsight Systems, and Nsight Compute analysis.
- **Docker:** dependency pinning, clean build/test/report checks, bounded parity experiments, and the future Modal image contract.
- **Modal:** controlled SM100/B200 experiments only after local release gates.

## Build

```bash
./docker/build.sh
```

The initial base candidate is `nvidia/cuda:13.2.1-devel-ubuntu24.04`. Verify the tag, digest, host-driver compatibility, Python/PyTorch/CuTe requirements, and final package lock before treating it as released. Do not install an NVIDIA kernel driver inside the image.

## Run

```bash
./docker/run.sh bash
./docker/run.sh make report
./docker/run.sh python scripts/doctor.py
```

`docker/run.sh` asks for `--gpus all` only when the host exposes `nvidia-smi`. GPU pass-through additionally requires a working Docker engine and NVIDIA Container Toolkit on the host. The agent may diagnose this, but may not use `sudo` or alter daemon/runtime configuration without explicit approval.

## Current local evidence

`EXP-20260703-002` confirmed Docker GPU pass-through using the already-local image:

```bash
docker run --pull=never --rm --gpus all --entrypoint nvidia-smi \
  nvidia/cuda:13.0.0-base-ubuntu24.04
```

The project candidate image tag `nvidia/cuda:13.2.1-devel-ubuntu24.04` was not present locally and was not pulled. Building or pulling that image remains a separate reproduction step.

## Profiling

Prefer host-side profiling first. Container profiling is a separate validation because it can depend on host kernel/security policy and performance-counter permissions.

- Run `nsys status --environment` inside the container before relying on CPU sampling.
- Do not add `--privileged` or `SYS_ADMIN` silently. Record and obtain approval for any privilege change.
- Nsight Compute hardware counters may remain unavailable even when CUDA kernels execute. Record `ERR_NVGPUCTRPERM` as a host-policy limitation rather than weakening security automatically.
- Save profiler CLI/UI versions; open reports with a compatible or newer GUI.

## Release evidence

Record the Dockerfile commit, base tag and digest, final image ID/digest, build command, build log, package lock, smoke-test results, GPU visibility, and native-versus-container parity experiment IDs.
