#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <bool Fast>
__device__ float block_sum(float value, float* partial_sums) {
  if constexpr (Fast) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int delta = 16; delta > 0; delta >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, delta);
    }
    if (lane == 0) partial_sums[warp] = value;
    __syncthreads();
    if (warp == 0) {
      value = lane < blockDim.x / 32 ? partial_sums[lane] : 0.0f;
      for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, delta);
      }
      if (lane == 0) partial_sums[0] = value;
    }
    __syncthreads();
  } else {
    partial_sums[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
      }
      __syncthreads();
    }
  }
  return partial_sums[0];
}

template <bool Fast>
__global__ void fused_residual_rmsnorm_f32_staged_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ y,
    float* __restrict__ residual_out,
    int64_t hidden,
    float eps) {
  extern __shared__ float partial_sums[];
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t row_offset = row * hidden;

  float thread_sum = 0.0f;
  for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int64_t idx = row_offset + col;
    const float r = x[idx] + residual[idx];
    residual_out[idx] = r;
    thread_sum += r * r;
  }

  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  const float inv_rms = rsqrtf(block_sum<Fast>(thread_sum, partial_sums) * inv_hidden + eps);

  for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int64_t idx = row_offset + col;
    const float r = residual_out[idx];
    y[idx] = r * inv_rms * weight[col];
  }
}

template <bool Fast>
__global__ void fused_residual_rmsnorm_f32x4_staged_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ y,
    float* __restrict__ residual_out,
    int64_t hidden,
    float eps) {
  extern __shared__ float partial_sums[];
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t vec_hidden = hidden >> 2;
  const int64_t vec_row_offset = row * vec_hidden;

  const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
  const float4* __restrict__ residual4 = reinterpret_cast<const float4*>(residual);
  const float4* __restrict__ weight4 = reinterpret_cast<const float4*>(weight);
  float4* __restrict__ y4 = reinterpret_cast<float4*>(y);
  float4* __restrict__ residual_out4 = reinterpret_cast<float4*>(residual_out);

  float thread_sum = 0.0f;
  for (int64_t vec_col = threadIdx.x; vec_col < vec_hidden; vec_col += blockDim.x) {
    const int64_t idx = vec_row_offset + vec_col;
    const float4 xv = x4[idx];
    const float4 rv = residual4[idx];
    const float r0 = xv.x + rv.x;
    const float r1 = xv.y + rv.y;
    const float r2 = xv.z + rv.z;
    const float r3 = xv.w + rv.w;
    residual_out4[idx] = make_float4(r0, r1, r2, r3);
    thread_sum += r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3;
  }

  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  const float inv_rms = rsqrtf(block_sum<Fast>(thread_sum, partial_sums) * inv_hidden + eps);

  for (int64_t vec_col = threadIdx.x; vec_col < vec_hidden; vec_col += blockDim.x) {
    const int64_t idx = vec_row_offset + vec_col;
    const float4 r = residual_out4[idx];
    const float4 wv = weight4[vec_col];
    y4[idx] = make_float4(
        r.x * inv_rms * wv.x,
        r.y * inv_rms * wv.y,
        r.z * inv_rms * wv.z,
        r.w * inv_rms * wv.w);
  }
}

namespace {

template <int Hidden>
__global__ void rmsnorm_register_kernel(
    const float4* x, const float4* residual, const float4* weight,
    float4* y, float4* residual_out, float eps) {
  constexpr int Vectors = Hidden / (256 * 4);
  __shared__ float sums[8];
  float4 values[Vectors];
  float total = 0.0f;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * (Hidden / 4);
  #pragma unroll
  for (int i = 0; i < Vectors; ++i) {
    const int col = threadIdx.x + i * 256;
    const float4 a = x[base + col], b = residual[base + col];
    values[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    const float4 r = values[i];
    total += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
    residual_out[base + col] = r;
  }
  const float inv = rsqrtf(block_sum<true>(total, sums) / Hidden + eps);
  #pragma unroll
  for (int i = 0; i < Vectors; ++i) {
    const int col = threadIdx.x + i * 256;
    const float4 r = values[i], w = weight[col];
    y[base + col] = make_float4(r.x * inv * w.x, r.y * inv * w.y,
                               r.z * inv * w.z, r.w * inv * w.w);
  }
}

template <int Hidden>
void launch_register_rmsnorm(const float* x, const float* residual, const float* weight,
                            float* y, float* residual_out, int64_t rows, float eps) {
  rmsnorm_register_kernel<Hidden><<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const float4*>(x), reinterpret_cast<const float4*>(residual),
      reinterpret_cast<const float4*>(weight), reinterpret_cast<float4*>(y),
      reinterpret_cast<float4*>(residual_out), eps);
}

bool is_aligned_16(const void* ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 0xF) == 0;
}

}  // namespace

template <bool Fast>
std::vector<torch::Tensor> launch_rmsnorm(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty_like(x);
  auto residual_out = torch::empty_like(residual);

  const int64_t rows = x.size(0);
  const int64_t hidden = x.size(1);
  constexpr int threads = 256;
  const dim3 blocks(static_cast<unsigned int>(rows));
  const size_t shared_bytes = threads * sizeof(float);

  const float* x_ptr = x.data_ptr<float>();
  const float* residual_ptr = residual.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  float* y_ptr = y.data_ptr<float>();
  float* residual_out_ptr = residual_out.data_ptr<float>();
  const bool use_vec4 =
      hidden % 4 == 0 && is_aligned_16(x_ptr) && is_aligned_16(residual_ptr) &&
      is_aligned_16(weight_ptr) && is_aligned_16(y_ptr) && is_aligned_16(residual_out_ptr);

  if constexpr (Fast) {
    if (use_vec4 && (hidden == 1024 || hidden == 2048 || hidden == 4096 || hidden == 8192)) {
      switch (hidden) {
        case 1024: launch_register_rmsnorm<1024>(x_ptr, residual_ptr, weight_ptr, y_ptr, residual_out_ptr, rows, eps); break;
        case 2048: launch_register_rmsnorm<2048>(x_ptr, residual_ptr, weight_ptr, y_ptr, residual_out_ptr, rows, eps); break;
        case 4096: launch_register_rmsnorm<4096>(x_ptr, residual_ptr, weight_ptr, y_ptr, residual_out_ptr, rows, eps); break;
        case 8192: launch_register_rmsnorm<8192>(x_ptr, residual_ptr, weight_ptr, y_ptr, residual_out_ptr, rows, eps); break;
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return {y, residual_out};
    }
  }

  if (use_vec4) {
    fused_residual_rmsnorm_f32x4_staged_kernel<Fast><<<
        blocks,
        threads,
        shared_bytes,
        at::cuda::getCurrentCUDAStream()>>>(
        x_ptr,
        residual_ptr,
        weight_ptr,
        y_ptr,
        residual_out_ptr,
        hidden,
        static_cast<float>(eps));
  } else {
    fused_residual_rmsnorm_f32_staged_kernel<Fast><<<
        blocks,
        threads,
        shared_bytes,
        at::cuda::getCurrentCUDAStream()>>>(
        x_ptr,
        residual_ptr,
        weight_ptr,
        y_ptr,
        residual_out_ptr,
        hidden,
        static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, residual_out};
}

std::vector<torch::Tensor> crossdsl_fused_residual_rmsnorm_cuda(
    torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps, bool fast) {
  return fast ? launch_rmsnorm<true>(x, residual, weight, eps)
              : launch_rmsnorm<false>(x, residual, weight, eps);
}
