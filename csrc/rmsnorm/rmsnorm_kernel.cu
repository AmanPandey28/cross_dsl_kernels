#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <torch/extension.h>

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

  partial_sums[threadIdx.x] = thread_sum;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  const float inv_rms = rsqrtf(partial_sums[0] * inv_hidden + eps);

  for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int64_t idx = row_offset + col;
    const float r = residual_out[idx];
    y[idx] = r * inv_rms * weight[col];
  }
}

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

  partial_sums[threadIdx.x] = thread_sum;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  const float inv_rms = rsqrtf(partial_sums[0] * inv_hidden + eps);

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

bool is_aligned_16(const void* ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 0xF) == 0;
}

}  // namespace

std::vector<torch::Tensor> crossdsl_fused_residual_rmsnorm_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps) {
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

  if (use_vec4) {
    fused_residual_rmsnorm_f32x4_staged_kernel<<<
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
    fused_residual_rmsnorm_f32_staged_kernel<<<
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
