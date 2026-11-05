#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int WarpSize = 32;
constexpr int Warps = 8;
constexpr int Threads = WarpSize * Warps;
constexpr int Rows = 4;

__device__ float warp_sum(float value) {
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <bool WeightIsNK>
__global__ void gemv_warp_kernel(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int64_t total_outputs, int64_t k, int64_t n) {
  const int lane = threadIdx.x & (WarpSize - 1);
  const int warp_id = threadIdx.x / WarpSize;
  const int64_t output_index = static_cast<int64_t>(blockIdx.x) * Warps + warp_id;
  // Uniform for a whole warp; no CTA barrier is used in this kernel.
  if (output_index >= total_outputs) return;
  const int64_t out_row = output_index / n;
  const int64_t out_col = output_index % n;
  float partial_sum = 0.0f;
  for (int64_t k_index = lane; k_index < k; k_index += WarpSize) {
    float w;
    if constexpr (WeightIsNK) {
      w = weight[out_col * k + k_index];
    } else {
      w = weight[k_index * n + out_col];
    }
    partial_sum += x[out_row * k + k_index] * w;
  }
  const float sum = warp_sum(partial_sum);
  if (lane == 0) {
    output[output_index] = sum + (bias != nullptr ? bias[out_col] : 0.0f);
  }
}

__global__ void gemv_rows_kn_kernel(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int64_t m, int64_t k, int64_t n) {
  const int lane = threadIdx.x & (WarpSize - 1);
  const int warp_id = threadIdx.x / WarpSize;
  const int64_t out_col = static_cast<int64_t>(blockIdx.x) * WarpSize + lane;
  const int64_t row_start = static_cast<int64_t>(blockIdx.y) * Rows;
  float accumulators[Rows] = {};
  // Neighboring lanes load neighboring KN columns; warps partition K eight ways.
  for (int64_t k_index = warp_id; k_index < k; k_index += Warps) {
    const float w = out_col < n ? weight[k_index * n + out_col] : 0.0f;
    #pragma unroll
    for (int row = 0; row < Rows; ++row) {
      if (row_start + row < m) {
        accumulators[row] += x[(row_start + row) * k + k_index] * w;
      }
    }
  }
  __shared__ float partial_sums[Rows][Warps][WarpSize];
  #pragma unroll
  for (int row = 0; row < Rows; ++row) {
    partial_sums[row][warp_id][lane] = accumulators[row];
  }
  // Warp 0 must see partial sums from all eight warps, including masked columns.
  __syncthreads();
  if (warp_id == 0 && out_col < n) {
    #pragma unroll
    for (int row = 0; row < Rows; ++row) {
      float sum = 0.0f;
      #pragma unroll
      for (int source_warp = 0; source_warp < Warps; ++source_warp) {
        sum += partial_sums[row][source_warp][lane];
      }
      if (row_start + row < m) {
        output[(row_start + row) * n + out_col] = sum + (bias != nullptr ? bias[out_col] : 0.0f);
      }
    }
  }
}

__global__ void gemv_rows_nk_kernel(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int64_t m, int64_t k, int64_t n) {
  const int lane = threadIdx.x & (WarpSize - 1);
  const int warp_id = threadIdx.x / WarpSize;
  const int64_t out_col = static_cast<int64_t>(blockIdx.x) * Warps + warp_id;
  const int64_t row_start = static_cast<int64_t>(blockIdx.y) * Rows;
  float accumulators[Rows] = {};
  // A warp owns one NK column, so lanes load contiguous reduction elements.
  for (int64_t k_index = lane; k_index < k; k_index += WarpSize) {
    const float w = out_col < n ? weight[out_col * k + k_index] : 0.0f;
    #pragma unroll
    for (int row = 0; row < Rows; ++row) {
      if (row_start + row < m) {
        accumulators[row] += x[(row_start + row) * k + k_index] * w;
      }
    }
  }
  #pragma unroll
  for (int row = 0; row < Rows; ++row) {
    const float sum = warp_sum(accumulators[row]);
    if (lane == 0 && out_col < n && row_start + row < m) {
      output[(row_start + row) * n + out_col] = sum + (bias != nullptr ? bias[out_col] : 0.0f);
    }
  }
}
}  // namespace

void launch_gemv_warp(const float* x, const float* weight, const float* bias,
                            float* output, int64_t m, int64_t k, int64_t n,
                            bool weight_is_nk, cudaStream_t stream) {
  const unsigned int blocks = static_cast<unsigned int>((m * n - 1) / Warps + 1);
  if (weight_is_nk) {
    gemv_warp_kernel<true><<<blocks, Threads, 0, stream>>>(x, weight, bias, output, m * n, k, n);
  } else {
    gemv_warp_kernel<false><<<blocks, Threads, 0, stream>>>(x, weight, bias, output, m * n, k, n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gemv_rows(const float* x, const float* weight, const float* bias,
                            float* output, int64_t m, int64_t k, int64_t n,
                            bool weight_is_nk, cudaStream_t stream) {
  const int columns = weight_is_nk ? Warps : WarpSize;
  const dim3 grid(static_cast<unsigned int>((n - 1) / columns + 1),
                  static_cast<unsigned int>((m - 1) / Rows + 1));
  if (weight_is_nk) {
    gemv_rows_nk_kernel<<<grid, Threads, 0, stream>>>(x, weight, bias, output, m, k, n);
  } else {
    gemv_rows_kn_kernel<<<grid, Threads, 0, stream>>>(x, weight, bias, output, m, k, n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
