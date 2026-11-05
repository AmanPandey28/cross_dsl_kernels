#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int BM = 8;
constexpr int BN = 32;
constexpr int BK = 32;
constexpr int Threads = BM * BN;

template <bool WeightIsNK>
__global__ void gemm_tiled_kernel(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int64_t m, int64_t k, int64_t n) {
  __shared__ float x_tile[BM][BK];
  __shared__ float weight_tile[BK][BN];
  const int local_row = threadIdx.x / BN;
  const int local_col = threadIdx.x % BN;
  const int64_t row_start = static_cast<int64_t>(blockIdx.y) * BM;
  const int64_t col_start = static_cast<int64_t>(blockIdx.x) * BN;
  const int64_t out_row = row_start + local_row;
  const int64_t out_col = col_start + local_col;
  float accumulator = 0.0f;

  for (int64_t k_base = 0; k_base < k; k_base += BK) {
    const int64_t x_k = k_base + local_col;
    x_tile[local_row][local_col] = out_row < m && x_k < k ? x[out_row * k + x_k] : 0.0f;
    // Four elements/thread cover the entire 32x32 logical weight tile.
    for (int index = threadIdx.x; index < BK * BN; index += Threads) {
      if constexpr (WeightIsNK) {
        const int col = index / BK;
        const int kk = index % BK;
        // Lanes traverse K in global NK storage, then transpose into shared [K,N].
        weight_tile[kk][col] = col_start + col < n && k_base + kk < k
            ? weight[(col_start + col) * k + k_base + kk] : 0.0f;
      } else {
        const int kk = index / BN;
        const int col = index % BN;
        weight_tile[kk][col] = col_start + col < n && k_base + kk < k
            ? weight[(k_base + kk) * n + col_start + col] : 0.0f;
      }
    }
    // Every operand, including zero-filled tails, must be visible before compute.
    __syncthreads();
    float tile_sum = 0.0f;
    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      tile_sum += x_tile[local_row][kk] * weight_tile[kk][local_col];
    }
    // Tile-local summation avoids one long K-element FP32 accumulation chain.
    accumulator += tile_sum;
    // No warp may overwrite shared operands while another still reads this tile.
    __syncthreads();
  }
  if (out_row < m && out_col < n) {
    output[out_row * n + out_col] = accumulator + (bias != nullptr ? bias[out_col] : 0.0f);
  }
}
}  // namespace

void launch_gemm_tiled(const float* x, const float* weight, const float* bias,
                       float* output, int64_t m, int64_t k, int64_t n,
                       bool weight_is_nk, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>((n - 1) / BN + 1),
                  static_cast<unsigned int>((m - 1) / BM + 1));
  if (weight_is_nk) {
    gemm_tiled_kernel<true><<<grid, Threads, 0, stream>>>(x, weight, bias, output, m, k, n);
  } else {
    gemm_tiled_kernel<false><<<grid, Threads, 0, stream>>>(x, weight, bias, output, m, k, n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
