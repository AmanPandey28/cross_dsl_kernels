#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

template <int Rows>
__global__ void w4a16_warp_kernel(const half* x, const uint8_t* packed,
                                const half* scales, const half* bias, half* y,
                                int m, int k, int n, int group_size, bool has_bias) {
  const int lane = threadIdx.x % 32;
  const int col = blockIdx.x * 8 + threadIdx.x / 32;
  const int row = blockIdx.y * Rows;
  if (col >= n) return;  // Uniform within a warp, before any shuffle.
  const int pairs = (k + 1) / 2;
  const int groups = (k + group_size - 1) / group_size;
  float acc[Rows] = {};
  for (int pair = lane; pair < pairs; pair += 32) {
    const int kk = pair * 2;
    const int code = packed[col * pairs + pair];
    const float scale = __half2float(scales[col * groups + kk / group_size]);
    // The shared contract rounds dequantized weights to FP16 before the dot.
    const float w0 = __half2float(__float2half_rn(((code & 15) - 8) * scale));
    const float w1 = __half2float(__float2half_rn(((code >> 4) - 8) * scale));
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
      if (row + r < m) {
        acc[r] = fmaf(__half2float(x[(row + r) * k + kk]), w0, acc[r]);
        if (kk + 1 < k)
          acc[r] = fmaf(__half2float(x[(row + r) * k + kk + 1]), w1, acc[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < Rows; ++r) {
    for (int offset = 16; offset > 0; offset /= 2)
      acc[r] += __shfl_down_sync(0xffffffff, acc[r], offset);
    if (lane == 0 && row + r < m) {
      const float b = has_bias ? __half2float(bias[col]) : 0.0f;
      y[(row + r) * n + col] = __float2half_rn(acc[r] + b);
    }
  }
}

void w4a16_launch(torch::Tensor x, torch::Tensor packed, torch::Tensor scales,
                 torch::Tensor bias, torch::Tensor y, int group_size,
                 int rows_per_block, bool has_bias) {
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* properties = at::cuda::getDeviceProperties(x.get_device());
  TORCH_CHECK(properties->major == 12 && properties->minor == 0, "SM120 required");
  const int m = x.size(0), k = x.size(1), n = packed.size(0);
  const dim3 grid((n + 7) / 8, (m + rows_per_block - 1) / rows_per_block);
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  const auto* xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  const auto* sp = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const auto* bp = reinterpret_cast<const half*>(bias.data_ptr<at::Half>());
  auto* yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
  if (rows_per_block == 1)
    w4a16_warp_kernel<1><<<grid, 256, 0, stream>>>(xp, packed.data_ptr<uint8_t>(), sp, bp, yp, m, k, n, group_size, has_bias);
  else
    w4a16_warp_kernel<4><<<grid, 256, 0, stream>>>(xp, packed.data_ptr<uint8_t>(), sp, bp, yp, m, k, n, group_size, has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
