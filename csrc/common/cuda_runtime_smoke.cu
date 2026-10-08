#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(expr)                                                     \
  do {                                                                       \
    cudaError_t status = (expr);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "%s failed: %s\n", #expr, cudaGetErrorString(status)); \
      return 1;                                                              \
    }                                                                        \
  } while (0)

__global__ void add_kernel(const float* x, const float* y, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = x[idx] + y[idx];
  }
}

int main() {
  int driver_version = 0;
  int runtime_version = 0;
  CHECK_CUDA(cudaDriverGetVersion(&driver_version));
  CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));

  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  std::printf("cudaDriverGetVersion=%d\n", driver_version);
  std::printf("cudaRuntimeGetVersion=%d\n", runtime_version);
  std::printf("cudaGetDeviceCount=%d\n", device_count);
  if (device_count <= 0) {
    return 2;
  }

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("device0=%s\n", prop.name);
  std::printf("capability=%d.%d\n", prop.major, prop.minor);
  std::printf("totalGlobalMem=%zu\n", static_cast<size_t>(prop.totalGlobalMem));

  constexpr int n = 4096;
  constexpr int bytes = n * static_cast<int>(sizeof(float));
  float* h_x = static_cast<float*>(std::malloc(bytes));
  float* h_y = static_cast<float*>(std::malloc(bytes));
  float* h_out = static_cast<float*>(std::malloc(bytes));
  if (!h_x || !h_y || !h_out) {
    std::fprintf(stderr, "host allocation failed\n");
    return 3;
  }
  for (int i = 0; i < n; ++i) {
    h_x[i] = static_cast<float>(i);
    h_y[i] = 3.0f;
  }

  float* d_x = nullptr;
  float* d_y = nullptr;
  float* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x, bytes));
  CHECK_CUDA(cudaMalloc(&d_y, bytes));
  CHECK_CUDA(cudaMalloc(&d_out, bytes));
  CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

  add_kernel<<<(n + 255) / 256, 256>>>(d_x, d_y, d_out, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

  float max_abs = 0.0f;
  for (int i = 0; i < n; ++i) {
    max_abs = std::fmax(max_abs, std::fabs(h_out[i] - (h_x[i] + h_y[i])));
  }
  std::printf("max_abs_err=%g\n", max_abs);

  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFree(d_out));
  std::free(h_x);
  std::free(h_y);
  std::free(h_out);
  return max_abs == 0.0f ? 0 : 4;
}
