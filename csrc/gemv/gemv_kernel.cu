#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <limits>
#include <torch/extension.h>

namespace {

void check_cublas(cublasStatus_t status, const char* context) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      context,
      " failed with cublasStatus_t=",
      static_cast<int>(status));
}

constexpr size_t kCublasLtWorkspaceLimitBytes = 4 * 1024 * 1024;

struct LtMatmulDesc {
  cublasLtMatmulDesc_t value = nullptr;

  LtMatmulDesc(cublasComputeType_t compute_type, cudaDataType_t scale_type) {
    check_cublas(cublasLtMatmulDescCreate(&value, compute_type, scale_type), "cublasLtMatmulDescCreate");
  }

  ~LtMatmulDesc() {
    if (value != nullptr) {
      cublasLtMatmulDescDestroy(value);
    }
  }

  LtMatmulDesc(const LtMatmulDesc&) = delete;
  LtMatmulDesc& operator=(const LtMatmulDesc&) = delete;
};

struct LtMatrixLayout {
  cublasLtMatrixLayout_t value = nullptr;

  LtMatrixLayout(cudaDataType_t type, uint64_t rows, uint64_t cols, int64_t ld) {
    check_cublas(cublasLtMatrixLayoutCreate(&value, type, rows, cols, ld), "cublasLtMatrixLayoutCreate");
  }

  ~LtMatrixLayout() {
    if (value != nullptr) {
      cublasLtMatrixLayoutDestroy(value);
    }
  }

  LtMatrixLayout(const LtMatrixLayout&) = delete;
  LtMatrixLayout& operator=(const LtMatrixLayout&) = delete;
};

struct LtMatmulPreference {
  cublasLtMatmulPreference_t value = nullptr;

  LtMatmulPreference() {
    check_cublas(cublasLtMatmulPreferenceCreate(&value), "cublasLtMatmulPreferenceCreate");
  }

  ~LtMatmulPreference() {
    if (value != nullptr) {
      cublasLtMatmulPreferenceDestroy(value);
    }
  }

  LtMatmulPreference(const LtMatmulPreference&) = delete;
  LtMatmulPreference& operator=(const LtMatmulPreference&) = delete;
};

}  // namespace

__global__ void decode_gemv_f32_v0_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int64_t k,
    int64_t n,
    bool weight_is_nk,
    bool has_bias) {
  extern __shared__ float partial_sums[];
  const int64_t out_col = static_cast<int64_t>(blockIdx.x);
  const int64_t out_row = static_cast<int64_t>(blockIdx.y);

  float thread_sum = 0.0f;
  for (int64_t kk = threadIdx.x; kk < k; kk += blockDim.x) {
    const float x_val = x[out_row * k + kk];
    const int64_t weight_idx = weight_is_nk ? out_col * k + kk : kk * n + out_col;
    thread_sum += x_val * weight[weight_idx];
  }

  partial_sums[threadIdx.x] = thread_sum;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float out = partial_sums[0];
    if (has_bias) {
      out += bias[out_col];
    }
    y[out_row * n + out_col] = out;
  }
}

__global__ void decode_gemv_f32_warp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int64_t total_outputs,
    int64_t k,
    int64_t n,
    bool weight_is_nk,
    bool has_bias) {
  constexpr int warp_size = 32;
  constexpr int warps_per_block = 8;
  const int lane = threadIdx.x & (warp_size - 1);
  const int warp_in_block = threadIdx.x >> 5;
  const int64_t linear_output =
      (static_cast<int64_t>(blockIdx.x) * warps_per_block) + warp_in_block;
  if (linear_output >= total_outputs) {
    return;
  }

  const int64_t out_row = linear_output / n;
  const int64_t out_col = linear_output - (out_row * n);
  float thread_sum = 0.0f;
  for (int64_t kk = lane; kk < k; kk += warp_size) {
    const float x_val = x[out_row * k + kk];
    const int64_t weight_idx = weight_is_nk ? out_col * k + kk : kk * n + out_col;
    thread_sum += x_val * weight[weight_idx];
  }

  for (int offset = warp_size >> 1; offset > 0; offset >>= 1) {
    thread_sum += __shfl_down_sync(0xffffffffu, thread_sum, offset);
  }

  if (lane == 0) {
    float out = thread_sum;
    if (has_bias) {
      out += bias[out_col];
    }
    y[out_row * n + out_col] = out;
  }
}

__global__ void add_bias_rows_f32_kernel(
    float* __restrict__ y,
    const float* __restrict__ bias,
    int64_t total_outputs,
    int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total_outputs) {
    y[idx] += bias[idx % n];
  }
}

torch::Tensor crossdsl_decode_gemv_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    bool weight_is_nk) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t n = weight_is_nk ? weight.size(0) : weight.size(1);
  auto y = torch::empty({m, n}, x.options());

  constexpr int threads = 256;
  const dim3 blocks(static_cast<unsigned int>(n), static_cast<unsigned int>(m));
  const size_t shared_bytes = threads * sizeof(float);
  const bool has_bias = bias.numel() != 0;
  const float* bias_ptr = has_bias ? bias.data_ptr<float>() : nullptr;

  decode_gemv_f32_v0_kernel<<<
      blocks,
      threads,
      shared_bytes,
      at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias_ptr,
      y.data_ptr<float>(),
      k,
      n,
      weight_is_nk,
      has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor crossdsl_decode_gemv_cublas_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    bool weight_is_nk) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t m64 = x.size(0);
  const int64_t k64 = x.size(1);
  const int64_t n64 = weight_is_nk ? weight.size(0) : weight.size(1);
  const int64_t int_max = static_cast<int64_t>(std::numeric_limits<int>::max());
  TORCH_CHECK(m64 <= int_max && k64 <= int_max && n64 <= int_max, "cuBLAS dimensions must fit int");

  const int m = static_cast<int>(m64);
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  auto y = torch::empty({m64, n64}, x.options());

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const float* x_ptr = x.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  float* y_ptr = y.data_ptr<float>();

  if (m == 1) {
    const cublasOperation_t trans = weight_is_nk ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int rows = weight_is_nk ? k : n;
    const int cols = weight_is_nk ? n : k;
    const int lda = rows;
    check_cublas(
        cublasSgemv(handle, trans, rows, cols, &alpha, weight_ptr, lda, x_ptr, 1, &beta, y_ptr, 1),
        "cublasSgemv");
  } else {
    const cublasOperation_t transa = weight_is_nk ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int lda = weight_is_nk ? k : n;
    check_cublas(
        cublasSgemm(
            handle,
            transa,
            CUBLAS_OP_N,
            n,
            m,
            k,
            &alpha,
            weight_ptr,
            lda,
            x_ptr,
            k,
            &beta,
            y_ptr,
            n),
        "cublasSgemm");
  }

  if (bias.numel() != 0) {
    constexpr int threads = 256;
    const int64_t total_outputs = m64 * n64;
    const unsigned int blocks =
        static_cast<unsigned int>((total_outputs + threads - 1) / threads);
    add_bias_rows_f32_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        y_ptr,
        bias.data_ptr<float>(),
        total_outputs,
        n64);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return y;
}

torch::Tensor crossdsl_decode_gemv_cublaslt_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    bool weight_is_nk) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t m64 = x.size(0);
  const int64_t k64 = x.size(1);
  const int64_t n64 = weight_is_nk ? weight.size(0) : weight.size(1);
  const int64_t int_max = static_cast<int64_t>(std::numeric_limits<int>::max());
  TORCH_CHECK(m64 <= int_max && k64 <= int_max && n64 <= int_max, "cuBLASLt dimensions must fit int");

  const int m = static_cast<int>(m64);
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  auto y = torch::empty({m64, n64}, x.options());

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const float* x_ptr = x.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  float* y_ptr = y.data_ptr<float>();

  LtMatmulDesc operation(CUBLAS_COMPUTE_32F, CUDA_R_32F);
  const cublasOperation_t transa = weight_is_nk ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transb = CUBLAS_OP_N;
  check_cublas(
      cublasLtMatmulDescSetAttribute(operation.value, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)),
      "cublasLtMatmulDescSetAttribute(TRANSA)");
  check_cublas(
      cublasLtMatmulDescSetAttribute(operation.value, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)),
      "cublasLtMatmulDescSetAttribute(TRANSB)");

  const int a_rows = weight_is_nk ? k : n;
  const int a_cols = weight_is_nk ? n : k;
  const int a_ld = weight_is_nk ? k : n;
  LtMatrixLayout a_desc(CUDA_R_32F, a_rows, a_cols, a_ld);
  LtMatrixLayout b_desc(CUDA_R_32F, k, m, k);
  LtMatrixLayout c_desc(CUDA_R_32F, n, m, n);
  LtMatrixLayout d_desc(CUDA_R_32F, n, m, n);

  LtMatmulPreference preference;
  const size_t torch_workspace_bytes = at::cuda::getCUDABlasLtWorkspaceSize();
  const size_t workspace_bytes = std::min(torch_workspace_bytes, kCublasLtWorkspaceLimitBytes);
  check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference.value,
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspace_bytes,
          sizeof(workspace_bytes)),
      "cublasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE_BYTES)");

  constexpr int requested_algo_count = 8;
  cublasLtMatmulHeuristicResult_t heuristic_results[requested_algo_count] = {};
  int returned_algo_count = 0;
  cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();
  check_cublas(
      cublasLtMatmulAlgoGetHeuristic(
          handle,
          operation.value,
          a_desc.value,
          b_desc.value,
          c_desc.value,
          d_desc.value,
          preference.value,
          requested_algo_count,
          heuristic_results,
          &returned_algo_count),
      "cublasLtMatmulAlgoGetHeuristic");
  TORCH_CHECK(returned_algo_count > 0, "cublasLtMatmulAlgoGetHeuristic returned no algorithms");
  const cublasLtMatmulHeuristicResult_t* selected_algo = nullptr;
  for (int i = 0; i < returned_algo_count; ++i) {
    if (heuristic_results[i].state == CUBLAS_STATUS_SUCCESS) {
      selected_algo = &heuristic_results[i];
      break;
    }
  }
  TORCH_CHECK(
      selected_algo != nullptr,
      "cublasLtMatmulAlgoGetHeuristic returned no runnable algorithms; first state=",
      static_cast<int>(heuristic_results[0].state));
  TORCH_CHECK(
      selected_algo->workspaceSize <= workspace_bytes,
      "cuBLASLt heuristic workspace exceeds bounded preference");

  void* workspace = workspace_bytes == 0 ? nullptr : at::cuda::getCUDABlasLtWorkspace();
  check_cublas(
      cublasLtMatmul(
          handle,
          operation.value,
          &alpha,
          weight_ptr,
          a_desc.value,
          x_ptr,
          b_desc.value,
          &beta,
          y_ptr,
          c_desc.value,
          y_ptr,
          d_desc.value,
          &selected_algo->algo,
          workspace,
          workspace_bytes,
          at::cuda::getCurrentCUDAStream()),
      "cublasLtMatmul");

  if (bias.numel() != 0) {
    constexpr int threads = 256;
    const int64_t total_outputs = m64 * n64;
    const unsigned int blocks =
        static_cast<unsigned int>((total_outputs + threads - 1) / threads);
    add_bias_rows_f32_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        y_ptr,
        bias.data_ptr<float>(),
        total_outputs,
        n64);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return y;
}

torch::Tensor crossdsl_decode_gemv_warp_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    bool weight_is_nk) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t n = weight_is_nk ? weight.size(0) : weight.size(1);
  const int64_t total_outputs = m * n;
  auto y = torch::empty({m, n}, x.options());

  constexpr int threads = 256;
  constexpr int warps_per_block = threads / 32;
  const unsigned int block_count =
      static_cast<unsigned int>((total_outputs + warps_per_block - 1) / warps_per_block);
  const bool has_bias = bias.numel() != 0;
  const float* bias_ptr = has_bias ? bias.data_ptr<float>() : nullptr;

  decode_gemv_f32_warp_kernel<<<
      block_count,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias_ptr,
      y.data_ptr<float>(),
      total_outputs,
      k,
      n,
      weight_is_nk,
      has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// KN lanes span adjacent columns; NK lanes span adjacent reduction elements.
template <bool NK>
__global__ void decode_gemv_rows_kernel(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ y,
    int64_t m, int64_t k, int64_t n, bool has_bias) {
  constexpr int Rows = 4;
  constexpr int Warps = 8;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t col = blockIdx.x * (NK ? Warps : 32) + (NK ? warp : lane);
  const int64_t row = blockIdx.y * Rows;
  float acc[Rows] = {};
  for (int64_t kk = NK ? lane : warp; kk < k; kk += NK ? 32 : Warps) {
    const float w = col < n ? weight[NK ? col * k + kk : kk * n + col] : 0.0f;
    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
      if (row + r < m) acc[r] += x[(row + r) * k + kk] * w;
    }
  }
  if constexpr (NK) {
    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
      for (int delta = 16; delta > 0; delta >>= 1) {
        acc[r] += __shfl_down_sync(0xffffffff, acc[r], delta);
      }
      if (lane == 0 && col < n && row + r < m) {
        y[(row + r) * n + col] = acc[r] + (has_bias ? bias[col] : 0.0f);
      }
    }
  } else {
    __shared__ float partial[Rows][Warps][32];
    #pragma unroll
    for (int r = 0; r < Rows; ++r) partial[r][warp][lane] = acc[r];
    __syncthreads();
    if (warp == 0 && col < n) {
      #pragma unroll
      for (int r = 0; r < Rows; ++r) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < Warps; ++w) sum += partial[r][w][lane];
        if (row + r < m) y[(row + r) * n + col] = sum + (has_bias ? bias[col] : 0.0f);
      }
    }
  }
}

torch::Tensor crossdsl_decode_gemv_rows_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, bool weight_is_nk) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t m = x.size(0), k = x.size(1);
  const int64_t n = weight_is_nk ? weight.size(0) : weight.size(1);
  auto y = torch::empty({m, n}, x.options());
  const dim3 grid((n + (weight_is_nk ? 7 : 31)) / (weight_is_nk ? 8 : 32), (m + 3) / 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  const float* bias_ptr = bias.numel() != 0 ? bias.data_ptr<float>() : nullptr;
  if (weight_is_nk) {
    decode_gemv_rows_kernel<true><<<grid, 256, 0, stream>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
        y.data_ptr<float>(), m, k, n, bias.numel() != 0);
  } else {
    decode_gemv_rows_kernel<false><<<grid, 256, 0, stream>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
        y.data_ptr<float>(), m, k, n, bias.numel() != 0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
