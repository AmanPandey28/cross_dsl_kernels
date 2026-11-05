#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <optional>
#include <string>
#include <torch/extension.h>

using Launcher = void (*)(const float*, const float*, const float*, float*,
                         int64_t, int64_t, int64_t, bool, cudaStream_t);
void launch_gemv_warp_clean(const float*, const float*, const float*, float*,
                            int64_t, int64_t, int64_t, bool, cudaStream_t);
void launch_gemv_rows_clean(const float*, const float*, const float*, float*,
                            int64_t, int64_t, int64_t, bool, cudaStream_t);
void launch_gemm_tiled(const float*, const float*, const float*, float*,
                       int64_t, int64_t, int64_t, bool, cudaStream_t);

namespace {

bool validate_inputs(const torch::Tensor& x, const torch::Tensor& weight,
                     const std::optional<torch::Tensor>& bias,
                     const std::string& layout) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "x and weight must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32,
              "x and weight must be float32");
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "x and weight must be 2D");
  TORCH_CHECK(layout == "KN" || layout == "NK", "layout must be KN or NK");
  const bool weight_is_nk = layout == "NK";
  const int64_t m = x.size(0), k = x.size(1);
  const int64_t n = weight_is_nk ? weight.size(0) : weight.size(1);
  TORCH_CHECK(m > 0 && k > 0 && n > 0, "M, K and N must be positive");
  TORCH_CHECK((weight_is_nk ? weight.size(1) : weight.size(0)) == k,
              "weight K dimension must match x");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "x and weight must be contiguous");
  TORCH_CHECK(x.device() == weight.device(), "x and weight must be on the same device");
  // Check products before multiplying or converting launch dimensions.
  constexpr int64_t max_elements = std::numeric_limits<int64_t>::max() / sizeof(float);
  TORCH_CHECK(m <= max_elements / n && x.numel() <= max_elements && weight.numel() <= max_elements,
              "tensor byte offsets exceed signed 64-bit range");
  if (bias.has_value()) {
    const auto& b = *bias;
    TORCH_CHECK(b.is_cuda() && b.device() == x.device(), "bias must be on the same CUDA device");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(b.dim() == 1 && b.size(0) == n, "bias must be shaped [N]; use None for no bias");
    TORCH_CHECK(b.is_contiguous(), "bias must be contiguous");
  }
  return weight_is_nk;
}

torch::Tensor project(torch::Tensor x, torch::Tensor weight,
                      std::optional<torch::Tensor> bias, const std::string& layout,
                      Launcher launch, int rows_per_block, int cols_per_block) {
  const bool weight_is_nk = validate_inputs(x, weight, bias, layout);
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t m = x.size(0), k = x.size(1);
  const int64_t n = weight_is_nk ? weight.size(0) : weight.size(1);
  // Warp baseline flattens outputs; the other algorithms use a two-dimensional grid.
  const int64_t grid_x = rows_per_block == 0 ? (m * n - 1) / 8 + 1
                                            : (n - 1) / cols_per_block + 1;
  const int64_t grid_y = rows_per_block == 0 ? 1 : (m - 1) / rows_per_block + 1;
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(grid_x <= properties->maxGridSize[0] && grid_y <= properties->maxGridSize[1],
              "shape exceeds this device's CUDA grid limits");
  auto output = torch::empty({m, n}, x.options());
  const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  launch(x.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
         m, k, n, weight_is_nk, at::cuda::getCurrentCUDAStream());
  return output;
}

torch::Tensor decode_gemv_warp_clean(torch::Tensor x, torch::Tensor weight,
                                    std::optional<torch::Tensor> bias, const std::string& layout) {
  return project(x, weight, bias, layout, launch_gemv_warp_clean, 0, 8);
}

torch::Tensor decode_gemv_rows_clean(torch::Tensor x, torch::Tensor weight,
                                    std::optional<torch::Tensor> bias, const std::string& layout) {
  return project(x, weight, bias, layout, launch_gemv_rows_clean, 4, layout == "NK" ? 8 : 32);
}

torch::Tensor decode_gemm_tiled(torch::Tensor x, torch::Tensor weight,
                              std::optional<torch::Tensor> bias, const std::string& layout) {
  return project(x, weight, bias, layout, launch_gemm_tiled, 8, 32);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("decode_gemv_warp_clean", &decode_gemv_warp_clean,
             pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias") = pybind11::none(),
             pybind11::arg("layout") = "KN");
  module.def("decode_gemv_rows_clean", &decode_gemv_rows_clean,
             pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias") = pybind11::none(),
             pybind11::arg("layout") = "KN");
  module.def("decode_gemm_tiled", &decode_gemm_tiled,
             pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias") = pybind11::none(),
             pybind11::arg("layout") = "KN");
}
