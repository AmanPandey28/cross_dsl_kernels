#include <string>
#include <torch/extension.h>

torch::Tensor crossdsl_decode_gemv_rows_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, bool weight_is_nk);

torch::Tensor crossdsl_decode_gemv_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    bool weight_is_nk);

torch::Tensor crossdsl_decode_gemv_warp_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    bool weight_is_nk);

torch::Tensor crossdsl_decode_gemv_cublas_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    bool weight_is_nk);

torch::Tensor crossdsl_decode_gemv_cublaslt_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    bool weight_is_nk);

bool validate_decode_gemv_inputs(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, const std::string& layout) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(bias.numel() == 0 || bias.is_cuda(), "bias must be empty or a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(bias.numel() == 0 || bias.scalar_type() == torch::kFloat32, "bias must be float32 when present");
  TORCH_CHECK(x.dim() == 2, "x must be shaped [M, K]");
  TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
  TORCH_CHECK(layout == "KN" || layout == "NK", "layout must be 'KN' or 'NK'");
  const bool weight_is_nk = layout == "NK";
  const int64_t k = x.size(1);
  const int64_t n = weight_is_nk ? weight.size(0) : weight.size(1);
  TORCH_CHECK(x.size(0) > 0, "M must be positive");
  TORCH_CHECK(k > 0, "K must be positive");
  TORCH_CHECK(n > 0, "N must be positive");
  TORCH_CHECK(weight_is_nk ? weight.size(1) == k : weight.size(0) == k, "weight K dimension must match x");
  TORCH_CHECK(bias.numel() == 0 || (bias.dim() == 1 && bias.size(0) == n), "bias must be empty or shaped [N]");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(bias.numel() == 0 || bias.is_contiguous(), "bias must be contiguous when present");
  TORCH_CHECK(x.get_device() == weight.get_device(), "x and weight must be on the same CUDA device");
  TORCH_CHECK(bias.numel() == 0 || x.get_device() == bias.get_device(), "x and bias must be on the same CUDA device");
  return weight_is_nk;
}

torch::Tensor decode_gemv(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, const std::string& layout) {
  const bool weight_is_nk = validate_decode_gemv_inputs(x, weight, bias, layout);
  return crossdsl_decode_gemv_cuda(x, weight, bias, weight_is_nk);
}

torch::Tensor decode_gemv_warp(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, const std::string& layout) {
  const bool weight_is_nk = validate_decode_gemv_inputs(x, weight, bias, layout);
  return crossdsl_decode_gemv_warp_cuda(x, weight, bias, weight_is_nk);
}

torch::Tensor decode_gemv_cublas(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, const std::string& layout) {
  const bool weight_is_nk = validate_decode_gemv_inputs(x, weight, bias, layout);
  return crossdsl_decode_gemv_cublas_cuda(x, weight, bias, weight_is_nk);
}

torch::Tensor decode_gemv_cublaslt(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, const std::string& layout) {
  const bool weight_is_nk = validate_decode_gemv_inputs(x, weight, bias, layout);
  return crossdsl_decode_gemv_cublaslt_cuda(x, weight, bias, weight_is_nk);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_gemv_rows", [](torch::Tensor x, torch::Tensor w, torch::Tensor b, const std::string& layout) {
    return crossdsl_decode_gemv_rows_cuda(x, w, b, validate_decode_gemv_inputs(x, w, b, layout));
  }, "CrossDSL coalesced four-row GEMV");
  m.def("decode_gemv", &decode_gemv, "CrossDSL decode GEMV CUDA v0 kernel");
  m.def("decode_gemv_warp", &decode_gemv_warp, "CrossDSL decode GEMV CUDA v1 warp-per-output kernel");
  m.def("decode_gemv_cublas", &decode_gemv_cublas, "CrossDSL decode GEMV explicit cuBLAS baseline");
  m.def("decode_gemv_cublaslt", &decode_gemv_cublaslt, "CrossDSL decode GEMV explicit cuBLASLt baseline");
}
