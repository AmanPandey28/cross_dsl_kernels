#include <torch/extension.h>
#include <cmath>

std::vector<torch::Tensor> crossdsl_fused_residual_rmsnorm_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps, bool fast);

std::vector<torch::Tensor> fused_residual_rmsnorm(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps, bool fast) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(residual.scalar_type() == torch::kFloat32, "residual must be float32");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor shaped [rows, hidden]");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual must match x shape");
  TORCH_CHECK(weight.dim() == 1, "weight must be a 1D tensor shaped [hidden]");
  TORCH_CHECK(weight.size(0) == x.size(1), "weight length must match hidden size");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(x.get_device() == residual.get_device(), "x and residual must be on the same CUDA device");
  TORCH_CHECK(x.get_device() == weight.get_device(), "x and weight must be on the same CUDA device");
  TORCH_CHECK(x.size(0) > 0, "rows must be positive");
  TORCH_CHECK(x.size(1) > 0, "hidden size must be positive");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
  return crossdsl_fused_residual_rmsnorm_cuda(x, residual, weight, eps, fast);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "fused_residual_rmsnorm",
      &fused_residual_rmsnorm,
      "CrossDSL fused residual RMSNorm CUDA kernel",
      pybind11::arg("x"), pybind11::arg("residual"), pybind11::arg("weight"),
      pybind11::arg("eps"), pybind11::arg("fast") = false);
}
