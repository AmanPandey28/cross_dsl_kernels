#include <torch/extension.h>

void w4a16_launch(torch::Tensor x, torch::Tensor packed, torch::Tensor scales,
                 torch::Tensor bias, torch::Tensor y, int group_size,
                 int rows_per_block, bool has_bias);

void linear_out(torch::Tensor x, torch::Tensor packed, torch::Tensor scales,
                torch::Tensor bias, torch::Tensor y, int64_t group_size,
                int64_t rows_per_block, bool has_bias) {
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128 || group_size == 256,
              "group_size must be 32, 64, 128 or 256");
  TORCH_CHECK(rows_per_block == 1 || rows_per_block == 4, "rows_per_block must be 1 or 4");
  TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 16 && x.size(1) > 0,
              "x must be [M,K], 1 <= M <= 16, K > 0");
  const auto m = x.size(0), k = x.size(1);
  TORCH_CHECK(packed.dim() == 2 && packed.size(0) > 0 && packed.size(1) == (k + 1) / 2,
              "packed must be [N,ceil(K/2)]");
  const auto n = packed.size(0);
  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == n &&
              scales.size(1) == (k + group_size - 1) / group_size, "invalid scales shape");
  TORCH_CHECK(y.dim() == 2 && y.size(0) == m && y.size(1) == n, "invalid output shape");
  TORCH_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == n), "bias must be [N]");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 && scales.scalar_type() == torch::kFloat16 &&
              bias.scalar_type() == torch::kFloat16 && y.scalar_type() == torch::kFloat16 &&
              packed.scalar_type() == torch::kUInt8, "invalid W4A16 dtypes");
  for (const auto& tensor : {x, packed, scales, bias, y}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == x.device() && tensor.is_contiguous(),
                "all tensors must be contiguous CUDA tensors on the same device");
    TORCH_CHECK(!tensor.requires_grad(), "forward-only prototype");
    TORCH_CHECK(tensor.numel() < INT32_MAX, "signed 32-bit indexing required");
  }
  const auto begin = reinterpret_cast<uintptr_t>(y.data_ptr());
  const auto end = begin + y.nbytes();
  for (const auto& input : {x, packed, scales, bias}) {
    const auto start = reinterpret_cast<uintptr_t>(input.data_ptr());
    TORCH_CHECK(end <= start || start + input.nbytes() <= begin, "out overlaps an input");
  }
  w4a16_launch(x, packed, scales, bias, y, group_size, rows_per_block, has_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_out", &linear_out, "Packed INT4 decode projection into a fixed output");
}
