#include <string>
#include <torch/extension.h>

torch::Tensor crossdsl_rope_gqa_paged_kv_append_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor positions,
    torch::Tensor page_table,
    torch::Tensor sequence_ids,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int64_t page_size,
    int64_t rope_dim,
    bool interleaved,
    bool cache_layout_hnd);

bool validate_common_tensor(torch::Tensor tensor, const char* name, torch::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unexpected dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  return true;
}

torch::Tensor rope_gqa_paged_kv_append(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor positions,
    torch::Tensor page_table,
    torch::Tensor sequence_ids,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int64_t page_size,
    int64_t rope_dim,
    bool interleaved,
    const std::string& kv_layout) {
  validate_common_tensor(q, "q", torch::kFloat32);
  validate_common_tensor(k, "k", torch::kFloat32);
  validate_common_tensor(v, "v", torch::kFloat32);
  validate_common_tensor(cos, "cos", torch::kFloat32);
  validate_common_tensor(sin, "sin", torch::kFloat32);
  validate_common_tensor(positions, "positions", torch::kInt64);
  validate_common_tensor(page_table, "page_table", torch::kInt64);
  validate_common_tensor(sequence_ids, "sequence_ids", torch::kInt64);
  validate_common_tensor(k_cache, "k_cache", torch::kFloat32);
  validate_common_tensor(v_cache, "v_cache", torch::kFloat32);

  TORCH_CHECK(q.dim() == 3, "q must be shaped [tokens, q_heads, head_dim]");
  TORCH_CHECK(k.dim() == 3, "k must be shaped [tokens, kv_heads, head_dim]");
  TORCH_CHECK(v.dim() == 3, "v must be shaped [tokens, kv_heads, head_dim]");
  TORCH_CHECK(cos.dim() == 2, "cos must be shaped [max_position, rope_dim / 2]");
  TORCH_CHECK(sin.dim() == 2, "sin must be shaped [max_position, rope_dim / 2]");
  TORCH_CHECK(positions.dim() == 1, "positions must be shaped [tokens]");
  TORCH_CHECK(page_table.dim() == 2, "page_table must be shaped [sequences, max_pages]");
  TORCH_CHECK(sequence_ids.dim() == 1, "sequence_ids must be shaped [tokens]");
  TORCH_CHECK(k_cache.dim() == 4, "k_cache must be a 4D paged cache tensor");
  TORCH_CHECK(v_cache.dim() == 4, "v_cache must be a 4D paged cache tensor");

  const auto tokens = q.size(0);
  const auto q_heads = q.size(1);
  const auto kv_heads = k.size(1);
  const auto head_dim = q.size(2);
  TORCH_CHECK(tokens > 0, "tokens must be positive");
  TORCH_CHECK(q_heads > 0, "q_heads must be positive");
  TORCH_CHECK(kv_heads > 0, "kv_heads must be positive");
  TORCH_CHECK(head_dim > 0, "head_dim must be positive");
  TORCH_CHECK(k.size(0) == tokens && v.size(0) == tokens, "q, k, and v token counts must match");
  TORCH_CHECK(k.size(2) == head_dim && v.size(2) == head_dim, "q, k, and v head_dim must match");
  TORCH_CHECK(v.size(1) == kv_heads, "k and v kv_heads must match");
  TORCH_CHECK(positions.size(0) == tokens, "positions length must match tokens");
  TORCH_CHECK(sequence_ids.size(0) == tokens, "sequence_ids length must match tokens");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(rope_dim > 0 && rope_dim % 2 == 0, "rope_dim must be a positive even integer");
  TORCH_CHECK(rope_dim <= head_dim, "rope_dim cannot exceed head_dim");
  TORCH_CHECK(cos.size(1) >= rope_dim / 2, "cos must have at least rope_dim / 2 columns");
  TORCH_CHECK(sin.sizes() == cos.sizes(), "sin must match cos shape");
  TORCH_CHECK(page_table.size(1) > 0, "page_table must contain at least one page per sequence row");

  const bool cache_layout_hnd = kv_layout == "HND";
  TORCH_CHECK(kv_layout == "NHD" || kv_layout == "HND", "kv_layout must be NHD or HND");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache shapes must match");
  if (cache_layout_hnd) {
    TORCH_CHECK(k_cache.size(1) == kv_heads, "HND cache dimension 1 must match kv_heads");
    TORCH_CHECK(k_cache.size(2) == page_size, "HND cache dimension 2 must match page_size");
    TORCH_CHECK(k_cache.size(3) == head_dim, "HND cache dimension 3 must match head_dim");
  } else {
    TORCH_CHECK(k_cache.size(1) == page_size, "NHD cache dimension 1 must match page_size");
    TORCH_CHECK(k_cache.size(2) == kv_heads, "NHD cache dimension 2 must match kv_heads");
    TORCH_CHECK(k_cache.size(3) == head_dim, "NHD cache dimension 3 must match head_dim");
  }

  const int device = q.get_device();
  TORCH_CHECK(k.get_device() == device, "k must be on the same CUDA device as q");
  TORCH_CHECK(v.get_device() == device, "v must be on the same CUDA device as q");
  TORCH_CHECK(cos.get_device() == device, "cos must be on the same CUDA device as q");
  TORCH_CHECK(sin.get_device() == device, "sin must be on the same CUDA device as q");
  TORCH_CHECK(positions.get_device() == device, "positions must be on the same CUDA device as q");
  TORCH_CHECK(page_table.get_device() == device, "page_table must be on the same CUDA device as q");
  TORCH_CHECK(sequence_ids.get_device() == device, "sequence_ids must be on the same CUDA device as q");
  TORCH_CHECK(k_cache.get_device() == device, "k_cache must be on the same CUDA device as q");
  TORCH_CHECK(v_cache.get_device() == device, "v_cache must be on the same CUDA device as q");

  return crossdsl_rope_gqa_paged_kv_append_cuda(
      q,
      k,
      v,
      cos,
      sin,
      positions,
      page_table,
      sequence_ids,
      k_cache,
      v_cache,
      page_size,
      rope_dim,
      interleaved,
      cache_layout_hnd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "rope_gqa_paged_kv_append",
      &rope_gqa_paged_kv_append,
      "CrossDSL RoPE plus GQA paged-KV append CUDA baseline");
}
