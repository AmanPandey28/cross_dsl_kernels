#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreadsPerBlock = 128;

__device__ __forceinline__ int64_t nhd_cache_index(
    int64_t physical_page,
    int64_t offset,
    int64_t head,
    int64_t dim,
    int64_t page_size,
    int64_t heads,
    int64_t head_dim) {
  return (((physical_page * page_size + offset) * heads + head) * head_dim) + dim;
}

__device__ __forceinline__ int64_t hnd_cache_index(
    int64_t physical_page,
    int64_t offset,
    int64_t head,
    int64_t dim,
    int64_t page_size,
    int64_t heads,
    int64_t head_dim) {
  return (((physical_page * heads + head) * page_size + offset) * head_dim) + dim;
}

__device__ __forceinline__ void rotate_pair(
    const float a,
    const float b,
    const float c,
    const float s,
    float* out_a,
    float* out_b) {
  *out_a = a * c - b * s;
  *out_b = a * s + b * c;
}

__global__ void rotate_q_f32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    const int64_t* __restrict__ positions,
    float* __restrict__ q_out,
    int64_t tokens,
    int64_t q_heads,
    int64_t head_dim,
    int64_t cos_width,
    int64_t rope_dim,
    bool interleaved) {
  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t pairs = rope_dim / 2;
  const int64_t position = positions[token];
  const int64_t base = (token * q_heads + head) * head_dim;

  for (int64_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
    const float c = cos[position * cos_width + pair];
    const float s = sin[position * cos_width + pair];
    if (interleaved) {
      const int64_t a_dim = pair * 2;
      const int64_t b_dim = a_dim + 1;
      float out_a;
      float out_b;
      rotate_pair(q[base + a_dim], q[base + b_dim], c, s, &out_a, &out_b);
      q_out[base + a_dim] = out_a;
      q_out[base + b_dim] = out_b;
    } else {
      const int64_t a_dim = pair;
      const int64_t b_dim = pair + pairs;
      float out_a;
      float out_b;
      rotate_pair(q[base + a_dim], q[base + b_dim], c, s, &out_a, &out_b);
      q_out[base + a_dim] = out_a;
      q_out[base + b_dim] = out_b;
    }
  }

  for (int64_t dim = rope_dim + threadIdx.x; dim < head_dim; dim += blockDim.x) {
    q_out[base + dim] = q[base + dim];
  }
}

__global__ void append_kv_f32_kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ page_table,
    const int64_t* __restrict__ sequence_ids,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    int64_t tokens,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t cos_width,
    int64_t max_pages_per_sequence,
    int64_t page_size,
    int64_t rope_dim,
    bool interleaved,
    bool cache_layout_hnd) {
  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t pairs = rope_dim / 2;
  const int64_t position = positions[token];
  const int64_t logical_page = position / page_size;
  const int64_t offset = position - logical_page * page_size;
  const int64_t sequence = sequence_ids[token];
  const int64_t physical_page = page_table[sequence * max_pages_per_sequence + logical_page];
  const int64_t src_base = (token * kv_heads + head) * head_dim;

  for (int64_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
    const float c = cos[position * cos_width + pair];
    const float s = sin[position * cos_width + pair];
    if (interleaved) {
      const int64_t a_dim = pair * 2;
      const int64_t b_dim = a_dim + 1;
      const int64_t a_cache = cache_layout_hnd
          ? hnd_cache_index(physical_page, offset, head, a_dim, page_size, kv_heads, head_dim)
          : nhd_cache_index(physical_page, offset, head, a_dim, page_size, kv_heads, head_dim);
      const int64_t b_cache = cache_layout_hnd
          ? hnd_cache_index(physical_page, offset, head, b_dim, page_size, kv_heads, head_dim)
          : nhd_cache_index(physical_page, offset, head, b_dim, page_size, kv_heads, head_dim);
      float out_a;
      float out_b;
      rotate_pair(k[src_base + a_dim], k[src_base + b_dim], c, s, &out_a, &out_b);
      k_cache[a_cache] = out_a;
      k_cache[b_cache] = out_b;
    } else {
      const int64_t a_dim = pair;
      const int64_t b_dim = pair + pairs;
      const int64_t a_cache = cache_layout_hnd
          ? hnd_cache_index(physical_page, offset, head, a_dim, page_size, kv_heads, head_dim)
          : nhd_cache_index(physical_page, offset, head, a_dim, page_size, kv_heads, head_dim);
      const int64_t b_cache = cache_layout_hnd
          ? hnd_cache_index(physical_page, offset, head, b_dim, page_size, kv_heads, head_dim)
          : nhd_cache_index(physical_page, offset, head, b_dim, page_size, kv_heads, head_dim);
      float out_a;
      float out_b;
      rotate_pair(k[src_base + a_dim], k[src_base + b_dim], c, s, &out_a, &out_b);
      k_cache[a_cache] = out_a;
      k_cache[b_cache] = out_b;
    }
  }

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    const int64_t cache_idx = cache_layout_hnd
        ? hnd_cache_index(physical_page, offset, head, dim, page_size, kv_heads, head_dim)
        : nhd_cache_index(physical_page, offset, head, dim, page_size, kv_heads, head_dim);
    v_cache[cache_idx] = v[src_base + dim];
    if (dim >= rope_dim) {
      k_cache[cache_idx] = k[src_base + dim];
    }
  }
}

__global__ void rope_append_f32_kernel(
    const float* q, const float* k, const float* v, const float* cos, const float* sin,
    const int64_t* positions, const int64_t* table, const int64_t* sequences,
    float* q_out, float* k_cache, float* v_cache,
    int64_t hq, int64_t hkv, int64_t dim, int64_t cos_width,
    int64_t max_pages, int64_t page_size, int64_t rope_dim, bool interleaved, bool hnd) {
  const int64_t token = blockIdx.x;
  const bool is_q = blockIdx.y < hq;
  const int64_t head = is_q ? blockIdx.y : blockIdx.y - hq;
  const int64_t position = positions[token];
  const int64_t src_base = (token * (is_q ? hq : hkv) + head) * dim;
  const float* src = is_q ? q : k;
  float* dst = is_q ? q_out : k_cache;
  int64_t dst_base = src_base;
  if (!is_q) {
    const int64_t page = table[sequences[token] * max_pages + position / page_size];
    const int64_t offset = position % page_size;
    dst_base = hnd ? hnd_cache_index(page, offset, head, 0, page_size, hkv, dim)
                   : nhd_cache_index(page, offset, head, 0, page_size, hkv, dim);
  }
  const int64_t pairs = rope_dim / 2;
  for (int64_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
    const int64_t a = interleaved ? 2 * pair : pair;
    const int64_t b = interleaved ? a + 1 : pair + pairs;
    float ra, rb;
    rotate_pair(src[src_base + a], src[src_base + b], cos[position * cos_width + pair],
                sin[position * cos_width + pair], &ra, &rb);
    dst[dst_base + a] = ra;
    dst[dst_base + b] = rb;
  }
  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    if (d >= rope_dim) dst[dst_base + d] = src[src_base + d];
    if (!is_q) v_cache[dst_base + d] = v[src_base + d];
  }
}

}  // namespace

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
    bool cache_layout_hnd, bool fused) {
  const at::cuda::CUDAGuard device_guard(q.device());
  auto q_out = torch::empty_like(q);

  const int64_t tokens = q.size(0);
  const int64_t q_heads = q.size(1);
  const int64_t kv_heads = k.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t cos_width = cos.size(1);
  const int64_t max_pages_per_sequence = page_table.size(1);
  const dim3 q_grid(tokens, q_heads);
  const dim3 kv_grid(tokens, kv_heads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (fused) {
    rope_append_f32_kernel<<<dim3(tokens, q_heads + kv_heads), 64, 0, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), cos.data_ptr<float>(),
        sin.data_ptr<float>(), positions.data_ptr<int64_t>(), page_table.data_ptr<int64_t>(),
        sequence_ids.data_ptr<int64_t>(), q_out.data_ptr<float>(), k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(), q_heads, kv_heads, head_dim, cos_width,
        max_pages_per_sequence, page_size, rope_dim, interleaved, cache_layout_hnd);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return q_out;
  }

  rotate_q_f32_kernel<<<q_grid, kThreadsPerBlock, 0, stream>>>(
      q.data_ptr<float>(),
      cos.data_ptr<float>(),
      sin.data_ptr<float>(),
      positions.data_ptr<int64_t>(),
      q_out.data_ptr<float>(),
      tokens,
      q_heads,
      head_dim,
      cos_width,
      rope_dim,
      interleaved);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  append_kv_f32_kernel<<<kv_grid, kThreadsPerBlock, 0, stream>>>(
      k.data_ptr<float>(),
      v.data_ptr<float>(),
      cos.data_ptr<float>(),
      sin.data_ptr<float>(),
      positions.data_ptr<int64_t>(),
      page_table.data_ptr<int64_t>(),
      sequence_ids.data_ptr<int64_t>(),
      k_cache.data_ptr<float>(),
      v_cache.data_ptr<float>(),
      tokens,
      kv_heads,
      head_dim,
      cos_width,
      max_pages_per_sequence,
      page_size,
      rope_dim,
      interleaved,
      cache_layout_hnd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return q_out;
}
