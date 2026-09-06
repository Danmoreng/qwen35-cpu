#pragma once

#include "qwen35x/cpu/q8_0.h"

#include <cstddef>
#include <cstdint>

namespace qwen35x::cpu {

// Optional FP16 row table for segmented/paged caches. Each entry references one
// contiguous [kv_head, head_dim] row and stays valid for the synchronous call.
struct AttentionKvRows {
  const std::uint16_t *const *keys = nullptr;
  const std::uint16_t *const *values = nullptr;
};
inline const std::uint16_t *attention_cache_row(
    const std::uint16_t *base, const std::uint16_t *const *rows,
    std::size_t context, std::size_t width) noexcept {
  return rows ? rows[context] : base + context * width;
}

void attention_cache_store_f16(
  const float * input,
  std::uint16_t * output,
  std::size_t count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// Computes a range of flattened [token, query_head] rows for causal GQA.
// Queries, gates, and output are token-major. K/V cache entries are
// [context, kv_head, head_dim]. Scores is caller-owned scratch with one
// context_stride row per flattened query row, which makes disjoint row ranges
// safe to execute concurrently. FP16 pointers take precedence when supplied;
// the corresponding FP32 cache may then be null, including for scalar fallback.
// With reuse_score_row, scores needs only context_stride floats and must be
// private to this call. Rows within the call reuse it sequentially.
void causal_attention_batch_rows(
  const float * queries,
  const float * gates,
  const float * k_cache,
  const float * v_cache,
  const std::uint16_t * k_cache_f16,
  const std::uint16_t * v_cache_f16,
  float * scores,
  float * output,
  std::size_t context_stride,
  std::size_t query_width,
  std::size_t kv_width,
  int position_start,
  int head_count,
  int kv_head_count,
  int head_dim,
  float attention_scale,
  std::size_t row_begin,
  std::size_t row_end,
  Q8_0Backend backend = Q8_0Backend::auto_select,
  bool reuse_score_row = false,
  const AttentionKvRows *pages = nullptr) noexcept;

// Decode-only GQA path that evaluates two query heads sharing one KV head
// together. pair_begin/pair_end address adjacent query-head pairs.
void causal_attention_decode_gqa_pairs(
  const float * queries,
  const float * gates,
  const float * k_cache,
  const float * v_cache,
  const std::uint16_t * k_cache_f16,
  const std::uint16_t * v_cache_f16,
  float * scores,
  float * output,
  std::size_t context_stride,
  std::size_t query_width,
  std::size_t kv_width,
  int sequence_length,
  int head_count,
  int kv_head_count,
  int head_dim,
  float attention_scale,
  std::size_t pair_begin,
  std::size_t pair_end,
  Q8_0Backend backend = Q8_0Backend::auto_select,
  const AttentionKvRows *pages = nullptr) noexcept;

} // namespace qwen35x::cpu
