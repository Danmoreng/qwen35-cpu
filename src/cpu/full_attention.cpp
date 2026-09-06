#include "qwen35x/cpu/full_attention.h"
#include "q8_0_internal.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace qwen35x::cpu {

namespace detail {

[[nodiscard]] std::uint16_t attention_float_to_half(const float value) noexcept {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t magnitude = bits & 0x7fffffffU;
  if (magnitude >= 0x7f800000U) {
    return static_cast<std::uint16_t>(sign | (magnitude > 0x7f800000U ? 0x7e00U : 0x7c00U));
  }
  if (magnitude > 0x477fefffU) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  if (magnitude < 0x33000001U) {
    return static_cast<std::uint16_t>(sign);
  }
  if (magnitude < 0x38800000U) {
    const std::uint32_t exponent = magnitude >> 23U;
    const std::uint32_t mantissa = (magnitude & 0x7fffffU) | 0x800000U;
    const std::uint32_t shift = 113U - exponent;
    std::uint32_t half_mantissa = mantissa >> (shift + 13U);
    const std::uint32_t remainder_mask = (1U << (shift + 13U)) - 1U;
    const std::uint32_t remainder = mantissa & remainder_mask;
    const std::uint32_t halfway = 1U << (shift + 12U);
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1U) != 0U)) {
      ++half_mantissa;
    }
    return static_cast<std::uint16_t>(sign | half_mantissa);
  }
  const std::uint32_t rounded = magnitude + 0x00000fffU + ((magnitude >> 13U) & 1U);
  return static_cast<std::uint16_t>(sign | ((rounded - 0x38000000U) >> 13U));
}

[[nodiscard]] float attention_sigmoid_scalar(const float value) noexcept {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + std::exp(-value));
  }
  const float exp_value = std::exp(value);
  return exp_value / (1.0F + exp_value);
}

void causal_attention_batch_rows_scalar(
  const float * queries,
  const float * gates,
  const float * k_cache,
  const float * v_cache,
  const std::uint16_t * k_cache_f16,
  const std::uint16_t * v_cache_f16,
  float * scores,
  float * output,
  const std::size_t context_stride,
  const std::size_t query_width,
  const std::size_t kv_width,
  const int position_start,
  const int head_count,
  const int kv_head_count,
  const int head_dim,
  const float attention_scale,
  const std::size_t row_begin,
  const std::size_t row_end,
  const bool reuse_score_row,
  const AttentionKvRows *pages) noexcept {
  const int heads_per_kv = head_count / kv_head_count;
  for (std::size_t row = row_begin; row < row_end; ++row) {
    const std::size_t token = row / static_cast<std::size_t>(head_count);
    const int head = static_cast<int>(row % static_cast<std::size_t>(head_count));
    const int kv_head = head / heads_per_kv;
    const int sequence_length = position_start + static_cast<int>(token) + 1;
    const std::size_t head_offset =
      static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
    const float * query = queries + token * query_width + head_offset;
    const float * gate = gates + token * query_width + head_offset;
    float * score_row = scores + (reuse_score_row ? 0 : row * context_stride);
    float max_score = -std::numeric_limits<float>::infinity();
    for (int context = 0; context < sequence_length; ++context) {
      const std::size_t cache_offset =
        static_cast<std::size_t>(context) * kv_width +
        static_cast<std::size_t>(kv_head * head_dim);
      float dot = 0.0F;
      for (int column = 0; column < head_dim; ++column) {
        dot += query[static_cast<std::size_t>(column)] * ((k_cache_f16 != nullptr || (pages && pages->keys))
          ? half_to_float(attention_cache_row(k_cache_f16, pages ? pages->keys : nullptr, context, kv_width)[kv_head * head_dim + column]) : k_cache[cache_offset + column]);
      }
      dot *= attention_scale;
      score_row[static_cast<std::size_t>(context)] = dot;
      max_score = std::max(max_score, dot);
    }
    float denominator = 0.0F;
    for (int context = 0; context < sequence_length; ++context) {
      const float probability =
        std::exp(score_row[static_cast<std::size_t>(context)] - max_score);
      score_row[static_cast<std::size_t>(context)] = probability;
      denominator += probability;
    }
    const float inverse_denominator = 1.0F / (denominator > 0.0F ? denominator : 1.0F);
    float * output_head = output + token * query_width + head_offset;
    std::fill_n(output_head, static_cast<std::size_t>(head_dim), 0.0F);
    for (int context = 0; context < sequence_length; ++context) {
      const float probability =
        score_row[static_cast<std::size_t>(context)] * inverse_denominator;
      const std::size_t cache_offset =
        static_cast<std::size_t>(context) * kv_width +
        static_cast<std::size_t>(kv_head * head_dim);
      for (int column = 0; column < head_dim; ++column) {
        output_head[static_cast<std::size_t>(column)] += probability * ((v_cache_f16 != nullptr || (pages && pages->values))
          ? half_to_float(attention_cache_row(v_cache_f16, pages ? pages->values : nullptr, context, kv_width)[kv_head * head_dim + column]) : v_cache[cache_offset + column]);
      }
    }
    for (int column = 0; column < head_dim; ++column) {
      output_head[static_cast<std::size_t>(column)] *=
        attention_sigmoid_scalar(gate[column]);
    }
  }
}

#if QWEN35X_Q8_0_HAS_AVX2_TU
void attention_cache_store_f16_avx2(
  const float * input,
  std::uint16_t * output,
  std::size_t count) noexcept;

void causal_attention_batch_rows_avx2(
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
  const bool reuse_score_row,
  const AttentionKvRows *pages) noexcept;

void causal_attention_decode_gqa_pairs_avx2(
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
  const AttentionKvRows *pages) noexcept;
#endif

} // namespace detail

void attention_cache_store_f16(
  const float * input,
  std::uint16_t * output,
  const std::size_t count,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) {
    detail::attention_cache_store_f16_avx2(input, output, count);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  for (std::size_t index = 0; index < count; ++index) {
    output[index] = detail::attention_float_to_half(input[index]);
  }
}

void causal_attention_batch_rows(
  const float * queries,
  const float * gates,
  const float * k_cache,
  const float * v_cache,
  const std::uint16_t * k_cache_f16,
  const std::uint16_t * v_cache_f16,
  float * scores,
  float * output,
  const std::size_t context_stride,
  const std::size_t query_width,
  const std::size_t kv_width,
  const int position_start,
  const int head_count,
  const int kv_head_count,
  const int head_dim,
  const float attention_scale,
  const std::size_t row_begin,
  const std::size_t row_end,
  const Q8_0Backend backend,
  const bool reuse_score_row,
  const AttentionKvRows *pages) noexcept {
  if (queries == nullptr || gates == nullptr || (k_cache == nullptr && k_cache_f16 == nullptr && !(pages && pages->keys)) ||
      (v_cache == nullptr && v_cache_f16 == nullptr && !(pages && pages->values)) || scores == nullptr || output == nullptr ||
      context_stride == 0 || query_width == 0 || kv_width == 0 ||
      position_start < 0 || head_count <= 0 || kv_head_count <= 0 ||
      head_dim <= 0 || (head_count % kv_head_count) != 0 || row_begin >= row_end) {
    return;
  }
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) {
    detail::causal_attention_batch_rows_avx2(
      queries, gates, k_cache, v_cache, k_cache_f16, v_cache_f16, scores, output, context_stride,
      query_width, kv_width, position_start, head_count, kv_head_count,
      head_dim, attention_scale, row_begin, row_end, reuse_score_row, pages);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  detail::causal_attention_batch_rows_scalar(
    queries, gates, k_cache, v_cache, k_cache_f16, v_cache_f16, scores, output, context_stride,
    query_width, kv_width, position_start, head_count, kv_head_count,
    head_dim, attention_scale, row_begin, row_end, reuse_score_row, pages);
}

void causal_attention_decode_gqa_pairs(
  const float * queries,
  const float * gates,
  const float * k_cache,
  const float * v_cache,
  const std::uint16_t * k_cache_f16,
  const std::uint16_t * v_cache_f16,
  float * scores,
  float * output,
  const std::size_t context_stride,
  const std::size_t query_width,
  const std::size_t kv_width,
  const int sequence_length,
  const int head_count,
  const int kv_head_count,
  const int head_dim,
  const float attention_scale,
  const std::size_t pair_begin,
  const std::size_t pair_end,
  const Q8_0Backend backend,
  const AttentionKvRows *pages) noexcept {
  const std::size_t pair_count = static_cast<std::size_t>(head_count) / 2U;
  if (queries == nullptr || gates == nullptr || (k_cache == nullptr && k_cache_f16 == nullptr && !(pages && pages->keys)) ||
      (v_cache == nullptr && v_cache_f16 == nullptr && !(pages && pages->values)) || scores == nullptr || output == nullptr ||
      context_stride < static_cast<std::size_t>(sequence_length) ||
      query_width == 0 || kv_width == 0 || sequence_length <= 0 ||
      head_count <= 0 || kv_head_count <= 0 || head_dim <= 0 ||
      (head_count % kv_head_count) != 0 ||
      ((head_count / kv_head_count) % 2) != 0 || pair_begin >= pair_end ||
      pair_end > pair_count) {
    return;
  }
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) {
    detail::causal_attention_decode_gqa_pairs_avx2(
      queries, gates, k_cache, v_cache, k_cache_f16, v_cache_f16,
      scores, output, context_stride, query_width, kv_width,
      sequence_length, head_count, kv_head_count, head_dim, attention_scale,
      pair_begin, pair_end, pages);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  detail::causal_attention_batch_rows_scalar(
    queries, gates, k_cache, v_cache, k_cache_f16, v_cache_f16,
    scores, output, context_stride, query_width, kv_width,
    sequence_length - 1, head_count, kv_head_count, head_dim, attention_scale,
    pair_begin * 2U, pair_end * 2U, false, pages);
}

} // namespace qwen35x::cpu
