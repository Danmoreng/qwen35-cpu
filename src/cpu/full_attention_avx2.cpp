#include "qwen35x/cpu/full_attention.h"

#include "f16c_compat.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace qwen35x::cpu::detail {

namespace {

[[nodiscard]] float horizontal_sum_f32(const __m256 value) noexcept {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

// Cephes-derived exp approximation used by common SIMD math libraries. The
// range reduction keeps softmax inputs in a compact interval and the degree-5
// polynomial is accurate enough for FP32 attention while avoiding one libm
// call per score.
[[nodiscard]] __m256 exp_f32_avx2(__m256 value) noexcept {
  value = _mm256_min_ps(value, _mm256_set1_ps(88.3762626647949F));
  value = _mm256_max_ps(value, _mm256_set1_ps(-88.3762626647949F));

  __m256 exponent = _mm256_fmadd_ps(
    value, _mm256_set1_ps(1.44269504088896341F), _mm256_set1_ps(0.5F));
  exponent = _mm256_floor_ps(exponent);
  value = _mm256_fnmadd_ps(exponent, _mm256_set1_ps(0.693359375F), value);
  value = _mm256_fnmadd_ps(exponent, _mm256_set1_ps(-2.12194440e-4F), value);

  const __m256 squared = _mm256_mul_ps(value, value);
  __m256 polynomial = _mm256_set1_ps(1.9875691500e-4F);
  polynomial = _mm256_fmadd_ps(polynomial, value, _mm256_set1_ps(1.3981999507e-3F));
  polynomial = _mm256_fmadd_ps(polynomial, value, _mm256_set1_ps(8.3334519073e-3F));
  polynomial = _mm256_fmadd_ps(polynomial, value, _mm256_set1_ps(4.1665795894e-2F));
  polynomial = _mm256_fmadd_ps(polynomial, value, _mm256_set1_ps(1.6666665459e-1F));
  polynomial = _mm256_fmadd_ps(polynomial, value, _mm256_set1_ps(5.0000001201e-1F));
  polynomial = _mm256_fmadd_ps(polynomial, squared, value);
  polynomial = _mm256_add_ps(polynomial, _mm256_set1_ps(1.0F));

  __m256i integer_exponent = _mm256_cvttps_epi32(exponent);
  integer_exponent = _mm256_add_epi32(integer_exponent, _mm256_set1_epi32(127));
  integer_exponent = _mm256_slli_epi32(integer_exponent, 23);
  return _mm256_mul_ps(polynomial, _mm256_castsi256_ps(integer_exponent));
}

[[nodiscard]] float dot_f32_avx2(
  const float * lhs,
  const float * rhs,
  const int count) noexcept {
  __m256 accum0 = _mm256_setzero_ps();
  __m256 accum1 = _mm256_setzero_ps();
  __m256 accum2 = _mm256_setzero_ps();
  __m256 accum3 = _mm256_setzero_ps();
  int index = 0;
  for (; index + 32 <= count; index += 32) {
    accum0 = _mm256_fmadd_ps(
      _mm256_loadu_ps(lhs + index), _mm256_loadu_ps(rhs + index), accum0);
    accum1 = _mm256_fmadd_ps(
      _mm256_loadu_ps(lhs + index + 8), _mm256_loadu_ps(rhs + index + 8), accum1);
    accum2 = _mm256_fmadd_ps(
      _mm256_loadu_ps(lhs + index + 16), _mm256_loadu_ps(rhs + index + 16), accum2);
    accum3 = _mm256_fmadd_ps(
      _mm256_loadu_ps(lhs + index + 24), _mm256_loadu_ps(rhs + index + 24), accum3);
  }
  accum0 = _mm256_add_ps(accum0, accum1);
  accum2 = _mm256_add_ps(accum2, accum3);
  float result = horizontal_sum_f32(_mm256_add_ps(accum0, accum2));
  for (; index < count; ++index) {
    result += lhs[index] * rhs[index];
  }
  return result;
}

[[nodiscard]] float dot_f16_avx2(
  const float * lhs,
  const std::uint16_t * rhs,
  const int count) noexcept {
  __m256 accum0 = _mm256_setzero_ps();
  __m256 accum1 = _mm256_setzero_ps();
  int index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m256 rhs0 = _mm256_cvtph_ps(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhs + index)));
    const __m256 rhs1 = _mm256_cvtph_ps(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhs + index + 8)));
    accum0 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + index), rhs0, accum0);
    accum1 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + index + 8), rhs1, accum1);
  }
  float result = horizontal_sum_f32(_mm256_add_ps(accum0, accum1));
  for (; index < count; ++index) {
    result += lhs[index] * f16c_half_to_float(rhs[index]);
  }
  return result;
}

void dot_pair_f32_avx2(
  const float * lhs0,
  const float * lhs1,
  const float * rhs,
  const int count,
  float & result0,
  float & result1) noexcept {
  __m256 accum00 = _mm256_setzero_ps();
  __m256 accum01 = _mm256_setzero_ps();
  __m256 accum10 = _mm256_setzero_ps();
  __m256 accum11 = _mm256_setzero_ps();
  int index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m256 rhs0 = _mm256_loadu_ps(rhs + index);
    const __m256 rhs1 = _mm256_loadu_ps(rhs + index + 8);
    accum00 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs0 + index), rhs0, accum00);
    accum01 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs0 + index + 8), rhs1, accum01);
    accum10 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs1 + index), rhs0, accum10);
    accum11 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs1 + index + 8), rhs1, accum11);
  }
  result0 = horizontal_sum_f32(_mm256_add_ps(accum00, accum01));
  result1 = horizontal_sum_f32(_mm256_add_ps(accum10, accum11));
  for (; index < count; ++index) {
    result0 += lhs0[index] * rhs[index];
    result1 += lhs1[index] * rhs[index];
  }
}

void dot_pair_f16_avx2(
  const float * lhs0,
  const float * lhs1,
  const std::uint16_t * rhs,
  const int count,
  float & result0,
  float & result1) noexcept {
  __m256 accum00 = _mm256_setzero_ps();
  __m256 accum01 = _mm256_setzero_ps();
  __m256 accum10 = _mm256_setzero_ps();
  __m256 accum11 = _mm256_setzero_ps();
  int index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m256 rhs0 = _mm256_cvtph_ps(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhs + index)));
    const __m256 rhs1 = _mm256_cvtph_ps(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhs + index + 8)));
    accum00 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs0 + index), rhs0, accum00);
    accum01 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs0 + index + 8), rhs1, accum01);
    accum10 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs1 + index), rhs0, accum10);
    accum11 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs1 + index + 8), rhs1, accum11);
  }
  result0 = horizontal_sum_f32(_mm256_add_ps(accum00, accum01));
  result1 = horizontal_sum_f32(_mm256_add_ps(accum10, accum11));
  for (; index < count; ++index) {
    const float value = f16c_half_to_float(rhs[index]);
    result0 += lhs0[index] * value;
    result1 += lhs1[index] * value;
  }
}

void normalize_softmax_avx2(float * scores, const int count, const float maximum) noexcept {
  __m256 denominator_vector = _mm256_setzero_ps();
  const __m256 maximum_vector = _mm256_set1_ps(maximum);
  int index = 0;
  for (; index + 8 <= count; index += 8) {
    const __m256 probability = exp_f32_avx2(
      _mm256_sub_ps(_mm256_loadu_ps(scores + index), maximum_vector));
    _mm256_storeu_ps(scores + index, probability);
    denominator_vector = _mm256_add_ps(denominator_vector, probability);
  }
  float denominator = horizontal_sum_f32(denominator_vector);
  for (; index < count; ++index) {
    const float probability = std::exp(scores[index] - maximum);
    scores[index] = probability;
    denominator += probability;
  }
  const float inverse = 1.0F / (denominator > 0.0F ? denominator : 1.0F);
  const __m256 inverse_vector = _mm256_set1_ps(inverse);
  index = 0;
  for (; index + 8 <= count; index += 8) {
    _mm256_storeu_ps(
      scores + index, _mm256_mul_ps(_mm256_loadu_ps(scores + index), inverse_vector));
  }
  for (; index < count; ++index) {
    scores[index] *= inverse;
  }
}

void weighted_sum_pair_avx2(
  const float * v_cache,
  const std::uint16_t * v_cache_f16,
  const float * probabilities0,
  const float * probabilities1,
  float * output0,
  float * output1,
  const std::size_t kv_width,
  const std::size_t head_offset,
  const int sequence_length,
  const int head_dim,
  const AttentionKvRows *pages) noexcept {
  int column = 0;
  for (; column + 32 <= head_dim; column += 32) {
    __m256 accum00 = _mm256_setzero_ps();
    __m256 accum01 = _mm256_setzero_ps();
    __m256 accum02 = _mm256_setzero_ps();
    __m256 accum03 = _mm256_setzero_ps();
    __m256 accum10 = _mm256_setzero_ps();
    __m256 accum11 = _mm256_setzero_ps();
    __m256 accum12 = _mm256_setzero_ps();
    __m256 accum13 = _mm256_setzero_ps();
    for (int context = 0; context < sequence_length; ++context) {
      const std::size_t offset = static_cast<std::size_t>(context) * kv_width +
        head_offset + static_cast<std::size_t>(column);
      const __m256 value0 = (v_cache_f16 != nullptr || (pages && pages->values))
        ? _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(attention_cache_row(v_cache_f16, pages ? pages->values : nullptr, context, kv_width) + (offset - static_cast<std::size_t>(context) * kv_width))))
        : _mm256_loadu_ps(v_cache + offset);
      const __m256 value1 = (v_cache_f16 != nullptr || (pages && pages->values))
        ? _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(attention_cache_row(v_cache_f16, pages ? pages->values : nullptr, context, kv_width) + (offset - static_cast<std::size_t>(context) * kv_width) + 8)))
        : _mm256_loadu_ps(v_cache + offset + 8);
      const __m256 value2 = (v_cache_f16 != nullptr || (pages && pages->values))
        ? _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(attention_cache_row(v_cache_f16, pages ? pages->values : nullptr, context, kv_width) + (offset - static_cast<std::size_t>(context) * kv_width) + 16)))
        : _mm256_loadu_ps(v_cache + offset + 16);
      const __m256 value3 = (v_cache_f16 != nullptr || (pages && pages->values))
        ? _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(attention_cache_row(v_cache_f16, pages ? pages->values : nullptr, context, kv_width) + (offset - static_cast<std::size_t>(context) * kv_width) + 24)))
        : _mm256_loadu_ps(v_cache + offset + 24);
      const __m256 probability0 = _mm256_set1_ps(probabilities0[context]);
      const __m256 probability1 = _mm256_set1_ps(probabilities1[context]);
      accum00 = _mm256_fmadd_ps(probability0, value0, accum00);
      accum01 = _mm256_fmadd_ps(probability0, value1, accum01);
      accum02 = _mm256_fmadd_ps(probability0, value2, accum02);
      accum03 = _mm256_fmadd_ps(probability0, value3, accum03);
      accum10 = _mm256_fmadd_ps(probability1, value0, accum10);
      accum11 = _mm256_fmadd_ps(probability1, value1, accum11);
      accum12 = _mm256_fmadd_ps(probability1, value2, accum12);
      accum13 = _mm256_fmadd_ps(probability1, value3, accum13);
    }
    _mm256_storeu_ps(output0 + column, accum00);
    _mm256_storeu_ps(output0 + column + 8, accum01);
    _mm256_storeu_ps(output0 + column + 16, accum02);
    _mm256_storeu_ps(output0 + column + 24, accum03);
    _mm256_storeu_ps(output1 + column, accum10);
    _mm256_storeu_ps(output1 + column + 8, accum11);
    _mm256_storeu_ps(output1 + column + 16, accum12);
    _mm256_storeu_ps(output1 + column + 24, accum13);
  }
  for (; column < head_dim; ++column) {
    float accum0 = 0.0F;
    float accum1 = 0.0F;
    for (int context = 0; context < sequence_length; ++context) {
      const std::size_t offset = static_cast<std::size_t>(context) * kv_width +
        head_offset + static_cast<std::size_t>(column);
      const float value = (v_cache_f16 != nullptr || (pages && pages->values))
        ? f16c_half_to_float(attention_cache_row(v_cache_f16, pages ? pages->values : nullptr, context, kv_width)[offset - static_cast<std::size_t>(context) * kv_width]) : v_cache[offset];
      accum0 += probabilities0[context] * value;
      accum1 += probabilities1[context] * value;
    }
    output0[column] = accum0;
    output1[column] = accum1;
  }
}

[[nodiscard]] float stable_sigmoid(float value) noexcept;

void apply_gate_avx2(float * output, const float * gate, const int count) noexcept {
  const __m256 zero = _mm256_setzero_ps();
  const __m256 one = _mm256_set1_ps(1.0F);
  const __m256 sign_mask = _mm256_set1_ps(-0.0F);
  int column = 0;
  for (; column + 8 <= count; column += 8) {
    const __m256 gate_value = _mm256_loadu_ps(gate + column);
    const __m256 absolute = _mm256_andnot_ps(sign_mask, gate_value);
    const __m256 exponential = exp_f32_avx2(_mm256_sub_ps(zero, absolute));
    const __m256 denominator = _mm256_add_ps(one, exponential);
    const __m256 positive = _mm256_div_ps(one, denominator);
    const __m256 negative = _mm256_mul_ps(exponential, positive);
    const __m256 sigmoid = _mm256_blendv_ps(
      negative, positive, _mm256_cmp_ps(gate_value, zero, _CMP_GE_OQ));
    _mm256_storeu_ps(
      output + column, _mm256_mul_ps(_mm256_loadu_ps(output + column), sigmoid));
  }
  for (; column < count; ++column) {
    output[column] *= stable_sigmoid(gate[column]);
  }
}

void weighted_sum_f32_avx2(
  const float * v_cache,
  const std::uint16_t * v_cache_f16,
  const float * probabilities,
  float * output,
  const std::size_t kv_width,
  const std::size_t head_offset,
  const int sequence_length,
  const int head_dim,
  const AttentionKvRows *pages) noexcept {
  int column_base = 0;
  for (; column_base + 64 <= head_dim; column_base += 64) {
    __m256 accum0 = _mm256_setzero_ps();
    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    __m256 accum3 = _mm256_setzero_ps();
    __m256 accum4 = _mm256_setzero_ps();
    __m256 accum5 = _mm256_setzero_ps();
    __m256 accum6 = _mm256_setzero_ps();
    __m256 accum7 = _mm256_setzero_ps();
    for (int context = 0; context < sequence_length; ++context) {
      const __m256 probability = _mm256_set1_ps(
        probabilities[static_cast<std::size_t>(context)]);
      const std::size_t offset = static_cast<std::size_t>(context) * kv_width +
        head_offset + static_cast<std::size_t>(column_base);
      if ((v_cache_f16 != nullptr || (pages && pages->values))) {
        const std::uint16_t * value = attention_cache_row(v_cache_f16, pages ? pages->values : nullptr, context, kv_width) + (offset - static_cast<std::size_t>(context) * kv_width);
        accum0 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value))), accum0);
        accum1 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value + 8))), accum1);
        accum2 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value + 16))), accum2);
        accum3 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value + 24))), accum3);
        accum4 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value + 32))), accum4);
        accum5 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value + 40))), accum5);
        accum6 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value + 48))), accum6);
        accum7 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value + 56))), accum7);
      } else {
        const float * value = v_cache + offset;
        accum0 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value), accum0);
        accum1 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + 8), accum1);
        accum2 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + 16), accum2);
        accum3 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + 24), accum3);
        accum4 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + 32), accum4);
        accum5 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + 40), accum5);
        accum6 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + 48), accum6);
        accum7 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + 56), accum7);
      }
    }
    float * destination = output + column_base;
    _mm256_storeu_ps(destination, accum0);
    _mm256_storeu_ps(destination + 8, accum1);
    _mm256_storeu_ps(destination + 16, accum2);
    _mm256_storeu_ps(destination + 24, accum3);
    _mm256_storeu_ps(destination + 32, accum4);
    _mm256_storeu_ps(destination + 40, accum5);
    _mm256_storeu_ps(destination + 48, accum6);
    _mm256_storeu_ps(destination + 56, accum7);
  }
  for (; column_base + 8 <= head_dim; column_base += 8) {
    __m256 accumulator = _mm256_setzero_ps();
    for (int context = 0; context < sequence_length; ++context) {
      const __m256 probability = _mm256_set1_ps(
        probabilities[static_cast<std::size_t>(context)]);
      const std::size_t offset = static_cast<std::size_t>(context) * kv_width +
        head_offset + static_cast<std::size_t>(column_base);
      const __m256 value = (v_cache_f16 != nullptr || (pages && pages->values))
        ? _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i *>(attention_cache_row(v_cache_f16, pages ? pages->values : nullptr, context, kv_width) + (offset - static_cast<std::size_t>(context) * kv_width))))
        : _mm256_loadu_ps(v_cache + offset);
      accumulator = _mm256_fmadd_ps(probability, value, accumulator);
    }
    _mm256_storeu_ps(output + column_base, accumulator);
  }
  for (; column_base < head_dim; ++column_base) {
    float accumulator = 0.0F;
    for (int context = 0; context < sequence_length; ++context) {
      const std::size_t offset = static_cast<std::size_t>(context) * kv_width + head_offset;
      const float value = (v_cache_f16 != nullptr || (pages && pages->values))
        ? f16c_half_to_float(attention_cache_row(v_cache_f16, pages ? pages->values : nullptr, context, kv_width)[offset - static_cast<std::size_t>(context) * kv_width + static_cast<std::size_t>(column_base)])
        : v_cache[offset + static_cast<std::size_t>(column_base)];
      accumulator += probabilities[static_cast<std::size_t>(context)] * value;
    }
    output[column_base] = accumulator;
  }
}

[[nodiscard]] float stable_sigmoid(const float value) noexcept {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + std::exp(-value));
  }
  const float exp_value = std::exp(value);
  return exp_value / (1.0F + exp_value);
}

} // namespace

void attention_cache_store_f16_avx2(
  const float * input,
  std::uint16_t * output,
  const std::size_t count) noexcept {
  std::size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    _mm_storeu_si128(
      reinterpret_cast<__m128i *>(output + index),
      _mm256_cvtps_ph(_mm256_loadu_ps(input + index), _MM_FROUND_TO_NEAREST_INT));
  }
  for (; index < count; ++index) {
    output[index] = f16c_float_to_half(input[index]);
  }
}

void causal_attention_batch_rows_avx2(
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
    const std::size_t kv_head_offset =
      static_cast<std::size_t>(kv_head) * static_cast<std::size_t>(head_dim);
    const float * query = queries + token * query_width + head_offset;
    const float * gate = gates + token * query_width + head_offset;
    float * score_row = scores + (reuse_score_row ? 0 : row * context_stride);
    float max_score = -std::numeric_limits<float>::infinity();
    for (int context = 0; context < sequence_length; ++context) {
      const std::size_t cache_offset =
        static_cast<std::size_t>(context) * kv_width + kv_head_offset;
      const float score = ((k_cache_f16 != nullptr || (pages && pages->keys))
        ? dot_f16_avx2(query, attention_cache_row(k_cache_f16, pages ? pages->keys : nullptr, context, kv_width) + (cache_offset - static_cast<std::size_t>(context) * kv_width), head_dim)
        : dot_f32_avx2(query, k_cache + cache_offset, head_dim)) * attention_scale;
      score_row[static_cast<std::size_t>(context)] = score;
      max_score = std::max(max_score, score);
    }
    __m256 denominator_vector = _mm256_setzero_ps();
    const __m256 maximum = _mm256_set1_ps(max_score);
    int context = 0;
    for (; context + 8 <= sequence_length; context += 8) {
      const __m256 probability = exp_f32_avx2(
        _mm256_sub_ps(_mm256_loadu_ps(score_row + context), maximum));
      _mm256_storeu_ps(score_row + context, probability);
      denominator_vector = _mm256_add_ps(denominator_vector, probability);
    }
    float denominator = horizontal_sum_f32(denominator_vector);
    for (; context < sequence_length; ++context) {
      const float probability = std::exp(
        score_row[static_cast<std::size_t>(context)] - max_score);
      score_row[static_cast<std::size_t>(context)] = probability;
      denominator += probability;
    }
    const float inverse_denominator = 1.0F / (denominator > 0.0F ? denominator : 1.0F);
    const __m256 inverse_denominator_vector = _mm256_set1_ps(inverse_denominator);
    context = 0;
    for (; context + 8 <= sequence_length; context += 8) {
      _mm256_storeu_ps(
        score_row + context,
        _mm256_mul_ps(
          _mm256_loadu_ps(score_row + context), inverse_denominator_vector));
    }
    for (; context < sequence_length; ++context) {
      score_row[static_cast<std::size_t>(context)] *= inverse_denominator;
    }
    float * output_head = output + token * query_width + head_offset;
    weighted_sum_f32_avx2(
      v_cache, v_cache_f16, score_row, output_head, kv_width, kv_head_offset,
      sequence_length, head_dim, pages);
    int gate_column = 0;
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0F);
    const __m256 sign_mask = _mm256_set1_ps(-0.0F);
    for (; gate_column + 8 <= head_dim; gate_column += 8) {
      const __m256 gate_value = _mm256_loadu_ps(gate + gate_column);
      const __m256 absolute = _mm256_andnot_ps(sign_mask, gate_value);
      const __m256 exponential = exp_f32_avx2(_mm256_sub_ps(zero, absolute));
      const __m256 denominator = _mm256_add_ps(one, exponential);
      const __m256 positive = _mm256_div_ps(one, denominator);
      const __m256 negative = _mm256_mul_ps(exponential, positive);
      const __m256 sigmoid = _mm256_blendv_ps(
        negative, positive, _mm256_cmp_ps(gate_value, zero, _CMP_GE_OQ));
      _mm256_storeu_ps(
        output_head + gate_column,
        _mm256_mul_ps(_mm256_loadu_ps(output_head + gate_column), sigmoid));
    }
    for (; gate_column < head_dim; ++gate_column) {
      output_head[static_cast<std::size_t>(gate_column)] *= stable_sigmoid(gate[gate_column]);
    }
  }
}

void causal_attention_decode_gqa_pairs_avx2(
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
  const AttentionKvRows *pages) noexcept {
  const int heads_per_kv = head_count / kv_head_count;
  for (std::size_t pair = pair_begin; pair < pair_end; ++pair) {
    const int head0 = static_cast<int>(pair * 2U);
    const int head1 = head0 + 1;
    const int kv_head = head0 / heads_per_kv;
    const std::size_t head_offset0 =
      static_cast<std::size_t>(head0) * static_cast<std::size_t>(head_dim);
    const std::size_t head_offset1 = head_offset0 + static_cast<std::size_t>(head_dim);
    const std::size_t kv_head_offset =
      static_cast<std::size_t>(kv_head) * static_cast<std::size_t>(head_dim);
    const float * query0 = queries + head_offset0;
    const float * query1 = queries + head_offset1;
    float * score0 = scores + static_cast<std::size_t>(head0) * context_stride;
    float * score1 = scores + static_cast<std::size_t>(head1) * context_stride;
    float maximum0 = -std::numeric_limits<float>::infinity();
    float maximum1 = -std::numeric_limits<float>::infinity();
    for (int context = 0; context < sequence_length; ++context) {
      const std::size_t offset = static_cast<std::size_t>(context) * kv_width + kv_head_offset;
      float dot0 = 0.0F;
      float dot1 = 0.0F;
      if ((k_cache_f16 != nullptr || (pages && pages->keys))) {
        dot_pair_f16_avx2(query0, query1, attention_cache_row(k_cache_f16, pages ? pages->keys : nullptr, context, kv_width) + (offset - static_cast<std::size_t>(context) * kv_width), head_dim, dot0, dot1);
      } else {
        dot_pair_f32_avx2(query0, query1, k_cache + offset, head_dim, dot0, dot1);
      }
      const float scaled0 = dot0 * attention_scale;
      const float scaled1 = dot1 * attention_scale;
      score0[context] = scaled0;
      score1[context] = scaled1;
      maximum0 = std::max(maximum0, scaled0);
      maximum1 = std::max(maximum1, scaled1);
    }
    normalize_softmax_avx2(score0, sequence_length, maximum0);
    normalize_softmax_avx2(score1, sequence_length, maximum1);
    float * output0 = output + head_offset0;
    float * output1 = output + head_offset1;
    weighted_sum_pair_avx2(
      v_cache, v_cache_f16, score0, score1, output0, output1,
      kv_width, kv_head_offset, sequence_length, head_dim, pages);
    apply_gate_avx2(output0, gates + head_offset0, head_dim);
    apply_gate_avx2(output1, gates + head_offset1, head_dim);
  }
  static_cast<void>(query_width);
}

} // namespace qwen35x::cpu::detail
