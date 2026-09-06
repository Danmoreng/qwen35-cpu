#include "qwen35x/cpu/activation.h"

#include <immintrin.h>

#include <cmath>
#include <cstddef>

namespace qwen35x::cpu::detail {

namespace {

[[nodiscard]] float horizontal_sum_f32(const __m512 value) noexcept {
  return _mm512_reduce_add_ps(value);
}

[[nodiscard]] __m512 exp_f32_avx512(__m512 value) noexcept {
  value = _mm512_min_ps(value, _mm512_set1_ps(88.3762626647949F));
  value = _mm512_max_ps(value, _mm512_set1_ps(-88.3762626647949F));
  __m512 exponent = _mm512_fmadd_ps(
    value, _mm512_set1_ps(1.44269504088896341F), _mm512_set1_ps(0.5F));
  exponent = _mm512_floor_ps(exponent);
  value = _mm512_fnmadd_ps(exponent, _mm512_set1_ps(0.693359375F), value);
  value = _mm512_fnmadd_ps(exponent, _mm512_set1_ps(-2.12194440e-4F), value);
  const __m512 squared = _mm512_mul_ps(value, value);
  __m512 polynomial = _mm512_set1_ps(1.9875691500e-4F);
  polynomial = _mm512_fmadd_ps(polynomial, value, _mm512_set1_ps(1.3981999507e-3F));
  polynomial = _mm512_fmadd_ps(polynomial, value, _mm512_set1_ps(8.3334519073e-3F));
  polynomial = _mm512_fmadd_ps(polynomial, value, _mm512_set1_ps(4.1665795894e-2F));
  polynomial = _mm512_fmadd_ps(polynomial, value, _mm512_set1_ps(1.6666665459e-1F));
  polynomial = _mm512_fmadd_ps(polynomial, value, _mm512_set1_ps(5.0000001201e-1F));
  polynomial = _mm512_fmadd_ps(polynomial, squared, value);
  polynomial = _mm512_add_ps(polynomial, _mm512_set1_ps(1.0F));
  __m512i integer_exponent = _mm512_cvttps_epi32(exponent);
  integer_exponent = _mm512_add_epi32(integer_exponent, _mm512_set1_epi32(127));
  integer_exponent = _mm512_slli_epi32(integer_exponent, 23);
  return _mm512_mul_ps(polynomial, _mm512_castsi512_ps(integer_exponent));
}

} // namespace

void add_f32_avx512(
  const float * lhs,
  const float * rhs,
  float * output,
  const std::size_t count) noexcept {
  std::size_t index = 0;
  for (; index + 64 <= count; index += 64) {
    _mm512_storeu_ps(output + index, _mm512_add_ps(
      _mm512_loadu_ps(lhs + index), _mm512_loadu_ps(rhs + index)));
    _mm512_storeu_ps(output + index + 16, _mm512_add_ps(
      _mm512_loadu_ps(lhs + index + 16), _mm512_loadu_ps(rhs + index + 16)));
    _mm512_storeu_ps(output + index + 32, _mm512_add_ps(
      _mm512_loadu_ps(lhs + index + 32), _mm512_loadu_ps(rhs + index + 32)));
    _mm512_storeu_ps(output + index + 48, _mm512_add_ps(
      _mm512_loadu_ps(lhs + index + 48), _mm512_loadu_ps(rhs + index + 48)));
  }
  for (; index + 16 <= count; index += 16) {
    _mm512_storeu_ps(output + index, _mm512_add_ps(
      _mm512_loadu_ps(lhs + index), _mm512_loadu_ps(rhs + index)));
  }
  for (; index < count; ++index) {
    output[index] = lhs[index] + rhs[index];
  }
}

void rope_f32_avx512(
  float * values,
  const std::size_t head_count,
  const std::size_t head_dim,
  const std::size_t rope_dim,
  const float * cosine,
  const float * sine) noexcept {
  const std::size_t half = rope_dim / 2;
  for (std::size_t head = 0; head < head_count; ++head) {
    float * first = values + head * head_dim;
    float * second = first + half;
    std::size_t index = 0;
    for (; index + 16 <= half; index += 16) {
      const __m512 x0 = _mm512_loadu_ps(first + index);
      const __m512 x1 = _mm512_loadu_ps(second + index);
      const __m512 c = _mm512_loadu_ps(cosine + index);
      const __m512 s = _mm512_loadu_ps(sine + index);
      _mm512_storeu_ps(
        first + index, _mm512_fmsub_ps(x0, c, _mm512_mul_ps(x1, s)));
      _mm512_storeu_ps(
        second + index, _mm512_fmadd_ps(x1, c, _mm512_mul_ps(x0, s)));
    }
    for (; index < half; ++index) {
      const float x0 = first[index];
      const float x1 = second[index];
      first[index] = x0 * cosine[index] - x1 * sine[index];
      second[index] = x1 * cosine[index] + x0 * sine[index];
    }
  }
}

void causal_conv1d_silu_f32_avx512(
  float * state,
  std::size_t ring_index,
  const float * input,
  std::size_t input_stride,
  const float * kernel_major_weights,
  float * output,
  std::size_t batch_size,
  std::size_t channel_count,
  std::size_t kernel_size,
  std::size_t channel_begin,
  std::size_t channel_end) noexcept {
  const std::size_t history = kernel_size - 1;
  const __m512 one = _mm512_set1_ps(1.0F);
  const __m512 zero = _mm512_setzero_ps();
  for (std::size_t token = 0; token < batch_size; ++token) {
    std::size_t channel = channel_begin;
    for (; channel + 16 <= channel_end; channel += 16) {
      const __m512 input_value = _mm512_loadu_ps(
        input + token * input_stride + channel);
      __m512 sum = _mm512_mul_ps(
        input_value,
        _mm512_loadu_ps(kernel_major_weights + history * channel_count + channel));
      for (std::size_t kernel = 0; kernel < history; ++kernel) {
        const std::size_t slot = (ring_index + kernel) % history;
        sum = _mm512_fmadd_ps(
          _mm512_loadu_ps(state + slot * channel_count + channel),
          _mm512_loadu_ps(kernel_major_weights + kernel * channel_count + channel),
          sum);
      }
      if (history != 0) {
        _mm512_storeu_ps(state + ring_index * channel_count + channel, input_value);
      }
      const __m512 sigmoid = _mm512_div_ps(
        one, _mm512_add_ps(one, exp_f32_avx512(_mm512_sub_ps(zero, sum))));
      _mm512_storeu_ps(
        output + token * channel_count + channel, _mm512_mul_ps(sum, sigmoid));
    }
    for (; channel < channel_end; ++channel) {
      const float input_value = input[token * input_stride + channel];
      float sum = input_value *
        kernel_major_weights[history * channel_count + channel];
      for (std::size_t kernel = 0; kernel < history; ++kernel) {
        const std::size_t slot = history == 0 ? 0 : (ring_index + kernel) % history;
        sum += state[slot * channel_count + channel] *
          kernel_major_weights[kernel * channel_count + channel];
      }
      if (history != 0) {
        state[ring_index * channel_count + channel] = input_value;
      }
      const float exponential = std::exp(-std::fabs(sum));
      const float sigmoid = sum >= 0.0F
        ? 1.0F / (1.0F + exponential)
        : exponential / (1.0F + exponential);
      output[token * channel_count + channel] = sum * sigmoid;
    }
    if (history != 0) {
      ring_index = (ring_index + 1) % history;
    }
  }
}

void rms_norm_f32_avx512(
  const float * input,
  const float * weight,
  float * output,
  const std::size_t row_count,
  const std::size_t width,
  const float eps,
  const float weight_offset) noexcept {
  const __m512 offset = _mm512_set1_ps(weight_offset);
  for (std::size_t row = 0; row < row_count; ++row) {
    const float * x = input + row * width;
    float * y = output + row * width;
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    std::size_t column = 0;
    for (; column + 64 <= width; column += 64) {
      const __m512 x0 = _mm512_loadu_ps(x + column);
      const __m512 x1 = _mm512_loadu_ps(x + column + 16);
      const __m512 x2 = _mm512_loadu_ps(x + column + 32);
      const __m512 x3 = _mm512_loadu_ps(x + column + 48);
      sum0 = _mm512_fmadd_ps(x0, x0, sum0);
      sum1 = _mm512_fmadd_ps(x1, x1, sum1);
      sum2 = _mm512_fmadd_ps(x2, x2, sum2);
      sum3 = _mm512_fmadd_ps(x3, x3, sum3);
    }
    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    float squared_sum = horizontal_sum_f32(_mm512_add_ps(sum0, sum2));
    for (; column < width; ++column) {
      squared_sum += x[column] * x[column];
    }
    const float inverse_scalar = 1.0F /
      std::sqrt(squared_sum / static_cast<float>(width) + eps);
    const __m512 inverse = _mm512_set1_ps(inverse_scalar);
    column = 0;
    for (; column + 16 <= width; column += 16) {
      _mm512_storeu_ps(
        y + column,
        _mm512_mul_ps(
          _mm512_mul_ps(_mm512_loadu_ps(x + column), inverse),
          _mm512_add_ps(_mm512_loadu_ps(weight + column), offset)));
    }
    for (; column < width; ++column) {
      y[column] = x[column] * inverse_scalar * (weight[column] + weight_offset);
    }
  }
}

void l2_normalize_f32_avx512(
  float * values,
  const std::size_t row_count,
  const std::size_t width,
  const float eps,
  const float output_scale) noexcept {
  for (std::size_t row = 0; row < row_count; ++row) {
    float * current = values + row * width;
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    std::size_t column = 0;
    for (; column + 32 <= width; column += 32) {
      const __m512 x0 = _mm512_loadu_ps(current + column);
      const __m512 x1 = _mm512_loadu_ps(current + column + 16);
      sum0 = _mm512_fmadd_ps(x0, x0, sum0);
      sum1 = _mm512_fmadd_ps(x1, x1, sum1);
    }
    float squared_sum = horizontal_sum_f32(_mm512_add_ps(sum0, sum1));
    for (; column < width; ++column) {
      squared_sum += current[column] * current[column];
    }
    const float multiplier_scalar = output_scale / std::sqrt(squared_sum + eps);
    const __m512 multiplier = _mm512_set1_ps(multiplier_scalar);
    column = 0;
    for (; column + 16 <= width; column += 16) {
      _mm512_storeu_ps(
        current + column, _mm512_mul_ps(_mm512_loadu_ps(current + column), multiplier));
    }
    for (; column < width; ++column) {
      current[column] *= multiplier_scalar;
    }
  }
}

void silu_f32_avx512(
  const float * input,
  float * output,
  const std::size_t count) noexcept {
  const __m512 one = _mm512_set1_ps(1.0F);
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m512 value = _mm512_loadu_ps(input + index);
    const __m512 sigmoid = _mm512_div_ps(
      one,
      _mm512_add_ps(one, exp_f32_avx512(_mm512_sub_ps(_mm512_setzero_ps(), value))));
    _mm512_storeu_ps(output + index, _mm512_mul_ps(value, sigmoid));
  }
  for (; index < count; ++index) {
    const float value = input[index];
    const float exponential = std::exp(-std::fabs(value));
    const float sigmoid = value >= 0.0F
      ? 1.0F / (1.0F + exponential)
      : exponential / (1.0F + exponential);
    output[index] = value * sigmoid;
  }
}

void silu_mul_f32_avx512(
  const float * gate,
  const float * up,
  float * output,
  const std::size_t count) noexcept {
  const __m512 one = _mm512_set1_ps(1.0F);
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m512 value = _mm512_loadu_ps(gate + index);
    const __m512 sigmoid = _mm512_div_ps(
      one,
      _mm512_add_ps(one, exp_f32_avx512(_mm512_sub_ps(_mm512_setzero_ps(), value))));
    _mm512_storeu_ps(
      output + index,
      _mm512_mul_ps(_mm512_mul_ps(value, sigmoid), _mm512_loadu_ps(up + index)));
  }
  for (; index < count; ++index) {
    const float value = gate[index];
    const float exponential = std::exp(-std::fabs(value));
    const float sigmoid = value >= 0.0F
      ? 1.0F / (1.0F + exponential)
      : exponential / (1.0F + exponential);
    output[index] = value * sigmoid * up[index];
  }
}

} // namespace qwen35x::cpu::detail
