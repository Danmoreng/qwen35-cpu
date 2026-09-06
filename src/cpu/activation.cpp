#include "qwen35x/cpu/activation.h"

#include <cmath>
#include <cstddef>

#ifndef QWEN35X_Q8_0_HAS_AVX2_TU
#define QWEN35X_Q8_0_HAS_AVX2_TU 0
#endif

#ifndef QWEN35X_Q8_0_HAS_AVX512_TU
#define QWEN35X_Q8_0_HAS_AVX512_TU 0
#endif

namespace qwen35x::cpu {

namespace detail {

#if QWEN35X_Q8_0_HAS_AVX512_TU
void add_f32_avx512(
  const float * lhs, const float * rhs, float * output, std::size_t count) noexcept;
void rope_f32_avx512(
  float * values, std::size_t head_count, std::size_t head_dim,
  std::size_t rope_dim, const float * cosine, const float * sine) noexcept;
void causal_conv1d_silu_f32_avx512(
  float * state, std::size_t ring_index, const float * input,
  std::size_t input_stride, const float * kernel_major_weights, float * output,
  std::size_t batch_size, std::size_t channel_count, std::size_t kernel_size,
  std::size_t channel_begin, std::size_t channel_end) noexcept;
void rms_norm_f32_avx512(
  const float * input, const float * weight, float * output,
  std::size_t row_count, std::size_t width, float eps, float weight_offset) noexcept;
void l2_normalize_f32_avx512(
  float * values, std::size_t row_count, std::size_t width,
  float eps, float output_scale) noexcept;
void silu_f32_avx512(
  const float * input, float * output, std::size_t count) noexcept;
void silu_mul_f32_avx512(
  const float * gate, const float * up, float * output, std::size_t count) noexcept;
#endif

#if QWEN35X_Q8_0_HAS_AVX2_TU
void add_f32_avx2(
  const float * lhs,
  const float * rhs,
  float * output,
  std::size_t count) noexcept;

void rope_f32_avx2(
  float * values,
  std::size_t head_count,
  std::size_t head_dim,
  std::size_t rope_dim,
  const float * cosine,
  const float * sine) noexcept;

void causal_conv1d_silu_f32_avx2(
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
  std::size_t channel_end) noexcept;

void rms_norm_f32_avx2(
  const float * input,
  const float * weight,
  float * output,
  std::size_t row_count,
  std::size_t width,
  float eps,
  float weight_offset) noexcept;

void l2_normalize_f32_avx2(
  float * values,
  std::size_t row_count,
  std::size_t width,
  float eps,
  float output_scale) noexcept;

void silu_f32_avx2(
  const float * input,
  float * output,
  std::size_t count) noexcept;

void silu_mul_f32_avx2(
  const float * gate,
  const float * up,
  float * output,
  std::size_t count) noexcept;
#endif

} // namespace detail

void add_f32(
  const float * lhs,
  const float * rhs,
  float * output,
  const std::size_t count,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX512_TU
  if (q8_0_backend_uses_avx512(backend)) {
    detail::add_f32_avx512(lhs, rhs, output, count);
    return;
  }
#endif
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) {
    detail::add_f32_avx2(lhs, rhs, output, count);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  for (std::size_t index = 0; index < count; ++index) {
    output[index] = lhs[index] + rhs[index];
  }
}

void rope_f32(
  float * values,
  const std::size_t head_count,
  const std::size_t head_dim,
  const std::size_t rope_dim,
  const float * cosine,
  const float * sine,
  const Q8_0Backend backend) noexcept {
  if (rope_dim == 0 || rope_dim > head_dim || (rope_dim & 1U) != 0U) {
    return;
  }
#if QWEN35X_Q8_0_HAS_AVX512_TU
  if (q8_0_backend_uses_avx512(backend)) {
    detail::rope_f32_avx512(
      values, head_count, head_dim, rope_dim, cosine, sine);
    return;
  }
#endif
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) {
    detail::rope_f32_avx2(
      values, head_count, head_dim, rope_dim, cosine, sine);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  const std::size_t half = rope_dim / 2;
  for (std::size_t head = 0; head < head_count; ++head) {
    float * first = values + head * head_dim;
    float * second = first + half;
    for (std::size_t index = 0; index < half; ++index) {
      const float x0 = first[index];
      const float x1 = second[index];
      first[index] = x0 * cosine[index] - x1 * sine[index];
      second[index] = x1 * cosine[index] + x0 * sine[index];
    }
  }
}

void causal_conv1d_silu_f32(
  float * state,
  std::size_t ring_index,
  const float * input,
  const std::size_t input_stride,
  const float * kernel_major_weights,
  float * output,
  const std::size_t batch_size,
  const std::size_t channel_count,
  const std::size_t kernel_size,
  const std::size_t channel_begin,
  const std::size_t channel_end,
  const Q8_0Backend backend) noexcept {
  if (kernel_size == 0 || channel_begin >= channel_end) {
    return;
  }
#if QWEN35X_Q8_0_HAS_AVX512_TU
  if (q8_0_backend_uses_avx512(backend)) {
    detail::causal_conv1d_silu_f32_avx512(
      state, ring_index, input, input_stride, kernel_major_weights, output,
      batch_size, channel_count, kernel_size, channel_begin, channel_end);
    return;
  }
#endif
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) {
    detail::causal_conv1d_silu_f32_avx2(
      state, ring_index, input, input_stride, kernel_major_weights, output,
      batch_size, channel_count, kernel_size, channel_begin, channel_end);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  const std::size_t history = kernel_size - 1;
  for (std::size_t token = 0; token < batch_size; ++token) {
    for (std::size_t channel = channel_begin; channel < channel_end; ++channel) {
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

void rms_norm_f32(
  const float * input,
  const float * weight,
  float * output,
  const std::size_t row_count,
  const std::size_t width,
  const float eps,
  const float weight_offset,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX512_TU
  if (q8_0_backend_uses_avx512(backend)) {
    detail::rms_norm_f32_avx512(
      input, weight, output, row_count, width, eps, weight_offset);
    return;
  }
#endif
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) {
    detail::rms_norm_f32_avx2(
      input, weight, output, row_count, width, eps, weight_offset);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  for (std::size_t row = 0; row < row_count; ++row) {
    const float * x = input + row * width;
    float * y = output + row * width;
    float squared_sum = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      squared_sum += x[column] * x[column];
    }
    const float inverse = 1.0F /
      std::sqrt(squared_sum / static_cast<float>(width) + eps);
    for (std::size_t column = 0; column < width; ++column) {
      y[column] = x[column] * inverse * (weight[column] + weight_offset);
    }
  }
}

void l2_normalize_f32(
  float * values,
  const std::size_t row_count,
  const std::size_t width,
  const float eps,
  const float output_scale,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX512_TU
  if (q8_0_backend_uses_avx512(backend)) {
    detail::l2_normalize_f32_avx512(
      values, row_count, width, eps, output_scale);
    return;
  }
#endif
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) {
    detail::l2_normalize_f32_avx2(
      values, row_count, width, eps, output_scale);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  for (std::size_t row = 0; row < row_count; ++row) {
    float * current = values + row * width;
    float squared_sum = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      squared_sum += current[column] * current[column];
    }
    const float multiplier = output_scale / std::sqrt(squared_sum + eps);
    for (std::size_t column = 0; column < width; ++column) {
      current[column] *= multiplier;
    }
  }
}

void silu_f32(
  const float * input,
  float * output,
  const std::size_t count,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX512_TU
  if (q8_0_backend_uses_avx512(backend)) {
    detail::silu_f32_avx512(input, output, count);
    return;
  }
#endif
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) {
    detail::silu_f32_avx2(input, output, count);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  for (std::size_t index = 0; index < count; ++index) {
    const float value = input[index];
    const float exponential = std::exp(-std::fabs(value));
    const float sigmoid = value >= 0.0F
      ? 1.0F / (1.0F + exponential)
      : exponential / (1.0F + exponential);
    output[index] = value * sigmoid;
  }
}

void silu_mul_f32(
  const float * gate,
  const float * up,
  float * output,
  const std::size_t count,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX512_TU
  if (q8_0_backend_uses_avx512(backend)) {
    detail::silu_mul_f32_avx512(gate, up, output, count);
    return;
  }
#endif
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) {
    detail::silu_mul_f32_avx2(gate, up, output, count);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  for (std::size_t index = 0; index < count; ++index) {
    const float value = gate[index];
    const float exponential = std::exp(-std::fabs(value));
    const float sigmoid = value >= 0.0F
      ? 1.0F / (1.0F + exponential)
      : exponential / (1.0F + exponential);
    output[index] = value * sigmoid * up[index];
  }
}

} // namespace qwen35x::cpu
