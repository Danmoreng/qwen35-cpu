#include "qwen35x/cpu/q4_h128.h"

#include "q4_h128_internal.h"
#include "q4_0_internal.h"
#include "q8_0_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace qwen35x::cpu {
namespace detail {

constexpr std::uint64_t kBlockStride = UINT64_C(0x9e3779b97f4a7c15);
constexpr std::uint64_t kWordStride = UINT64_C(0xd1b54a32d192ed03);

[[nodiscard]] std::uint64_t splitmix64(std::uint64_t value) noexcept {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

std::uint64_t q4_h128_sign_word(
  const std::size_t transform_block_index,
  const std::size_t word_index,
  const std::uint64_t sign_seed) noexcept {
  const std::uint64_t block_key =
    sign_seed + static_cast<std::uint64_t>(transform_block_index) * kBlockStride;
  return splitmix64(block_key + static_cast<std::uint64_t>(word_index) * kWordStride);
}

void apply_signs(
  float * values,
  const std::size_t transform_block_index,
  const std::uint64_t sign_seed) noexcept {
  for (std::size_t word = 0; word < 2; ++word) {
    const std::uint64_t signs = q4_h128_sign_word(
      transform_block_index, word, sign_seed);
    for (std::size_t bit = 0; bit < 64; ++bit) {
      if (((signs >> bit) & 1U) != 0U) {
        values[word * 64 + bit] = -values[word * 64 + bit];
      }
    }
  }
}

void hadamard_128_unscaled_inplace(float * values) noexcept {
  for (std::size_t stride = 1; stride < q4_h128_transform_size; stride *= 2) {
    for (std::size_t base = 0; base < q4_h128_transform_size; base += 2 * stride) {
      for (std::size_t lane = 0; lane < stride; ++lane) {
        const float lhs = values[base + lane];
        const float rhs = values[base + stride + lane];
        values[base + lane] = lhs + rhs;
        values[base + stride + lane] = lhs - rhs;
      }
    }
  }
}

void q4_h128_transform_block_scalar(
  const float * input,
  float * output,
  const std::size_t transform_block_index,
  const std::uint64_t sign_seed) noexcept {
  std::copy_n(input, q4_h128_transform_size, output);
  apply_signs(output, transform_block_index, sign_seed);
  hadamard_128_unscaled_inplace(output);
  for (std::size_t index = 0; index < q4_h128_transform_size; ++index) {
    output[index] *= q4_h128_inverse_sqrt_size;
  }
}

void q4_h128_transform_block_scalar_unscaled(
  const float * input,
  float * output,
  const std::size_t transform_block_index,
  const std::uint64_t sign_seed) noexcept {
  std::copy_n(input, q4_h128_transform_size, output);
  apply_signs(output, transform_block_index, sign_seed);
  hadamard_128_unscaled_inplace(output);
}

} // namespace detail

void q4_h128_transform_block(
  const float * input,
  float * output,
  const std::size_t transform_block_index,
  const std::uint64_t sign_seed,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) {
    detail::q4_h128_transform_block_avx2(
      input, output, transform_block_index, sign_seed);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  detail::q4_h128_transform_block_scalar(
    input, output, transform_block_index, sign_seed);
}

void q4_h128_transform_block_inplace(
  float * values,
  const std::size_t transform_block_index,
  const std::uint64_t sign_seed,
  const Q8_0Backend backend) noexcept {
  alignas(32) float transformed[q4_h128_transform_size];
  q4_h128_transform_block(
    values, transformed, transform_block_index, sign_seed, backend);
  std::copy_n(transformed, q4_h128_transform_size, values);
}

bool q4_h128_transform_rows(
  const float * input,
  float * output,
  const std::size_t row_count,
  const std::size_t column_count,
  const std::uint64_t sign_seed,
  const Q8_0Backend backend) noexcept {
  if (input == nullptr || output == nullptr || column_count == 0 ||
      column_count % q4_h128_transform_size != 0) {
    return false;
  }
  const std::size_t transform_blocks = column_count / q4_h128_transform_size;
  for (std::size_t row = 0; row < row_count; ++row) {
    for (std::size_t block = 0; block < transform_blocks; ++block) {
      const std::size_t offset = row * column_count + block * q4_h128_transform_size;
      q4_h128_transform_block(
        input + offset, output + offset, block, sign_seed, backend);
    }
  }
  return true;
}

bool q4_h128_transform_rows_unscaled(
  const float * input,
  float * output,
  const std::size_t row_count,
  const std::size_t column_count,
  const std::uint64_t sign_seed,
  const Q8_0Backend backend) noexcept {
  if (input == nullptr || output == nullptr || column_count == 0 ||
      column_count % q4_h128_transform_size != 0) {
    return false;
  }
  const std::size_t transform_blocks = column_count / q4_h128_transform_size;
  for (std::size_t row = 0; row < row_count; ++row) {
    for (std::size_t block = 0; block < transform_blocks; ++block) {
      const std::size_t offset = row * column_count + block * q4_h128_transform_size;
#if QWEN35X_Q8_0_HAS_AVX2_TU
      if (q8_0_backend_uses_avx2(backend)) {
        detail::q4_h128_transform_block_avx2_unscaled(
          input + offset, output + offset, block, sign_seed);
        continue;
      }
#else
      static_cast<void>(backend);
#endif
      detail::q4_h128_transform_block_scalar_unscaled(
        input + offset, output + offset, block, sign_seed);
    }
  }
  return true;
}

void q4_h128_quantize_transformed(
  const float * input,
  Q4_0Block * output,
  const std::size_t q4_block_count) noexcept {
  for (std::size_t block = 0; block < q4_block_count; ++block) {
    const float * values = input + block * q4_0_values_per_block;
    float absolute_max = 0.0F;
    for (std::size_t index = 0; index < q4_0_values_per_block; ++index) {
      absolute_max = std::max(absolute_max, std::fabs(values[index]));
    }
    const float scale = absolute_max / 7.0F;
    output[block].d = detail::float_to_half(scale);
    const float stored_scale = detail::half_to_float(output[block].d);
    const float inverse_scale = stored_scale == 0.0F ? 0.0F : 1.0F / stored_scale;
    for (std::size_t index = 0; index < q4_0_values_per_block / 2; ++index) {
      const int low = static_cast<int>(std::clamp(
        std::round(values[index] * inverse_scale), -7.0F, 7.0F));
      const int high = static_cast<int>(std::clamp(
        std::round(values[index + 16] * inverse_scale), -7.0F, 7.0F));
      output[block].qs[index] = static_cast<std::uint8_t>(
        (low + 8) | ((high + 8) << 4));
    }
  }
}

bool q4_h128_quantize_matrix(
  const float * input,
  Q4_0Block * output,
  const std::size_t row_count,
  const std::size_t column_count,
  const std::uint64_t sign_seed) noexcept {
  if (input == nullptr || output == nullptr || column_count == 0 ||
      column_count % q4_h128_transform_size != 0) {
    return false;
  }
  alignas(32) float transformed[q4_h128_transform_size];
  const std::size_t transform_blocks = column_count / q4_h128_transform_size;
  const std::size_t q4_blocks_per_row = column_count / q4_0_values_per_block;
  for (std::size_t row = 0; row < row_count; ++row) {
    for (std::size_t block = 0; block < transform_blocks; ++block) {
      const std::size_t input_offset =
        row * column_count + block * q4_h128_transform_size;
      q4_h128_transform_block(
        input + input_offset, transformed, block, sign_seed);
      q4_h128_quantize_transformed(
        transformed,
        output + row * q4_blocks_per_row + block * q4_h128_q4_blocks_per_transform,
        q4_h128_q4_blocks_per_transform);
    }
  }
  return true;
}

bool q4_h128_prepare_activation(
  const float * input,
  float * transformed_scratch,
  Q8_0Block * output,
  const std::size_t column_count,
  const std::uint64_t sign_seed,
  const Q8_0Backend backend) noexcept {
  if (!q4_h128_transform_rows(
        input, transformed_scratch, 1, column_count, sign_seed, backend)) {
    return false;
  }
  q8_0_quantize(
    transformed_scratch,
    output,
    column_count / q8_0_values_per_block,
    backend);
  return true;
}

void q4_h128_prepare_signs(Q4H128SignBlock * output, const std::size_t block_count,
                          const std::uint64_t sign_seed) noexcept {
  for (std::size_t block = 0; block < block_count; ++block) {
    output[block].words[0] = detail::q4_h128_sign_word(block, 0, sign_seed);
    output[block].words[1] = detail::q4_h128_sign_word(block, 1, sign_seed);
  }
}

bool q4_h128_prepare_activation_1(
  const float * input, Q8_0BlockX1 * output, const std::size_t column_count,
  const std::uint64_t sign_seed, const Q8_0Backend backend,
  const Q4H128SignBlock * signs) noexcept {
  if (input == nullptr || output == nullptr || column_count == 0 ||
      column_count % q4_h128_transform_size != 0) {
    return false;
  }
  [[maybe_unused]] const bool avx2 = q8_0_backend_uses_avx2(backend);
  [[maybe_unused]] const bool avx512 = q8_0_backend_uses_avx512(backend);
  alignas(32) float transformed[q4_h128_transform_size];
  for (std::size_t block = 0; block < column_count / q4_h128_transform_size; ++block) {
    auto * destination = output + block * q4_h128_q4_blocks_per_transform;
#if QWEN35X_Q8_0_HAS_AVX2_TU
    if (avx2) {
#if QWEN35X_Q8_0_HAS_AVX512_TU
      if (avx512) {
        Q4H128SignBlock local_signs;
        if (signs == nullptr) {
          local_signs.words[0] = detail::q4_h128_sign_word(block, 0, sign_seed);
          local_signs.words[1] = detail::q4_h128_sign_word(block, 1, sign_seed);
        }
        detail::q4_h128_transform_block_avx512_signed(
          input + block * q4_h128_transform_size, transformed,
          signs == nullptr ? local_signs.words : signs[block].words);
      } else
#endif
      if (signs != nullptr) {
        detail::q4_h128_transform_block_avx2_signed(
          input + block * q4_h128_transform_size, transformed, signs[block].words);
      } else {
        detail::q4_h128_transform_block_avx2_unscaled(
          input + block * q4_h128_transform_size, transformed, block, sign_seed);
      }
      detail::q8_0_quantize_vector_1_avx2(
        transformed, destination, q4_h128_q4_blocks_per_transform);
    } else
#endif
    {
      detail::q4_h128_transform_block_scalar_unscaled(
        input + block * q4_h128_transform_size, transformed, block, sign_seed);
      detail::q8_0_quantize_vector_1_scalar(
        transformed, destination, q4_h128_q4_blocks_per_transform);
    }
    for (std::size_t group = 0; group < q4_h128_q4_blocks_per_transform; ++group) {
      destination[group].scales[0] *= q4_h128_inverse_sqrt_size;
    }
  }
  return true;
}

bool q4_h128_prepare_activations_4(
  const float * input,
  Q8_0BlockX4 * output,
  const std::size_t vector_count,
  const std::size_t column_count,
  const std::uint64_t sign_seed,
  const Q8_0Backend backend, const Q4H128SignBlock * signs) noexcept {
  if (input == nullptr || output == nullptr || vector_count == 0 ||
      vector_count % q8_0_packed_vectors != 0 || column_count == 0 ||
      column_count % q4_h128_transform_size != 0) {
    return false;
  }
  // Full-model trials favor keeping the four-token prefill transform on AVX2.
  [[maybe_unused]] const bool avx2 = q8_0_backend_uses_avx2(backend);
  const std::size_t blocks_per_vector = column_count / q8_0_values_per_block;
  const std::size_t transform_blocks = column_count / q4_h128_transform_size;
  alignas(32) float transformed[
    q8_0_packed_vectors * q4_h128_transform_size];
  for (std::size_t vector_tile = 0;
       vector_tile < vector_count / q8_0_packed_vectors;
       ++vector_tile) {
    for (std::size_t transform_block = 0;
         transform_block < transform_blocks;
         ++transform_block) {
      Q4H128SignBlock local_signs;
      const Q4H128SignBlock * block_signs = signs == nullptr ? &local_signs : signs + transform_block;
      if (signs == nullptr) {
        local_signs.words[0] = detail::q4_h128_sign_word(transform_block, 0, sign_seed);
        local_signs.words[1] = detail::q4_h128_sign_word(transform_block, 1, sign_seed);
      }
      for (std::size_t token = 0; token < q8_0_packed_vectors; ++token) {
        const std::size_t vector = vector_tile * q8_0_packed_vectors + token;
#if QWEN35X_Q8_0_HAS_AVX2_TU
        if (avx2) {
          detail::q4_h128_transform_block_avx2_signed(
            input + vector * column_count +
              transform_block * q4_h128_transform_size,
            transformed + token * q4_h128_transform_size,
            block_signs->words);
          continue;
        }
#endif
        detail::q4_h128_transform_block_scalar_unscaled(
          input + vector * column_count +
            transform_block * q4_h128_transform_size,
          transformed + token * q4_h128_transform_size,
          transform_block,
          sign_seed);
      }
      Q8_0BlockX4 * const packed_transform =
        output + vector_tile * blocks_per_vector +
        transform_block * q4_h128_q4_blocks_per_transform;
      q8_0_quantize_vectors_4(
        transformed,
        packed_transform,
        q8_0_packed_vectors,
        q4_h128_q4_blocks_per_transform,
        backend);
      for (std::size_t block = 0;
           block < q4_h128_q4_blocks_per_transform;
           ++block) {
        for (float & scale : packed_transform[block].scales) {
          scale *= q4_h128_inverse_sqrt_size;
        }
      }
    }
  }
  return true;
}

} // namespace qwen35x::cpu
