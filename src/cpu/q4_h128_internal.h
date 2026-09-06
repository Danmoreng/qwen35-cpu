#pragma once

#include <cstddef>
#include <cstdint>

#ifndef QWEN35X_Q8_0_HAS_AVX2_TU
#define QWEN35X_Q8_0_HAS_AVX2_TU 0
#endif

#ifndef QWEN35X_Q8_0_HAS_AVX512_TU
#define QWEN35X_Q8_0_HAS_AVX512_TU 0
#endif

namespace qwen35x::cpu::detail {

[[nodiscard]] std::uint64_t q4_h128_sign_word(
  std::size_t transform_block_index,
  std::size_t word_index,
  std::uint64_t sign_seed) noexcept;

void q4_h128_transform_block_scalar(
  const float * input,
  float * output,
  std::size_t transform_block_index,
  std::uint64_t sign_seed) noexcept;

void q4_h128_transform_block_scalar_unscaled(
  const float * input,
  float * output,
  std::size_t transform_block_index,
  std::uint64_t sign_seed) noexcept;

#if QWEN35X_Q8_0_HAS_AVX2_TU
void q4_h128_transform_block_avx2_signed(
  const float * input, float * output, const std::uint64_t * sign_words) noexcept;
void q4_h128_transform_block_avx2(
  const float * input,
  float * output,
  std::size_t transform_block_index,
  std::uint64_t sign_seed) noexcept;
void q4_h128_transform_block_avx2_unscaled(
  const float * input,
  float * output,
  std::size_t transform_block_index,
  std::uint64_t sign_seed) noexcept;
#endif

#if QWEN35X_Q8_0_HAS_AVX512_TU
void q4_h128_transform_block_avx512_signed(
  const float * input, float * output, const std::uint64_t * sign_words) noexcept;
#endif

} // namespace qwen35x::cpu::detail
