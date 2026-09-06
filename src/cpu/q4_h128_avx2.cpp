#include "q4_h128_internal.h"

#include "qwen35x/cpu/q4_h128.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace qwen35x::cpu::detail {

void q4_h128_transform_block_avx2_unscaled(
  const float * input,
  float * output,
  const std::size_t transform_block_index,
  const std::uint64_t sign_seed) noexcept {
  const std::uint64_t sign_words[2] = {
    q4_h128_sign_word(transform_block_index, 0, sign_seed),
    q4_h128_sign_word(transform_block_index, 1, sign_seed),
  };
  q4_h128_transform_block_avx2_signed(input, output, sign_words);
}

void q4_h128_transform_block_avx2_signed(
  const float * input, float * output, const std::uint64_t * sign_words) noexcept {
  const __m256i sign_shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  for (std::size_t vector = 0; vector < 16; ++vector) {
    const std::size_t value_base = vector * 8;
    const std::size_t word = value_base / 64;
    const std::size_t bit_base = value_base % 64;
    const std::uint32_t signs = static_cast<std::uint32_t>(
      (sign_words[word] >> bit_base) & UINT64_C(0xff));
    const __m256i sign_mask = _mm256_slli_epi32(
      _mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(signs)), sign_shifts),
      31);
    __m256 values = _mm256_xor_ps(
      _mm256_loadu_ps(input + value_base),
      _mm256_castsi256_ps(sign_mask));

    __m256 shuffled = _mm256_permute_ps(values, 0xb1);
    __m256 sums = _mm256_add_ps(values, shuffled);
    __m256 differences = _mm256_sub_ps(values, shuffled);
    values = _mm256_blend_ps(
      sums, _mm256_permute_ps(differences, 0xb1), 0xaa);

    shuffled = _mm256_permute_ps(values, 0x4e);
    sums = _mm256_add_ps(values, shuffled);
    differences = _mm256_sub_ps(values, shuffled);
    values = _mm256_blend_ps(
      sums, _mm256_permute_ps(differences, 0x4e), 0xcc);

    shuffled = _mm256_permute2f128_ps(values, values, 0x01);
    sums = _mm256_add_ps(values, shuffled);
    differences = _mm256_sub_ps(values, shuffled);
    values = _mm256_blend_ps(
      sums, _mm256_permute2f128_ps(differences, differences, 0x01), 0xf0);
    _mm256_storeu_ps(output + value_base, values);
  }

  for (std::size_t stride = 8; stride < q4_h128_transform_size; stride *= 2) {
    for (std::size_t base = 0; base < q4_h128_transform_size; base += 2 * stride) {
      for (std::size_t lane = 0; lane < stride; lane += 8) {
        const __m256 lhs = _mm256_loadu_ps(output + base + lane);
        const __m256 rhs = _mm256_loadu_ps(output + base + stride + lane);
        _mm256_storeu_ps(output + base + lane, _mm256_add_ps(lhs, rhs));
        _mm256_storeu_ps(output + base + stride + lane, _mm256_sub_ps(lhs, rhs));
      }
    }
  }
}

void q4_h128_transform_block_avx2(
  const float * input,
  float * output,
  const std::size_t transform_block_index,
  const std::uint64_t sign_seed) noexcept {
  q4_h128_transform_block_avx2_unscaled(
    input, output, transform_block_index, sign_seed);
  const __m256 scale = _mm256_set1_ps(q4_h128_inverse_sqrt_size);
  for (std::size_t index = 0; index < q4_h128_transform_size; index += 8) {
    _mm256_storeu_ps(
      output + index,
      _mm256_mul_ps(_mm256_loadu_ps(output + index), scale));
  }
}

} // namespace qwen35x::cpu::detail
