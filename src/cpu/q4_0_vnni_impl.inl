#include "q4_0_internal.h"

#include "f16c_compat.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace qwen35x::cpu::detail {
namespace {

template <typename Block>
[[nodiscard]] __m256i load_q8_token_half(
  const Block & block,
  const std::size_t token,
  const std::size_t half) noexcept {
  const std::int8_t * base = block.qs + token * 32 + half * 16;
  return _mm256_broadcastsi128_si256(
    _mm_loadu_si128(reinterpret_cast<const __m128i *>(base)));
}

template <std::size_t TokenCount, typename Block>
void accumulate_packed_block_x8_vnni(
  const Q4_0BlockX8 & weights,
  const Block & activations0,
  const std::type_identity_t<Block> * activations1,
  __m256 (&accumulators)[TokenCount]) noexcept {
  static_assert(TokenCount == 1 || TokenCount == 4 || TokenCount == 8);
  const __m256i nibble_mask = _mm256_set1_epi8(0x0f);

  const __m256i raw_0123_0 = _mm256_loadu_si256(
    reinterpret_cast<const __m256i *>(weights.qs));
  const __m256i raw_4567_0 = _mm256_loadu_si256(
    reinterpret_cast<const __m256i *>(weights.qs + 32));
  const __m256i raw_0123_1 = _mm256_loadu_si256(
    reinterpret_cast<const __m256i *>(weights.qs + 64));
  const __m256i raw_4567_1 = _mm256_loadu_si256(
    reinterpret_cast<const __m256i *>(weights.qs + 96));

  const __m256i weight_0123_0 = _mm256_and_si256(raw_0123_0, nibble_mask);
  const __m256i weight_4567_0 = _mm256_and_si256(raw_4567_0, nibble_mask);
  const __m256i weight_0123_1 = _mm256_and_si256(raw_0123_1, nibble_mask);
  const __m256i weight_4567_1 = _mm256_and_si256(raw_4567_1, nibble_mask);
  const __m256i weight_0123_2 = _mm256_and_si256(
    _mm256_srli_epi16(raw_0123_0, 4), nibble_mask);
  const __m256i weight_4567_2 = _mm256_and_si256(
    _mm256_srli_epi16(raw_4567_0, 4), nibble_mask);
  const __m256i weight_0123_3 = _mm256_and_si256(
    _mm256_srli_epi16(raw_0123_1, 4), nibble_mask);
  const __m256i weight_4567_3 = _mm256_and_si256(
    _mm256_srli_epi16(raw_4567_1, 4), nibble_mask);

  // Lanes remain ordered 0,4,1,5,2,6,3,7 until the final permutation.
  const __m256 natural_weight_scales = _mm256_cvtph_ps(_mm_loadu_si128(
    reinterpret_cast<const __m128i *>(weights.d)));
  const __m256 weight_scales = _mm256_permutevar8x32_ps(
    natural_weight_scales,
    _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

  for (std::size_t token = 0; token < TokenCount; ++token) {
    const Block & activation_block = token < 4
      ? activations0 : *activations1;
    const std::size_t activation_lane = token % 4;
    const __m256i activation_0 =
      load_q8_token_half(activation_block, activation_lane, 0);
    const __m256i activation_1 =
      load_q8_token_half(activation_block, activation_lane, 1);
    __m256i integer_dot = _mm256_setzero_si256();
    integer_dot = QWEN35X_DPBUSD(
      integer_dot,
      _mm256_blend_epi32(weight_0123_0, _mm256_shuffle_epi32(weight_4567_0, 177), 170),
      _mm256_shuffle_epi32(activation_0, 0));
    integer_dot = QWEN35X_DPBUSD(
      integer_dot,
      _mm256_blend_epi32(_mm256_shuffle_epi32(weight_0123_0, 177), weight_4567_0, 170),
      _mm256_shuffle_epi32(activation_0, 85));
    integer_dot = QWEN35X_DPBUSD(
      integer_dot,
      _mm256_blend_epi32(weight_0123_1, _mm256_shuffle_epi32(weight_4567_1, 177), 170),
      _mm256_shuffle_epi32(activation_0, 170));
    integer_dot = QWEN35X_DPBUSD(
      integer_dot,
      _mm256_blend_epi32(_mm256_shuffle_epi32(weight_0123_1, 177), weight_4567_1, 170),
      _mm256_shuffle_epi32(activation_0, 255));
    integer_dot = QWEN35X_DPBUSD(
      integer_dot,
      _mm256_blend_epi32(weight_0123_2, _mm256_shuffle_epi32(weight_4567_2, 177), 170),
      _mm256_shuffle_epi32(activation_1, 0));
    integer_dot = QWEN35X_DPBUSD(
      integer_dot,
      _mm256_blend_epi32(_mm256_shuffle_epi32(weight_0123_2, 177), weight_4567_2, 170),
      _mm256_shuffle_epi32(activation_1, 85));
    integer_dot = QWEN35X_DPBUSD(
      integer_dot,
      _mm256_blend_epi32(weight_0123_3, _mm256_shuffle_epi32(weight_4567_3, 177), 170),
      _mm256_shuffle_epi32(activation_1, 170));
    integer_dot = QWEN35X_DPBUSD(
      integer_dot,
      _mm256_blend_epi32(_mm256_shuffle_epi32(weight_0123_3, 177), weight_4567_3, 170),
      _mm256_shuffle_epi32(activation_1, 255));

    integer_dot = _mm256_sub_epi32(
      integer_dot,
      _mm256_set1_epi32(
        8 * static_cast<std::int32_t>(activation_block.sums[activation_lane])));
    const __m256 scale = _mm256_mul_ps(
      weight_scales,
      _mm256_set1_ps(activation_block.scales[activation_lane]));
    accumulators[token] = _mm256_fmadd_ps(
      _mm256_cvtepi32_ps(integer_dot), scale, accumulators[token]);
  }
}

[[nodiscard]] __m256 final_row_order(const __m256 value) noexcept {
  return _mm256_permutevar8x32_ps(
    value, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
}

} // namespace

void QWEN35X_VNNI_FN(q4_0_packed_matmul_q8_0)(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX4 * vectors,
  float * output,
  const std::size_t row_count,
  const std::size_t vector_count,
  const std::size_t blocks_per_row,
  const std::size_t output_row_stride) noexcept {
  const std::size_t row_tiles = row_count / q4_0_packed_rows;
  const std::size_t vector_tiles = vector_count / q8_0_packed_vectors;
  std::size_t vector_tile = 0;
  for (; vector_tile + 2 <= vector_tiles; vector_tile += 2) {
    const Q8_0BlockX4 * vector_tile_data = vectors + vector_tile * blocks_per_row;
    for (std::size_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
      __m256 accumulators[8] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
      };
      const Q4_0BlockX8 * row_tile_data = matrix + row_tile * blocks_per_row;
      for (std::size_t block = 0; block < blocks_per_row; ++block) {
        accumulate_packed_block_x8_vnni<8>(
          row_tile_data[block], vector_tile_data[block],
          vector_tile_data + blocks_per_row + block, accumulators);
      }
      for (std::size_t token = 0; token < 8; ++token) {
        _mm256_storeu_ps(
          output + (vector_tile * 4 + token) * output_row_stride + row_tile * 8,
          final_row_order(accumulators[token]));
      }
    }
  }
  for (; vector_tile < vector_tiles; ++vector_tile) {
    const Q8_0BlockX4 * vector_tile_data = vectors + vector_tile * blocks_per_row;
    for (std::size_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
      __m256 accumulators[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
      };
      const Q4_0BlockX8 * row_tile_data = matrix + row_tile * blocks_per_row;
      for (std::size_t block = 0; block < blocks_per_row; ++block) {
        accumulate_packed_block_x8_vnni<4>(
          row_tile_data[block], vector_tile_data[block], nullptr, accumulators);
      }
      for (std::size_t token = 0; token < 4; ++token) {
        _mm256_storeu_ps(
          output + (vector_tile * 4 + token) * output_row_stride + row_tile * 8,
          final_row_order(accumulators[token]));
      }
    }
  }
}

void QWEN35X_VNNI_FN(q4_0_packed_matvec_q8_0)(
  const Q4_0BlockX8 * matrix,
  const Q8_0Block * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row) noexcept {
  for (std::size_t row_tile = 0; row_tile < row_count / 8; ++row_tile) {
    __m256 accumulator[1] = {_mm256_setzero_ps()};
    const Q4_0BlockX8 * row_tile_data = matrix + row_tile * blocks_per_row;
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      Q8_0BlockX4 packed_activation{};
      packed_activation.scales[0] = f16c_half_to_float(vector[block].d);
      std::int32_t activation_sum = 0;
      for (std::size_t index = 0; index < 32; ++index) {
        packed_activation.qs[index] = vector[block].qs[index];
        activation_sum += vector[block].qs[index];
      }
      packed_activation.sums[0] = static_cast<std::int16_t>(activation_sum);
      accumulate_packed_block_x8_vnni<1>(
        row_tile_data[block], packed_activation, nullptr, accumulator);
    }
    _mm256_storeu_ps(output + row_tile * 8, final_row_order(accumulator[0]));
  }
}

void QWEN35X_VNNI_FN(q4_0_packed_matvec_prepared_q8_0)(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX1 * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row) noexcept {
  for (std::size_t row_tile = 0; row_tile < row_count / 8; ++row_tile) {
    __m256 accumulator[1] = {_mm256_setzero_ps()};
    const Q4_0BlockX8 * row_tile_data = matrix + row_tile * blocks_per_row;
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      accumulate_packed_block_x8_vnni<1>(
        row_tile_data[block], vector[block], nullptr, accumulator);
    }
    _mm256_storeu_ps(output + row_tile * 8, final_row_order(accumulator[0]));
  }
}

Q4_0ArgmaxResult QWEN35X_VNNI_FN(q4_0_packed_matvec_prepared_q8_0_argmax)(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX1 * vector,
  const int * token_counts,
  const float repetition_penalty,
  const std::size_t row_offset,
  const std::size_t row_count,
  const std::size_t blocks_per_row) noexcept {
  Q4_0ArgmaxResult best{-std::numeric_limits<float>::infinity(), row_offset};
  alignas(32) float logits[8];
  for (std::size_t row_tile = 0; row_tile < row_count / 8; ++row_tile) {
    __m256 accumulator[1] = {_mm256_setzero_ps()};
    const Q4_0BlockX8 * row_tile_data = matrix + row_tile * blocks_per_row;
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      accumulate_packed_block_x8_vnni<1>(
        row_tile_data[block], vector[block], nullptr, accumulator);
    }
    _mm256_store_ps(logits, final_row_order(accumulator[0]));
    for (std::size_t lane = 0; lane < 8; ++lane) {
      const std::size_t index = row_offset + row_tile * 8 + lane;
      float value = logits[lane];
      if (token_counts != nullptr && token_counts[index] > 0 &&
          repetition_penalty > 1.0F) {
        value = value > 0.0F
          ? value / repetition_penalty
          : value * repetition_penalty;
      }
      if (value > best.value) {
        best = Q4_0ArgmaxResult{value, index};
      }
    }
  }
  return best;
}

} // namespace qwen35x::cpu::detail
