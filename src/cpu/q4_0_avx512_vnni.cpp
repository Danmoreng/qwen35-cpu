#include "q4_0_internal.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace qwen35x::cpu::detail {
namespace {

#if defined(_MSC_VER)
#define QWEN35X_FORCE_INLINE __forceinline
#else
#define QWEN35X_FORCE_INLINE inline __attribute__((always_inline))
#endif

struct DecodedWeights {
  __m256i values[8];
  __m256 scales;
};

[[nodiscard]] QWEN35X_FORCE_INLINE DecodedWeights decode_weights(
  const Q4_0BlockX8 & weights) noexcept {
  const __m256i nibble_mask = _mm256_set1_epi8(0x0f);
  const __m256i raw_0123_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights.qs));
  const __m256i raw_4567_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights.qs + 32));
  const __m256i raw_0123_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights.qs + 64));
  const __m256i raw_4567_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights.qs + 96));

  DecodedWeights decoded{};
  decoded.values[0] = _mm256_and_si256(raw_0123_0, nibble_mask);
  decoded.values[1] = _mm256_and_si256(raw_4567_0, nibble_mask);
  decoded.values[2] = _mm256_and_si256(raw_0123_1, nibble_mask);
  decoded.values[3] = _mm256_and_si256(raw_4567_1, nibble_mask);
  decoded.values[4] = _mm256_and_si256(
    _mm256_srli_epi16(raw_0123_0, 4), nibble_mask);
  decoded.values[5] = _mm256_and_si256(
    _mm256_srli_epi16(raw_4567_0, 4), nibble_mask);
  decoded.values[6] = _mm256_and_si256(
    _mm256_srli_epi16(raw_0123_1, 4), nibble_mask);
  decoded.values[7] = _mm256_and_si256(
    _mm256_srli_epi16(raw_4567_1, 4), nibble_mask);

  const __m256 natural_scales = _mm256_cvtph_ps(_mm_loadu_si128(
    reinterpret_cast<const __m128i *>(weights.d)));
  decoded.scales = _mm256_permutevar8x32_ps(
    natural_scales, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
  return decoded;
}

template <bool ShuffleFirst>
[[nodiscard]] QWEN35X_FORCE_INLINE __m256i combined_rows(
  const __m256i rows_0123,
  const __m256i rows_4567) noexcept {
  if constexpr (ShuffleFirst) {
    return _mm256_blend_epi32(
      _mm256_shuffle_epi32(rows_0123, 177), rows_4567, 170);
  } else {
    return _mm256_blend_epi32(
      rows_0123, _mm256_shuffle_epi32(rows_4567, 177), 170);
  }
}

template <std::size_t Lane>
QWEN35X_FORCE_INLINE void accumulate_token(
  const DecodedWeights & weights,
  const Q8_0BlockX4 & activations,
  __m256 & accumulator) noexcept {
  static_assert(Lane < 4);
  const std::int8_t * activation = activations.qs + Lane * 32;
  const __m256i activation_0 = _mm256_broadcastsi128_si256(
    _mm_loadu_si128(reinterpret_cast<const __m128i *>(activation)));
  const __m256i activation_1 = _mm256_broadcastsi128_si256(
    _mm_loadu_si128(reinterpret_cast<const __m128i *>(activation + 16)));

  __m256i integer_dot = _mm256_setzero_si256();
  integer_dot = _mm256_dpbusd_epi32(
    integer_dot, combined_rows<false>(weights.values[0], weights.values[1]),
    _mm256_shuffle_epi32(activation_0, 0));
  integer_dot = _mm256_dpbusd_epi32(
    integer_dot, combined_rows<true>(weights.values[0], weights.values[1]),
    _mm256_shuffle_epi32(activation_0, 85));
  integer_dot = _mm256_dpbusd_epi32(
    integer_dot, combined_rows<false>(weights.values[2], weights.values[3]),
    _mm256_shuffle_epi32(activation_0, 170));
  integer_dot = _mm256_dpbusd_epi32(
    integer_dot, combined_rows<true>(weights.values[2], weights.values[3]),
    _mm256_shuffle_epi32(activation_0, 255));
  integer_dot = _mm256_dpbusd_epi32(
    integer_dot, combined_rows<false>(weights.values[4], weights.values[5]),
    _mm256_shuffle_epi32(activation_1, 0));
  integer_dot = _mm256_dpbusd_epi32(
    integer_dot, combined_rows<true>(weights.values[4], weights.values[5]),
    _mm256_shuffle_epi32(activation_1, 85));
  integer_dot = _mm256_dpbusd_epi32(
    integer_dot, combined_rows<false>(weights.values[6], weights.values[7]),
    _mm256_shuffle_epi32(activation_1, 170));
  integer_dot = _mm256_dpbusd_epi32(
    integer_dot, combined_rows<true>(weights.values[6], weights.values[7]),
    _mm256_shuffle_epi32(activation_1, 255));

  integer_dot = _mm256_sub_epi32(
    integer_dot,
    _mm256_set1_epi32(8 * static_cast<std::int32_t>(activations.sums[Lane])));
  const __m256 scale = _mm256_mul_ps(
    weights.scales, _mm256_set1_ps(activations.scales[Lane]));
  accumulator = _mm256_fmadd_ps(
    _mm256_cvtepi32_ps(integer_dot), scale, accumulator);
}

QWEN35X_FORCE_INLINE void accumulate_tile(
  const DecodedWeights & weights,
  const Q8_0BlockX4 & vectors0,
  const Q8_0BlockX4 & vectors1,
  const Q8_0BlockX4 & vectors2,
  const Q8_0BlockX4 & vectors3,
  __m256 (&accumulators)[16]) noexcept {
  accumulate_token<0>(weights, vectors0, accumulators[0]);
  accumulate_token<1>(weights, vectors0, accumulators[1]);
  accumulate_token<2>(weights, vectors0, accumulators[2]);
  accumulate_token<3>(weights, vectors0, accumulators[3]);
  accumulate_token<0>(weights, vectors1, accumulators[4]);
  accumulate_token<1>(weights, vectors1, accumulators[5]);
  accumulate_token<2>(weights, vectors1, accumulators[6]);
  accumulate_token<3>(weights, vectors1, accumulators[7]);
  accumulate_token<0>(weights, vectors2, accumulators[8]);
  accumulate_token<1>(weights, vectors2, accumulators[9]);
  accumulate_token<2>(weights, vectors2, accumulators[10]);
  accumulate_token<3>(weights, vectors2, accumulators[11]);
  accumulate_token<0>(weights, vectors3, accumulators[12]);
  accumulate_token<1>(weights, vectors3, accumulators[13]);
  accumulate_token<2>(weights, vectors3, accumulators[14]);
  accumulate_token<3>(weights, vectors3, accumulators[15]);
}

[[nodiscard]] QWEN35X_FORCE_INLINE __m256 final_row_order(
  const __m256 value) noexcept {
  return _mm256_permutevar8x32_ps(
    value, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
}

#undef QWEN35X_FORCE_INLINE

} // namespace

void q4_0_packed_matmul_q8_0_avx512_vnni(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX4 * vectors,
  float * output,
  std::size_t row_count,
  std::size_t vector_count,
  std::size_t blocks_per_row,
  std::size_t output_row_stride) noexcept {
  const std::size_t row_tiles = row_count / q4_0_packed_rows;
  const std::size_t vector_tiles = vector_count / q8_0_packed_vectors;
  std::size_t vector_tile = 0;
  for (; vector_tile + 4 <= vector_tiles; vector_tile += 4) {
    const Q8_0BlockX4 * vector_tile_data = vectors + vector_tile * blocks_per_row;
    for (std::size_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
      __m256 accumulators[16] = {};
      const Q4_0BlockX8 * row_tile_data = matrix + row_tile * blocks_per_row;
      for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const DecodedWeights weights = decode_weights(row_tile_data[block]);
        accumulate_tile(
          weights,
          vector_tile_data[block],
          vector_tile_data[blocks_per_row + block],
          vector_tile_data[2 * blocks_per_row + block],
          vector_tile_data[3 * blocks_per_row + block],
          accumulators);
      }
      for (std::size_t token = 0; token < 16; ++token) {
        _mm256_storeu_ps(
          output + (vector_tile * 4 + token) * output_row_stride + row_tile * 8,
          final_row_order(accumulators[token]));
      }
    }
  }

  if (vector_tile < vector_tiles) {
    q4_0_packed_matmul_q8_0_evex_vnni(
      matrix,
      vectors + vector_tile * blocks_per_row,
      output + vector_tile * 4 * output_row_stride,
      row_count,
      (vector_tiles - vector_tile) * q8_0_packed_vectors,
      blocks_per_row,
      output_row_stride);
  }
}

} // namespace qwen35x::cpu::detail
