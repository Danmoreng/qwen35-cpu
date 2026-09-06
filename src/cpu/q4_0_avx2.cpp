#include "q4_0_internal.h"

#include "f16c_compat.h"

#include <immintrin.h>

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

[[nodiscard]] std::int32_t horizontal_sum_i32(const __m256i value) noexcept {
  const __m128i low = _mm256_castsi256_si128(value);
  const __m128i high = _mm256_extracti128_si256(value, 1);
  __m128i sum = _mm_add_epi32(low, high);
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

[[nodiscard]] __m256 quantize_q8_eight(
  const __m256 values,
  const __m256 inverse_scale) noexcept {
  const __m256 scaled = _mm256_mul_ps(values, inverse_scale);
  const __m256 sign = _mm256_and_ps(scaled, _mm256_set1_ps(-0.0F));
  return _mm256_add_ps(
    scaled, _mm256_or_ps(_mm256_set1_ps(0.5F), sign));
}

template <typename Block>
void quantize_q8_block_packed(
  const float * input,
  Block & output,
  const std::size_t token) noexcept {
  const __m256 x0 = _mm256_loadu_ps(input);
  const __m256 x1 = _mm256_loadu_ps(input + 8);
  const __m256 x2 = _mm256_loadu_ps(input + 16);
  const __m256 x3 = _mm256_loadu_ps(input + 24);
  const __m256 sign_bit = _mm256_set1_ps(-0.0F);
  __m256 absolute_max = _mm256_andnot_ps(sign_bit, x0);
  absolute_max = _mm256_max_ps(absolute_max, _mm256_andnot_ps(sign_bit, x1));
  absolute_max = _mm256_max_ps(absolute_max, _mm256_andnot_ps(sign_bit, x2));
  absolute_max = _mm256_max_ps(absolute_max, _mm256_andnot_ps(sign_bit, x3));
  __m128 max4 = _mm_max_ps(
    _mm256_castps256_ps128(absolute_max),
    _mm256_extractf128_ps(absolute_max, 1));
  max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
  max4 = _mm_max_ss(max4, _mm_shuffle_ps(max4, max4, 1));
  const float max_scalar = _mm_cvtss_f32(max4);
  const float scale = max_scalar / 127.0F;
  const float inverse = scale == 0.0F ? 0.0F : 1.0F / scale;
  const std::uint16_t half_scale = f16c_float_to_half(scale);
  output.scales[token] = f16c_half_to_float(half_scale);

  const __m256 inverse_scale = _mm256_set1_ps(inverse);
  const __m256i minimum = _mm256_set1_epi32(-127);
  const __m256i maximum = _mm256_set1_epi32(127);
  __m256i i0 = _mm256_cvttps_epi32(quantize_q8_eight(x0, inverse_scale));
  __m256i i1 = _mm256_cvttps_epi32(quantize_q8_eight(x1, inverse_scale));
  __m256i i2 = _mm256_cvttps_epi32(quantize_q8_eight(x2, inverse_scale));
  __m256i i3 = _mm256_cvttps_epi32(quantize_q8_eight(x3, inverse_scale));
  i0 = _mm256_min_epi32(_mm256_max_epi32(i0, minimum), maximum);
  i1 = _mm256_min_epi32(_mm256_max_epi32(i1, minimum), maximum);
  i2 = _mm256_min_epi32(_mm256_max_epi32(i2, minimum), maximum);
  i3 = _mm256_min_epi32(_mm256_max_epi32(i3, minimum), maximum);
  output.sums[token] = static_cast<std::int16_t>(horizontal_sum_i32(
    _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3))));
  i0 = _mm256_packs_epi32(i0, i1);
  i2 = _mm256_packs_epi32(i2, i3);
  const __m256i packed = _mm256_permutevar8x32_epi32(
    _mm256_packs_epi16(i0, i2),
    _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
  _mm_storeu_si128(
    reinterpret_cast<__m128i *>(output.qs + token * 32),
    _mm256_castsi256_si128(packed));
  _mm_storeu_si128(
    reinterpret_cast<__m128i *>(output.qs + token * 32 + 16),
    _mm256_extracti128_si256(packed, 1));
}

[[nodiscard]] __m256i unpack_q4_0(const Q4_0Block & block) noexcept {
  const __m128i packed = _mm_loadu_si128(
    reinterpret_cast<const __m128i *>(block.qs));
  const __m128i nibble_mask = _mm_set1_epi8(0x0f);
  const __m128i offset = _mm_set1_epi8(8);
  const __m128i low = _mm_sub_epi8(_mm_and_si128(packed, nibble_mask), offset);
  const __m128i high = _mm_sub_epi8(
    _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask), offset);
  return _mm256_set_m128i(high, low);
}

[[nodiscard]] __m256i dot_q4_q8_loaded(
  const __m256i weights,
  const __m256i activations,
  const __m256i activation_absolute) noexcept {
  const __m256i signed_weights = _mm256_sign_epi8(weights, activations);
  const __m256i pair_products = _mm256_maddubs_epi16(
    activation_absolute, signed_weights);
  return _mm256_madd_epi16(pair_products, _mm256_set1_epi16(1));
}

[[nodiscard]] __m256i mul_sum_i8_pairs_acc(
  const __m256i accumulator,
  const __m256i lhs,
  const __m256i rhs) noexcept {
  const __m256i lhs_absolute = _mm256_abs_epi8(lhs);
  const __m256i signed_rhs = _mm256_sign_epi8(rhs, lhs);
  const __m256i pair_products = _mm256_maddubs_epi16(lhs_absolute, signed_rhs);
  return _mm256_add_epi32(
    accumulator,
    _mm256_madd_epi16(pair_products, _mm256_set1_epi16(1)));
}

template <std::size_t TokenCount, typename Block>
void accumulate_packed_block_x8(
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

  // The lane order below is 0,4,1,5,2,6,3,7 until the final permutation.
  const __m256 natural_weight_scales = _mm256_cvtph_ps(_mm_loadu_si128(
    reinterpret_cast<const __m128i *>(weights.d)));
  const __m256 weight_scales = _mm256_permutevar8x32_ps(
    natural_weight_scales,
    _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

  for (std::size_t token = 0; token < TokenCount; ++token) {
    const Block & activation_block = token < 4
      ? activations0 : *activations1;
    const std::size_t activation_lane = token % 4;
    // Each packed row contributes eight bytes per chunk. Broadcast those
    // eight activation bytes and reduce in the existing row layout, avoiding
    // per-product weight shuffles/blends. A signed 16-bit lane accumulates
    // eight products at most: 8 * 15 * 128 = 15360, so neither the pairwise
    // saturating multiply-add nor the following additions can overflow.
    const std::int8_t * activation = activation_block.qs + activation_lane * 32;
    const __m256i a0 = _mm256_broadcastq_epi64(
      _mm_loadl_epi64(reinterpret_cast<const __m128i *>(activation)));
    const __m256i a1 = _mm256_broadcastq_epi64(
      _mm_loadl_epi64(reinterpret_cast<const __m128i *>(activation + 8)));
    const __m256i a2 = _mm256_broadcastq_epi64(
      _mm_loadl_epi64(reinterpret_cast<const __m128i *>(activation + 16)));
    const __m256i a3 = _mm256_broadcastq_epi64(
      _mm_loadl_epi64(reinterpret_cast<const __m128i *>(activation + 24)));
    const __m256i pairs0123 = _mm256_add_epi16(
      _mm256_add_epi16(_mm256_maddubs_epi16(weight_0123_0, a0),
                       _mm256_maddubs_epi16(weight_0123_1, a1)),
      _mm256_add_epi16(_mm256_maddubs_epi16(weight_0123_2, a2),
                       _mm256_maddubs_epi16(weight_0123_3, a3)));
    const __m256i pairs4567 = _mm256_add_epi16(
      _mm256_add_epi16(_mm256_maddubs_epi16(weight_4567_0, a0),
                       _mm256_maddubs_epi16(weight_4567_1, a1)),
      _mm256_add_epi16(_mm256_maddubs_epi16(weight_4567_2, a2),
                       _mm256_maddubs_epi16(weight_4567_3, a3)));
    const __m256i ones = _mm256_set1_epi16(1);
    // HADD yields 0,1,4,5,2,3,6,7. Match the established accumulator order
    // 0,4,1,5,2,6,3,7 without changing scale/FMA or block reduction order.
    __m256i integer_dot = _mm256_shuffle_epi32(_mm256_hadd_epi32(
      _mm256_madd_epi16(pairs0123, ones),
      _mm256_madd_epi16(pairs4567, ones)), 0xd8);

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

[[nodiscard]] __m256i dot_q4_q8(
  const Q4_0Block & weights,
  const Q8_0Block & activations) noexcept {
  const __m256i activation_bytes = _mm256_loadu_si256(
    reinterpret_cast<const __m256i *>(activations.qs));
  return dot_q4_q8_loaded(
    unpack_q4_0(weights), activation_bytes, _mm256_abs_epi8(activation_bytes));
}

[[nodiscard]] float dot_impl(
  const Q4_0Block * weights,
  const Q8_0Block * activations,
  const std::size_t block_count) noexcept {
  __m256 accumulator0 = _mm256_setzero_ps();
  __m256 accumulator1 = _mm256_setzero_ps();
  __m256 accumulator2 = _mm256_setzero_ps();
  __m256 accumulator3 = _mm256_setzero_ps();
  std::size_t block = 0;
  for (; block + 4 <= block_count; block += 4) {
    accumulator0 = _mm256_fmadd_ps(
      _mm256_cvtepi32_ps(dot_q4_q8(weights[block], activations[block])),
      _mm256_set1_ps(f16c_half_to_float(weights[block].d) * f16c_half_to_float(activations[block].d)),
      accumulator0);
    accumulator1 = _mm256_fmadd_ps(
      _mm256_cvtepi32_ps(dot_q4_q8(weights[block + 1], activations[block + 1])),
      _mm256_set1_ps(
        f16c_half_to_float(weights[block + 1].d) * f16c_half_to_float(activations[block + 1].d)),
      accumulator1);
    accumulator2 = _mm256_fmadd_ps(
      _mm256_cvtepi32_ps(dot_q4_q8(weights[block + 2], activations[block + 2])),
      _mm256_set1_ps(
        f16c_half_to_float(weights[block + 2].d) * f16c_half_to_float(activations[block + 2].d)),
      accumulator2);
    accumulator3 = _mm256_fmadd_ps(
      _mm256_cvtepi32_ps(dot_q4_q8(weights[block + 3], activations[block + 3])),
      _mm256_set1_ps(
        f16c_half_to_float(weights[block + 3].d) * f16c_half_to_float(activations[block + 3].d)),
      accumulator3);
  }
  accumulator0 = _mm256_add_ps(accumulator0, accumulator1);
  accumulator2 = _mm256_add_ps(accumulator2, accumulator3);
  for (; block < block_count; ++block) {
    accumulator0 = _mm256_fmadd_ps(
      _mm256_cvtepi32_ps(dot_q4_q8(weights[block], activations[block])),
      _mm256_set1_ps(f16c_half_to_float(weights[block].d) * f16c_half_to_float(activations[block].d)),
      accumulator0);
  }
  return horizontal_sum_f32(_mm256_add_ps(accumulator0, accumulator2));
}

void dot_eight_rows(
  const Q4_0Block * matrix,
  const Q8_0Block * vector,
  const std::size_t blocks_per_row,
  float * output) noexcept {
  __m256 accumulators[8] = {
    _mm256_setzero_ps(), _mm256_setzero_ps(),
    _mm256_setzero_ps(), _mm256_setzero_ps(),
    _mm256_setzero_ps(), _mm256_setzero_ps(),
    _mm256_setzero_ps(), _mm256_setzero_ps(),
  };
  for (std::size_t block = 0; block < blocks_per_row; ++block) {
    const __m256i activation_bytes = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(vector[block].qs));
    const __m256i activation_absolute = _mm256_abs_epi8(activation_bytes);
    const float activation_scale = f16c_half_to_float(vector[block].d);
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
    for (std::size_t lane = 0; lane < 8; ++lane) {
      const Q4_0Block & weight = matrix[lane * blocks_per_row + block];
      accumulators[lane] = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(dot_q4_q8_loaded(
          unpack_q4_0(weight), activation_bytes, activation_absolute)),
        _mm256_set1_ps(f16c_half_to_float(weight.d) * activation_scale),
        accumulators[lane]);
    }
  }
  for (std::size_t lane = 0; lane < 8; ++lane) {
    output[lane] = horizontal_sum_f32(accumulators[lane]);
  }
}

[[nodiscard]] float dot_with_scales(
  const Q4_0Block * matrix_row,
  const Q8_0Block * vectors,
  const float * vector_scales,
  const float * matrix_scales,
  const std::size_t row,
  const std::size_t vector_index,
  const std::size_t blocks_per_row) noexcept {
  __m256 accumulator = _mm256_setzero_ps();
  for (std::size_t block = 0; block < blocks_per_row; ++block) {
    const std::size_t vector_offset = vector_index * blocks_per_row + block;
    const float weight_scale = matrix_scales != nullptr
      ? matrix_scales[row * blocks_per_row + block]
      : f16c_half_to_float(matrix_row[block].d);
    const float activation_scale = vector_scales != nullptr
      ? vector_scales[vector_offset]
      : f16c_half_to_float(vectors[vector_offset].d);
    accumulator = _mm256_fmadd_ps(
      _mm256_cvtepi32_ps(dot_q4_q8(matrix_row[block], vectors[vector_offset])),
      _mm256_set1_ps(weight_scale * activation_scale), accumulator);
  }
  return horizontal_sum_f32(accumulator);
}

} // namespace

void q4_0_dequantize_avx2(
  const Q4_0Block * input,
  float * output,
  const std::size_t block_count) noexcept {
  for (std::size_t block = 0; block < block_count; ++block) {
    const __m256i values = unpack_q4_0(input[block]);
    const __m256 scale = _mm256_set1_ps(f16c_half_to_float(input[block].d));
    float * destination = output + block * q4_0_values_per_block;
    const __m128i low = _mm256_castsi256_si128(values);
    const __m128i high = _mm256_extracti128_si256(values, 1);
    _mm256_storeu_ps(destination, _mm256_mul_ps(
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low)), scale));
    _mm256_storeu_ps(destination + 8, _mm256_mul_ps(
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(low, 8))), scale));
    _mm256_storeu_ps(destination + 16, _mm256_mul_ps(
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high)), scale));
    _mm256_storeu_ps(destination + 24, _mm256_mul_ps(
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(high, 8))), scale));
  }
}

float q4_0_dot_q8_0_avx2(
  const Q4_0Block * weights,
  const Q8_0Block * activations,
  const std::size_t block_count) noexcept {
  return dot_impl(weights, activations, block_count);
}

void q4_0_matvec_q8_0_avx2(
  const Q4_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row) noexcept {
  if (blocks_per_row == 0) {
    for (std::size_t row = 0; row < row_count; ++row) {
      output[row] = 0.0F;
    }
    return;
  }
  std::size_t row = 0;
  for (; row + 8 <= row_count; row += 8) {
    dot_eight_rows(
      matrix + row * blocks_per_row, vector, blocks_per_row, output + row);
  }
  for (; row < row_count; ++row) {
    output[row] = dot_impl(matrix + row * blocks_per_row, vector, blocks_per_row);
  }
}

void q4_0_matmul_q8_0_avx2(
  const Q4_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  const std::size_t row_count,
  const std::size_t vector_count,
  const std::size_t blocks_per_row,
  const std::size_t output_row_stride,
  const float * vector_scales,
  const float * matrix_scales) noexcept {
  constexpr std::size_t vector_tile = 8;
  for (std::size_t row = 0; row < row_count; ++row) {
    const Q4_0Block * matrix_row = matrix + row * blocks_per_row;
    std::size_t vector_index = 0;
    for (; vector_index + vector_tile <= vector_count; vector_index += vector_tile) {
      __m256 accumulators[vector_tile] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
      };
      for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const __m256i weights = unpack_q4_0(matrix_row[block]);
        const float weight_scale = matrix_scales != nullptr
          ? matrix_scales[row * blocks_per_row + block]
          : f16c_half_to_float(matrix_row[block].d);
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
        for (std::size_t lane = 0; lane < vector_tile; ++lane) {
          const std::size_t vector_offset =
            (vector_index + lane) * blocks_per_row + block;
          const __m256i activation = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(vectors[vector_offset].qs));
          const __m256 integer_dot = _mm256_cvtepi32_ps(dot_q4_q8_loaded(
            weights, activation, _mm256_abs_epi8(activation)));
          const float activation_scale = vector_scales != nullptr
            ? vector_scales[vector_offset]
            : f16c_half_to_float(vectors[vector_offset].d);
          accumulators[lane] = _mm256_fmadd_ps(
            integer_dot,
            _mm256_set1_ps(weight_scale * activation_scale),
            accumulators[lane]);
        }
      }
      for (std::size_t lane = 0; lane < vector_tile; ++lane) {
        output[(vector_index + lane) * output_row_stride + row] =
          horizontal_sum_f32(accumulators[lane]);
      }
    }
    for (; vector_index < vector_count; ++vector_index) {
      output[vector_index * output_row_stride + row] = dot_with_scales(
        matrix_row, vectors, vector_scales, matrix_scales,
        row, vector_index, blocks_per_row);
    }
  }
}

void q4_0_packed_matmul_q8_0_avx2(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX4 * vectors,
  float * output,
  const std::size_t row_count,
  const std::size_t vector_count,
  const std::size_t blocks_per_row,
  const std::size_t output_row_stride) noexcept {
  const __m256i final_permutation = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
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
        accumulate_packed_block_x8<8>(
          row_tile_data[block], vector_tile_data[block],
          vector_tile_data + blocks_per_row + block, accumulators);
      }
      for (std::size_t token = 0; token < 8; ++token) {
        _mm256_storeu_ps(
          output + (vector_tile * 4 + token) * output_row_stride + row_tile * 8,
          _mm256_permutevar8x32_ps(accumulators[token], final_permutation));
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
        accumulate_packed_block_x8<4>(
          row_tile_data[block], vector_tile_data[block], nullptr, accumulators);
      }
      for (std::size_t token = 0; token < 4; ++token) {
        _mm256_storeu_ps(
          output + (vector_tile * 4 + token) * output_row_stride + row_tile * 8,
          _mm256_permutevar8x32_ps(accumulators[token], final_permutation));
      }
    }
  }
}

void q8_0_quantize_vectors_4_avx2(
  const float * input,
  Q8_0BlockX4 * packed,
  const std::size_t vector_count,
  const std::size_t blocks_per_vector) noexcept {
  const std::size_t values_per_vector = blocks_per_vector * q8_0_values_per_block;
  for (std::size_t vector_tile = 0; vector_tile < vector_count / 4; ++vector_tile) {
    for (std::size_t block = 0; block < blocks_per_vector; ++block) {
      Q8_0BlockX4 & destination = packed[vector_tile * blocks_per_vector + block];
      for (std::size_t token = 0; token < 4; ++token) {
        quantize_q8_block_packed(
          input + (vector_tile * 4 + token) * values_per_vector + block * 32,
          destination,
          token);
      }
    }
  }
}

void q8_0_quantize_vector_1_avx2(
  const float * input,
  Q8_0BlockX1 * packed,
  const std::size_t blocks_per_vector) noexcept {
  for (std::size_t block = 0; block < blocks_per_vector; ++block) {
    quantize_q8_block_packed(input + block * 32, packed[block], 0);
  }
}

void q4_0_packed_matvec_q8_0_avx2(
  const Q4_0BlockX8 * matrix,
  const Q8_0Block * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row) noexcept {
  const __m256i final_permutation = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
  for (std::size_t row_tile = 0; row_tile < row_count / 8; ++row_tile) {
    __m256 accumulator[1] = {_mm256_setzero_ps()};
    const Q4_0BlockX8 * row_tile_data = matrix + row_tile * blocks_per_row;
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      Q8_0BlockX4 packed_activation{};
      packed_activation.scales[0] = f16c_half_to_float(vector[block].d);
      std::int32_t activation_sum = 0;
      for (std::size_t chunk = 0; chunk < 4; ++chunk) {
        _mm_storel_epi64(
          reinterpret_cast<__m128i *>(packed_activation.qs + chunk * 8),
          _mm_loadl_epi64(reinterpret_cast<const __m128i *>(
            vector[block].qs + chunk * 8)));
        for (std::size_t index = 0; index < 8; ++index) {
          activation_sum += vector[block].qs[chunk * 8 + index];
        }
      }
      packed_activation.sums[0] = static_cast<std::int16_t>(activation_sum);
      accumulate_packed_block_x8<1>(
        row_tile_data[block], packed_activation, nullptr, accumulator);
    }
    _mm256_storeu_ps(
      output + row_tile * 8,
      _mm256_permutevar8x32_ps(accumulator[0], final_permutation));
  }
}

void q4_0_packed_matvec_prepared_q8_0_avx2(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX1 * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row) noexcept {
  const __m256i final_permutation = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
  for (std::size_t row_tile = 0; row_tile < row_count / 8; ++row_tile) {
    __m256 accumulator[1] = {_mm256_setzero_ps()};
    const Q4_0BlockX8 * row_tile_data = matrix + row_tile * blocks_per_row;
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      accumulate_packed_block_x8<1>(
        row_tile_data[block], vector[block], nullptr, accumulator);
    }
    _mm256_storeu_ps(
      output + row_tile * 8,
      _mm256_permutevar8x32_ps(accumulator[0], final_permutation));
  }
}

Q4_0ArgmaxResult q4_0_packed_matvec_prepared_q8_0_argmax_avx2(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX1 * vector,
  const int * token_counts,
  const float repetition_penalty,
  const std::size_t row_offset,
  const std::size_t row_count,
  const std::size_t blocks_per_row) noexcept {
  const __m256i final_permutation = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
  Q4_0ArgmaxResult best{-std::numeric_limits<float>::infinity(), row_offset};
  alignas(32) float logits[8];
  for (std::size_t row_tile = 0; row_tile < row_count / 8; ++row_tile) {
    __m256 accumulator[1] = {_mm256_setzero_ps()};
    const Q4_0BlockX8 * row_tile_data = matrix + row_tile * blocks_per_row;
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      accumulate_packed_block_x8<1>(
        row_tile_data[block], vector[block], nullptr, accumulator);
    }
    _mm256_store_ps(
      logits, _mm256_permutevar8x32_ps(accumulator[0], final_permutation));
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
