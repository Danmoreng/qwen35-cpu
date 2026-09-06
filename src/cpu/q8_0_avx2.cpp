#include "q8_0_internal.h"

#include "f16c_compat.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

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

[[nodiscard]] __m256i dot_block_i8x8(
  const std::int8_t * lhs,
  const std::int8_t * rhs) noexcept {
  const __m256i lhs_bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lhs));
  const __m256i rhs_bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(rhs));

  // Q8_0 never emits -128, so abs(lhs) fits in unsigned bytes and every
  // adjacent pair product fits in signed int16. This maps signed*signed i8
  // onto AVX2's unsigned*signed maddubs instruction without saturation.
  const __m256i lhs_absolute = _mm256_abs_epi8(lhs_bytes);
  const __m256i rhs_with_lhs_sign = _mm256_sign_epi8(rhs_bytes, lhs_bytes);
  const __m256i pair_products = _mm256_maddubs_epi16(lhs_absolute, rhs_with_lhs_sign);
  return _mm256_madd_epi16(pair_products, _mm256_set1_epi16(1));
}

[[nodiscard]] __m256i dot_block_i8x8_loaded_lhs(
  const __m256i lhs_bytes,
  const __m256i lhs_absolute,
  const __m256i rhs_bytes) noexcept {
  const __m256i rhs_with_lhs_sign = _mm256_sign_epi8(rhs_bytes, lhs_bytes);
  const __m256i pair_products = _mm256_maddubs_epi16(lhs_absolute, rhs_with_lhs_sign);
  return _mm256_madd_epi16(pair_products, _mm256_set1_epi16(1));
}

[[nodiscard]] __m256i dot_block_i8x8_loaded_lhs(
  const __m256i lhs_bytes,
  const __m256i lhs_absolute,
  const std::int8_t * rhs) noexcept {
  return dot_block_i8x8_loaded_lhs(
    lhs_bytes,
    lhs_absolute,
    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(rhs)));
}

[[nodiscard]] __m256i dot_block_i8x8_loaded_rhs(
  const __m256i lhs_bytes,
  const __m256i rhs_bytes,
  const __m256i rhs_absolute) noexcept {
  const __m256i lhs_with_rhs_sign = _mm256_sign_epi8(lhs_bytes, rhs_bytes);
  const __m256i pair_products = _mm256_maddubs_epi16(rhs_absolute, lhs_with_rhs_sign);
  return _mm256_madd_epi16(pair_products, _mm256_set1_epi16(1));
}

[[nodiscard]] __m256 quantize_eight(
  const __m256 values,
  const __m256 inverse_scale) noexcept {
  const __m256 scaled = _mm256_mul_ps(values, inverse_scale);
  const __m256 sign = _mm256_and_ps(scaled, _mm256_set1_ps(-0.0F));
  const __m256 half_away_from_zero = _mm256_or_ps(_mm256_set1_ps(0.5F), sign);
  return _mm256_add_ps(scaled, half_away_from_zero);
}

[[nodiscard]] float dot_avx2_impl(
  const Q8_0Block * lhs,
  const Q8_0Block * rhs,
  const std::size_t block_count) noexcept {
  // Keep eight independent FP32 accumulation lanes across all Q8 blocks and
  // reduce only once per row. Reducing each 32-value block separately creates
  // a long scalar dependency chain and leaves most of AVX2 idle.
  __m256 accumulator0 = _mm256_setzero_ps();
  __m256 accumulator1 = _mm256_setzero_ps();
  __m256 accumulator2 = _mm256_setzero_ps();
  __m256 accumulator3 = _mm256_setzero_ps();
  std::size_t block = 0;
  for (; block + 4 <= block_count; block += 4) {
    const __m256 integer_dot0 = _mm256_cvtepi32_ps(
      dot_block_i8x8(lhs[block].qs, rhs[block].qs));
    const __m256 integer_dot1 = _mm256_cvtepi32_ps(
      dot_block_i8x8(lhs[block + 1].qs, rhs[block + 1].qs));
    const __m256 integer_dot2 = _mm256_cvtepi32_ps(
      dot_block_i8x8(lhs[block + 2].qs, rhs[block + 2].qs));
    const __m256 integer_dot3 = _mm256_cvtepi32_ps(
      dot_block_i8x8(lhs[block + 3].qs, rhs[block + 3].qs));
    accumulator0 = _mm256_fmadd_ps(
      integer_dot0,
      _mm256_set1_ps(f16c_half_to_float(lhs[block].d) * f16c_half_to_float(rhs[block].d)),
      accumulator0);
    accumulator1 = _mm256_fmadd_ps(
      integer_dot1,
      _mm256_set1_ps(f16c_half_to_float(lhs[block + 1].d) * f16c_half_to_float(rhs[block + 1].d)),
      accumulator1);
    accumulator2 = _mm256_fmadd_ps(
      integer_dot2,
      _mm256_set1_ps(f16c_half_to_float(lhs[block + 2].d) * f16c_half_to_float(rhs[block + 2].d)),
      accumulator2);
    accumulator3 = _mm256_fmadd_ps(
      integer_dot3,
      _mm256_set1_ps(f16c_half_to_float(lhs[block + 3].d) * f16c_half_to_float(rhs[block + 3].d)),
      accumulator3);
  }
  accumulator0 = _mm256_add_ps(accumulator0, accumulator1);
  accumulator2 = _mm256_add_ps(accumulator2, accumulator3);
  for (; block < block_count; ++block) {
    const __m256 integer_dot = _mm256_cvtepi32_ps(
      dot_block_i8x8(lhs[block].qs, rhs[block].qs));
    accumulator0 = _mm256_fmadd_ps(
      integer_dot,
      _mm256_set1_ps(f16c_half_to_float(lhs[block].d) * f16c_half_to_float(rhs[block].d)),
      accumulator0);
  }
  return horizontal_sum_f32(_mm256_add_ps(accumulator0, accumulator2));
}

[[nodiscard]] float dot_row_major_avx2_impl(
  const Q8_0Block * matrix_row,
  const Q8_0Block * vectors,
  const float * vector_scales,
  const float * matrix_scales,
  const std::size_t matrix_row_index,
  const std::size_t vector_index,
  const std::size_t vector_count,
  const std::size_t blocks_per_row) noexcept {
  __m256 accumulator = _mm256_setzero_ps();
  for (std::size_t block = 0; block < blocks_per_row; ++block) {
    const std::size_t vector_offset = vector_index * blocks_per_row + block;
    const Q8_0Block & vector_block = vectors[vector_offset];
    const __m256 integer_dot = _mm256_cvtepi32_ps(
      dot_block_i8x8(matrix_row[block].qs, vector_block.qs));
    const float matrix_scale = matrix_scales != nullptr
      ? matrix_scales[matrix_row_index * blocks_per_row + block]
      : f16c_half_to_float(matrix_row[block].d);
    const float vector_scale = vector_scales != nullptr
      ? vector_scales[vector_offset]
      : f16c_half_to_float(vector_block.d);
    accumulator = _mm256_fmadd_ps(
      integer_dot, _mm256_set1_ps(matrix_scale * vector_scale), accumulator);
  }
  return horizontal_sum_f32(accumulator);
}

void dot_four_rows_avx2_impl(
  const Q8_0Block * row0,
  const Q8_0Block * row1,
  const Q8_0Block * row2,
  const Q8_0Block * row3,
  const Q8_0Block * vector,
  const std::size_t blocks_per_row,
  float * output) noexcept {
  __m256 accumulator0 = _mm256_setzero_ps();
  __m256 accumulator1 = _mm256_setzero_ps();
  __m256 accumulator2 = _mm256_setzero_ps();
  __m256 accumulator3 = _mm256_setzero_ps();
  for (std::size_t block = 0; block < blocks_per_row; ++block) {
    const __m256i vector_bytes = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(vector[block].qs));
    const __m256i vector_absolute = _mm256_abs_epi8(vector_bytes);
    const float vector_scale = f16c_half_to_float(vector[block].d);

    const __m256i weight0 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(row0[block].qs));
    const __m256i weight1 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(row1[block].qs));
    const __m256i weight2 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(row2[block].qs));
    const __m256i weight3 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(row3[block].qs));
    accumulator0 = _mm256_fmadd_ps(
      _mm256_cvtepi32_ps(dot_block_i8x8_loaded_rhs(
        weight0, vector_bytes, vector_absolute)),
      _mm256_set1_ps(f16c_half_to_float(row0[block].d) * vector_scale),
      accumulator0);
    accumulator1 = _mm256_fmadd_ps(
      _mm256_cvtepi32_ps(dot_block_i8x8_loaded_rhs(
        weight1, vector_bytes, vector_absolute)),
      _mm256_set1_ps(f16c_half_to_float(row1[block].d) * vector_scale),
      accumulator1);
    accumulator2 = _mm256_fmadd_ps(
      _mm256_cvtepi32_ps(dot_block_i8x8_loaded_rhs(
        weight2, vector_bytes, vector_absolute)),
      _mm256_set1_ps(f16c_half_to_float(row2[block].d) * vector_scale),
      accumulator2);
    accumulator3 = _mm256_fmadd_ps(
      _mm256_cvtepi32_ps(dot_block_i8x8_loaded_rhs(
        weight3, vector_bytes, vector_absolute)),
      _mm256_set1_ps(f16c_half_to_float(row3[block].d) * vector_scale),
      accumulator3);
  }
  output[0] = horizontal_sum_f32(accumulator0);
  output[1] = horizontal_sum_f32(accumulator1);
  output[2] = horizontal_sum_f32(accumulator2);
  output[3] = horizontal_sum_f32(accumulator3);
}

void dot_eight_rows_avx2_impl(
  const Q8_0Block * matrix,
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
    const __m256i vector_bytes = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(vector[block].qs));
    const __m256i vector_absolute = _mm256_abs_epi8(vector_bytes);
    const float vector_scale = f16c_half_to_float(vector[block].d);
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
    for (std::size_t lane = 0; lane < 8; ++lane) {
      const Q8_0Block & weight = matrix[lane * blocks_per_row + block];
      const __m256i weight_bytes = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(weight.qs));
      accumulators[lane] = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(dot_block_i8x8_loaded_rhs(
          weight_bytes, vector_bytes, vector_absolute)),
        _mm256_set1_ps(f16c_half_to_float(weight.d) * vector_scale),
        accumulators[lane]);
    }
  }
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
  for (std::size_t lane = 0; lane < 8; ++lane) {
    output[lane] = horizontal_sum_f32(accumulators[lane]);
  }
}

} // namespace

void q8_0_quantize_avx2(
  const float * input,
  Q8_0Block * output,
  float * scales,
  const std::size_t block_count) noexcept {
  const __m256 sign_bit = _mm256_set1_ps(-0.0F);
  const __m256i pack_order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

  for (std::size_t block = 0; block < block_count; ++block) {
    const float * x = input + block * q8_0_values_per_block;
    const __m256 x0 = _mm256_loadu_ps(x);
    const __m256 x1 = _mm256_loadu_ps(x + 8);
    const __m256 x2 = _mm256_loadu_ps(x + 16);
    const __m256 x3 = _mm256_loadu_ps(x + 24);

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
    const float inverse_scale_scalar = scale == 0.0F ? 0.0F : 1.0F / scale;
    const std::uint16_t half_scale = f16c_float_to_half(scale);
    output[block].d = half_scale;
    if (scales != nullptr) {
      scales[block] = f16c_half_to_float(half_scale);
    }
    const __m256 inverse_scale = _mm256_set1_ps(inverse_scale_scalar);

    __m256i i0 = _mm256_cvttps_epi32(quantize_eight(x0, inverse_scale));
    __m256i i1 = _mm256_cvttps_epi32(quantize_eight(x1, inverse_scale));
    __m256i i2 = _mm256_cvttps_epi32(quantize_eight(x2, inverse_scale));
    __m256i i3 = _mm256_cvttps_epi32(quantize_eight(x3, inverse_scale));
    const __m256i minimum = _mm256_set1_epi32(-127);
    const __m256i maximum = _mm256_set1_epi32(127);
    i0 = _mm256_min_epi32(_mm256_max_epi32(i0, minimum), maximum);
    i1 = _mm256_min_epi32(_mm256_max_epi32(i1, minimum), maximum);
    i2 = _mm256_min_epi32(_mm256_max_epi32(i2, minimum), maximum);
    i3 = _mm256_min_epi32(_mm256_max_epi32(i3, minimum), maximum);
    i0 = _mm256_packs_epi32(i0, i1);
    i2 = _mm256_packs_epi32(i2, i3);
    i0 = _mm256_packs_epi16(i0, i2);
    i0 = _mm256_permutevar8x32_epi32(i0, pack_order);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output[block].qs), i0);
  }
}

void q8_0_dequantize_avx2(
  const Q8_0Block * input,
  float * output,
  const std::size_t block_count) noexcept {
  for (std::size_t block = 0; block < block_count; ++block) {
    const __m256 scale = _mm256_set1_ps(f16c_half_to_float(input[block].d));
    float * y = output + block * q8_0_values_per_block;
    for (std::size_t group = 0; group < 4; ++group) {
      const __m128i bytes = _mm_loadl_epi64(
        reinterpret_cast<const __m128i *>(input[block].qs + group * 8));
      const __m256i integers = _mm256_cvtepi8_epi32(bytes);
      const __m256 values = _mm256_mul_ps(_mm256_cvtepi32_ps(integers), scale);
      _mm256_storeu_ps(y + group * 8, values);
    }
  }
}

float q8_0_dot_avx2(
  const Q8_0Block * lhs,
  const Q8_0Block * rhs,
  const std::size_t block_count) noexcept {
  return dot_avx2_impl(lhs, rhs, block_count);
}

void q8_0_matvec_avx2(
  const Q8_0Block * matrix,
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
    dot_eight_rows_avx2_impl(
      matrix + row * blocks_per_row,
      vector,
      blocks_per_row,
      output + row);
  }
  for (; row + 4 <= row_count; row += 4) {
    const Q8_0Block * row0 = matrix + row * blocks_per_row;
    dot_four_rows_avx2_impl(
      row0,
      row0 + blocks_per_row,
      row0 + 2 * blocks_per_row,
      row0 + 3 * blocks_per_row,
      vector,
      blocks_per_row,
      output + row);
  }
  for (; row < row_count; ++row) {
    output[row] = dot_avx2_impl(
      matrix + row * blocks_per_row,
      vector,
      blocks_per_row);
  }
}

void q8_0_matmul_avx2(
  const Q8_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  const std::size_t row_count,
  const std::size_t vector_count,
  const std::size_t blocks_per_row,
  const std::size_t output_row_stride,
  const float * vector_scales,
  const float * matrix_scales) noexcept {
  // Eight activation rows fit alongside the shared weight block and scratch
  // registers on AVX2. This halves weight traffic compared with the original
  // four-row tile, which matters because prefill repeatedly streams the much
  // larger model matrix through cache.
  constexpr std::size_t vector_tile = 8;
  std::size_t row = 0;
  // A 2x3 tile keeps all six accumulators, three activation blocks, and the
  // current weight block in AVX2's 16 registers. Reusing each activation for
  // two output rows cuts activation loads in half without register spills.
  for (; row + 2 <= row_count; row += 2) {
    const Q8_0Block * matrix_row0 = matrix + row * blocks_per_row;
    const Q8_0Block * matrix_row1 = matrix_row0 + blocks_per_row;
    std::size_t vector_index = 0;
    for (; vector_index + 3 <= vector_count; vector_index += 3) {
      __m256 accum00 = _mm256_setzero_ps();
      __m256 accum01 = _mm256_setzero_ps();
      __m256 accum02 = _mm256_setzero_ps();
      __m256 accum10 = _mm256_setzero_ps();
      __m256 accum11 = _mm256_setzero_ps();
      __m256 accum12 = _mm256_setzero_ps();
      for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const Q8_0Block & vector0 = vectors[(vector_index + 0) * blocks_per_row + block];
        const Q8_0Block & vector1 = vectors[(vector_index + 1) * blocks_per_row + block];
        const Q8_0Block & vector2 = vectors[(vector_index + 2) * blocks_per_row + block];
        const __m256i bytes0 = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(vector0.qs));
        const __m256i bytes1 = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(vector1.qs));
        const __m256i bytes2 = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(vector2.qs));
        const float vector_scale0 = vector_scales != nullptr
          ? vector_scales[(vector_index + 0) * blocks_per_row + block]
          : f16c_half_to_float(vector0.d);
        const float vector_scale1 = vector_scales != nullptr
          ? vector_scales[(vector_index + 1) * blocks_per_row + block]
          : f16c_half_to_float(vector1.d);
        const float vector_scale2 = vector_scales != nullptr
          ? vector_scales[(vector_index + 2) * blocks_per_row + block]
          : f16c_half_to_float(vector2.d);

        const __m256i weight0 = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(matrix_row0[block].qs));
        const __m256i absolute0 = _mm256_abs_epi8(weight0);
        const float scale0 = matrix_scales != nullptr
          ? matrix_scales[row * blocks_per_row + block]
          : f16c_half_to_float(matrix_row0[block].d);
        accum00 = _mm256_fmadd_ps(
          _mm256_cvtepi32_ps(dot_block_i8x8_loaded_lhs(weight0, absolute0, bytes0)),
          _mm256_set1_ps(scale0 * vector_scale0), accum00);
        accum01 = _mm256_fmadd_ps(
          _mm256_cvtepi32_ps(dot_block_i8x8_loaded_lhs(weight0, absolute0, bytes1)),
          _mm256_set1_ps(scale0 * vector_scale1), accum01);
        accum02 = _mm256_fmadd_ps(
          _mm256_cvtepi32_ps(dot_block_i8x8_loaded_lhs(weight0, absolute0, bytes2)),
          _mm256_set1_ps(scale0 * vector_scale2), accum02);

        const __m256i weight1 = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(matrix_row1[block].qs));
        const __m256i absolute1 = _mm256_abs_epi8(weight1);
        const float scale1 = matrix_scales != nullptr
          ? matrix_scales[(row + 1) * blocks_per_row + block]
          : f16c_half_to_float(matrix_row1[block].d);
        accum10 = _mm256_fmadd_ps(
          _mm256_cvtepi32_ps(dot_block_i8x8_loaded_lhs(weight1, absolute1, bytes0)),
          _mm256_set1_ps(scale1 * vector_scale0), accum10);
        accum11 = _mm256_fmadd_ps(
          _mm256_cvtepi32_ps(dot_block_i8x8_loaded_lhs(weight1, absolute1, bytes1)),
          _mm256_set1_ps(scale1 * vector_scale1), accum11);
        accum12 = _mm256_fmadd_ps(
          _mm256_cvtepi32_ps(dot_block_i8x8_loaded_lhs(weight1, absolute1, bytes2)),
          _mm256_set1_ps(scale1 * vector_scale2), accum12);
      }
      output[(vector_index + 0) * output_row_stride + row] = horizontal_sum_f32(accum00);
      output[(vector_index + 1) * output_row_stride + row] = horizontal_sum_f32(accum01);
      output[(vector_index + 2) * output_row_stride + row] = horizontal_sum_f32(accum02);
      output[(vector_index + 0) * output_row_stride + row + 1] = horizontal_sum_f32(accum10);
      output[(vector_index + 1) * output_row_stride + row + 1] = horizontal_sum_f32(accum11);
      output[(vector_index + 2) * output_row_stride + row + 1] = horizontal_sum_f32(accum12);
    }
    for (; vector_index < vector_count; ++vector_index) {
      output[vector_index * output_row_stride + row] = dot_row_major_avx2_impl(
        matrix_row0, vectors, vector_scales, matrix_scales, row,
        vector_index, vector_count, blocks_per_row);
      output[vector_index * output_row_stride + row + 1] = dot_row_major_avx2_impl(
        matrix_row1, vectors, vector_scales, matrix_scales, row + 1,
        vector_index, vector_count, blocks_per_row);
    }
  }
  for (; row < row_count; ++row) {
    const Q8_0Block * matrix_row = matrix + row * blocks_per_row;
    std::size_t vector_index = 0;
    for (; vector_index + vector_tile <= vector_count; vector_index += vector_tile) {
      __m256 accumulators[vector_tile];
      for (std::size_t lane = 0; lane < vector_tile; ++lane) {
        accumulators[lane] = _mm256_setzero_ps();
      }
      for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const __m256i weight_bytes = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(matrix_row[block].qs));
        const __m256i weight_absolute = _mm256_abs_epi8(weight_bytes);
        const float weight_scale = matrix_scales != nullptr
          ? matrix_scales[row * blocks_per_row + block]
          : f16c_half_to_float(matrix_row[block].d);
        for (std::size_t lane = 0; lane < vector_tile; ++lane) {
          const std::size_t vector_offset =
            (vector_index + lane) * blocks_per_row + block;
          const Q8_0Block & vector_block = vectors[vector_offset];
          const __m256 integer_dot = _mm256_cvtepi32_ps(
            dot_block_i8x8_loaded_lhs(weight_bytes, weight_absolute, vector_block.qs));
          accumulators[lane] = _mm256_fmadd_ps(
            integer_dot,
            _mm256_set1_ps(
              weight_scale *
              (vector_scales != nullptr
                 ? vector_scales[vector_offset]
                 : f16c_half_to_float(vector_block.d))),
            accumulators[lane]);
        }
      }
      for (std::size_t lane = 0; lane < vector_tile; ++lane) {
        output[(vector_index + lane) * output_row_stride + row] =
          horizontal_sum_f32(accumulators[lane]);
      }
    }
    for (; vector_index < vector_count; ++vector_index) {
      output[vector_index * output_row_stride + row] = dot_row_major_avx2_impl(
        matrix_row, vectors, vector_scales, matrix_scales, row,
        vector_index, vector_count, blocks_per_row);
    }
  }
}

} // namespace qwen35x::cpu::detail
