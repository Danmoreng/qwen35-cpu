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

[[nodiscard]] __m256i dot_block_i8x8_vnni(
  const std::int8_t * lhs,
  const std::int8_t * rhs) noexcept {
  const __m256i lhs_bytes = _mm256_loadu_si256(
    reinterpret_cast<const __m256i *>(lhs));
  const __m256i rhs_bytes = _mm256_loadu_si256(
    reinterpret_cast<const __m256i *>(rhs));
  const __m256i lhs_absolute = _mm256_abs_epi8(lhs_bytes);
  const __m256i rhs_with_lhs_sign = _mm256_sign_epi8(rhs_bytes, lhs_bytes);
  return _mm256_dpbusd_avx_epi32(
    _mm256_setzero_si256(), lhs_absolute, rhs_with_lhs_sign);
}

[[nodiscard]] __m256i dot_block_loaded_lhs_vnni(
  const __m256i lhs_bytes,
  const __m256i lhs_absolute,
  const __m256i rhs_bytes) noexcept {
  const __m256i rhs_with_lhs_sign = _mm256_sign_epi8(rhs_bytes, lhs_bytes);
  return _mm256_dpbusd_avx_epi32(
    _mm256_setzero_si256(), lhs_absolute, rhs_with_lhs_sign);
}

[[nodiscard]] __m256i dot_block_loaded_rhs_vnni(
  const __m256i lhs_bytes,
  const __m256i rhs_bytes,
  const __m256i rhs_absolute) noexcept {
  const __m256i lhs_with_rhs_sign = _mm256_sign_epi8(lhs_bytes, rhs_bytes);
  return _mm256_dpbusd_avx_epi32(
    _mm256_setzero_si256(), rhs_absolute, lhs_with_rhs_sign);
}

[[nodiscard]] float dot_vnni_impl(
  const Q8_0Block * lhs,
  const Q8_0Block * rhs,
  const std::size_t block_count) noexcept {
  __m256 accumulator0 = _mm256_setzero_ps();
  __m256 accumulator1 = _mm256_setzero_ps();
  __m256 accumulator2 = _mm256_setzero_ps();
  __m256 accumulator3 = _mm256_setzero_ps();
  std::size_t block = 0;
  for (; block + 4 <= block_count; block += 4) {
    const __m256 integer_dot0 = _mm256_cvtepi32_ps(
      dot_block_i8x8_vnni(lhs[block].qs, rhs[block].qs));
    const __m256 integer_dot1 = _mm256_cvtepi32_ps(
      dot_block_i8x8_vnni(lhs[block + 1].qs, rhs[block + 1].qs));
    const __m256 integer_dot2 = _mm256_cvtepi32_ps(
      dot_block_i8x8_vnni(lhs[block + 2].qs, rhs[block + 2].qs));
    const __m256 integer_dot3 = _mm256_cvtepi32_ps(
      dot_block_i8x8_vnni(lhs[block + 3].qs, rhs[block + 3].qs));
    accumulator0 = _mm256_fmadd_ps(
      integer_dot0,
      _mm256_set1_ps(
        f16c_half_to_float(lhs[block].d) * f16c_half_to_float(rhs[block].d)),
      accumulator0);
    accumulator1 = _mm256_fmadd_ps(
      integer_dot1,
      _mm256_set1_ps(
        f16c_half_to_float(lhs[block + 1].d) * f16c_half_to_float(rhs[block + 1].d)),
      accumulator1);
    accumulator2 = _mm256_fmadd_ps(
      integer_dot2,
      _mm256_set1_ps(
        f16c_half_to_float(lhs[block + 2].d) * f16c_half_to_float(rhs[block + 2].d)),
      accumulator2);
    accumulator3 = _mm256_fmadd_ps(
      integer_dot3,
      _mm256_set1_ps(
        f16c_half_to_float(lhs[block + 3].d) * f16c_half_to_float(rhs[block + 3].d)),
      accumulator3);
  }
  accumulator0 = _mm256_add_ps(accumulator0, accumulator1);
  accumulator2 = _mm256_add_ps(accumulator2, accumulator3);
  for (; block < block_count; ++block) {
    const __m256 integer_dot = _mm256_cvtepi32_ps(
      dot_block_i8x8_vnni(lhs[block].qs, rhs[block].qs));
    accumulator0 = _mm256_fmadd_ps(
      integer_dot,
      _mm256_set1_ps(
        f16c_half_to_float(lhs[block].d) * f16c_half_to_float(rhs[block].d)),
      accumulator0);
  }
  return horizontal_sum_f32(_mm256_add_ps(accumulator0, accumulator2));
}

[[nodiscard]] float dot_row_major_vnni_impl(
  const Q8_0Block * matrix_row,
  const Q8_0Block * vectors,
  const float * vector_scales,
  const float * matrix_scales,
  const std::size_t matrix_row_index,
  const std::size_t vector_index,
  const std::size_t blocks_per_row) noexcept {
  __m256 accumulator = _mm256_setzero_ps();
  for (std::size_t block = 0; block < blocks_per_row; ++block) {
    const std::size_t vector_offset = vector_index * blocks_per_row + block;
    const Q8_0Block & vector_block = vectors[vector_offset];
    const __m256 integer_dot = _mm256_cvtepi32_ps(
      dot_block_i8x8_vnni(matrix_row[block].qs, vector_block.qs));
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

void dot_eight_rows_vnni_impl(
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
    for (std::size_t lane = 0; lane < 8; ++lane) {
      const Q8_0Block & weight = matrix[lane * blocks_per_row + block];
      const __m256i weight_bytes = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(weight.qs));
      accumulators[lane] = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(dot_block_loaded_rhs_vnni(
          weight_bytes, vector_bytes, vector_absolute)),
        _mm256_set1_ps(f16c_half_to_float(weight.d) * vector_scale),
        accumulators[lane]);
    }
  }
  for (std::size_t lane = 0; lane < 8; ++lane) {
    output[lane] = horizontal_sum_f32(accumulators[lane]);
  }
}

} // namespace

float q8_0_dot_avx_vnni(
  const Q8_0Block * lhs,
  const Q8_0Block * rhs,
  const std::size_t block_count) noexcept {
  return dot_vnni_impl(lhs, rhs, block_count);
}

void q8_0_matvec_avx_vnni(
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
    dot_eight_rows_vnni_impl(
      matrix + row * blocks_per_row,
      vector,
      blocks_per_row,
      output + row);
  }
  for (; row < row_count; ++row) {
    output[row] = dot_vnni_impl(
      matrix + row * blocks_per_row, vector, blocks_per_row);
  }
}

void q8_0_matmul_avx_vnni(
  const Q8_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  const std::size_t row_count,
  const std::size_t vector_count,
  const std::size_t blocks_per_row,
  const std::size_t output_row_stride,
  const float * vector_scales,
  const float * matrix_scales) noexcept {
  constexpr std::size_t vector_tile = 4;
  std::size_t row = 0;
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
          const __m256i vector_bytes = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(vector_block.qs));
          accumulators[lane] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(dot_block_loaded_lhs_vnni(
              weight_bytes, weight_absolute, vector_bytes)),
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
      output[vector_index * output_row_stride + row] = dot_row_major_vnni_impl(
        matrix_row, vectors, vector_scales, matrix_scales, row,
        vector_index, blocks_per_row);
    }
  }
}

} // namespace qwen35x::cpu::detail
