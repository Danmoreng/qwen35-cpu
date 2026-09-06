#pragma once

#include "qwen35x/cpu/q4_0.h"

#include <cstddef>

namespace qwen35x::cpu::detail {

void q4_0_dequantize_scalar(
  const Q4_0Block * input,
  float * output,
  std::size_t block_count) noexcept;

[[nodiscard]] float q4_0_dot_q8_0_scalar(
  const Q4_0Block * weights,
  const Q8_0Block * activations,
  std::size_t block_count) noexcept;

void q4_0_matvec_q8_0_scalar(
  const Q4_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;

void q4_0_matmul_q8_0_scalar(
  const Q4_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  std::size_t row_count,
  std::size_t vector_count,
  std::size_t blocks_per_row,
  std::size_t output_row_stride,
  const float * vector_scales,
  const float * matrix_scales) noexcept;

void q4_0_packed_matmul_q8_0_scalar(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX4 * vectors,
  float * output,
  std::size_t row_count,
  std::size_t vector_count,
  std::size_t blocks_per_row,
  std::size_t output_row_stride) noexcept;

void q8_0_quantize_vectors_4_scalar(
  const float * input,
  Q8_0BlockX4 * packed,
  std::size_t vector_count,
  std::size_t blocks_per_vector) noexcept;

void q8_0_quantize_vector_1_scalar(
  const float * input,
  Q8_0BlockX1 * packed,
  std::size_t blocks_per_vector) noexcept;

void q4_0_packed_matvec_q8_0_scalar(
  const Q4_0BlockX8 * matrix,
  const Q8_0Block * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;

void q4_0_packed_matvec_prepared_q8_0_scalar(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX1 * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;

[[nodiscard]] Q4_0ArgmaxResult
q4_0_packed_matvec_prepared_q8_0_argmax_scalar(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX1 * vector,
  const int * token_counts,
  float repetition_penalty,
  std::size_t row_offset,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;

#if QWEN35X_Q8_0_HAS_AVX2_TU
void q4_0_dequantize_avx2(
  const Q4_0Block * input,
  float * output,
  std::size_t block_count) noexcept;

[[nodiscard]] float q4_0_dot_q8_0_avx2(
  const Q4_0Block * weights,
  const Q8_0Block * activations,
  std::size_t block_count) noexcept;

void q4_0_matvec_q8_0_avx2(
  const Q4_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;

void q4_0_matmul_q8_0_avx2(
  const Q4_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  std::size_t row_count,
  std::size_t vector_count,
  std::size_t blocks_per_row,
  std::size_t output_row_stride,
  const float * vector_scales,
  const float * matrix_scales) noexcept;

void q4_0_packed_matmul_q8_0_avx2(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX4 * vectors,
  float * output,
  std::size_t row_count,
  std::size_t vector_count,
  std::size_t blocks_per_row,
  std::size_t output_row_stride) noexcept;

void q8_0_quantize_vectors_4_avx2(
  const float * input,
  Q8_0BlockX4 * packed,
  std::size_t vector_count,
  std::size_t blocks_per_vector) noexcept;

void q8_0_quantize_vector_1_avx2(
  const float * input,
  Q8_0BlockX1 * packed,
  std::size_t blocks_per_vector) noexcept;

void q4_0_packed_matvec_q8_0_avx2(
  const Q4_0BlockX8 * matrix,
  const Q8_0Block * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;

void q4_0_packed_matvec_prepared_q8_0_avx2(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX1 * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;

[[nodiscard]] Q4_0ArgmaxResult
q4_0_packed_matvec_prepared_q8_0_argmax_avx2(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX1 * vector,
  const int * token_counts,
  float repetition_penalty,
  std::size_t row_offset,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;
#endif

#if QWEN35X_Q8_0_HAS_AVX_VNNI_TU
void q4_0_packed_matmul_q8_0_avx_vnni(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX4 * vectors,
  float * output,
  std::size_t row_count,
  std::size_t vector_count,
  std::size_t blocks_per_row,
  std::size_t output_row_stride) noexcept;

void q4_0_packed_matvec_q8_0_avx_vnni(
  const Q4_0BlockX8 * matrix,
  const Q8_0Block * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;

void q4_0_packed_matvec_prepared_q8_0_avx_vnni(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX1 * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;

[[nodiscard]] Q4_0ArgmaxResult
q4_0_packed_matvec_prepared_q8_0_argmax_avx_vnni(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX1 * vector,
  const int * token_counts,
  float repetition_penalty,
  std::size_t row_offset,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;
#endif

#if QWEN35X_Q8_0_HAS_AVX512_VNNI_TU
void q4_0_packed_matmul_q8_0_evex_vnni(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX4 * vectors,
  float * output,
  std::size_t row_count,
  std::size_t vector_count,
  std::size_t blocks_per_row,
  std::size_t output_row_stride) noexcept;

void q4_0_packed_matvec_q8_0_evex_vnni(
  const Q4_0BlockX8 * matrix,
  const Q8_0Block * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;

void q4_0_packed_matvec_prepared_q8_0_evex_vnni(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX1 * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;

[[nodiscard]] Q4_0ArgmaxResult
q4_0_packed_matvec_prepared_q8_0_argmax_evex_vnni(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX1 * vector,
  const int * token_counts,
  float repetition_penalty,
  std::size_t row_offset,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;
#endif

#if QWEN35X_Q8_0_HAS_AVX512_VNNI_TU
void q4_0_packed_matmul_q8_0_avx512_vnni(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX4 * vectors,
  float * output,
  std::size_t row_count,
  std::size_t vector_count,
  std::size_t blocks_per_row,
  std::size_t output_row_stride) noexcept;
#endif

} // namespace qwen35x::cpu::detail
