#pragma once

#include "qwen35x/cpu/q8_0.h"

#include <cstddef>
#include <cstdint>

#ifndef QWEN35X_Q8_0_HAS_AVX2_TU
#define QWEN35X_Q8_0_HAS_AVX2_TU 0
#endif

#ifndef QWEN35X_Q8_0_HAS_AVX_VNNI_TU
#define QWEN35X_Q8_0_HAS_AVX_VNNI_TU 0
#endif

#ifndef QWEN35X_Q8_0_HAS_AVX512_TU
#define QWEN35X_Q8_0_HAS_AVX512_TU 0
#endif

#ifndef QWEN35X_Q8_0_HAS_AVX512_VNNI_TU
#define QWEN35X_Q8_0_HAS_AVX512_VNNI_TU 0
#endif

namespace qwen35x::cpu::detail {

[[nodiscard]] std::uint16_t float_to_half(float value) noexcept;
[[nodiscard]] float half_to_float(std::uint16_t value) noexcept;

void q8_0_quantize_scalar(
  const float * input,
  Q8_0Block * output,
  float * scales,
  std::size_t block_count) noexcept;

void q8_0_dequantize_scalar(
  const Q8_0Block * input,
  float * output,
  std::size_t block_count) noexcept;

[[nodiscard]] float q8_0_dot_scalar(
  const Q8_0Block * lhs,
  const Q8_0Block * rhs,
  std::size_t block_count) noexcept;

void q8_0_matvec_scalar(
  const Q8_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;

void q8_0_matmul_scalar(
  const Q8_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  std::size_t row_count,
  std::size_t vector_count,
  std::size_t blocks_per_row,
  std::size_t output_row_stride,
  const float * vector_scales,
  const float * matrix_scales) noexcept;

#if QWEN35X_Q8_0_HAS_AVX2_TU
void q8_0_quantize_avx2(
  const float * input,
  Q8_0Block * output,
  float * scales,
  std::size_t block_count) noexcept;

void q8_0_dequantize_avx2(
  const Q8_0Block * input,
  float * output,
  std::size_t block_count) noexcept;

[[nodiscard]] float q8_0_dot_avx2(
  const Q8_0Block * lhs,
  const Q8_0Block * rhs,
  std::size_t block_count) noexcept;

void q8_0_matvec_avx2(
  const Q8_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;

void q8_0_matmul_avx2(
  const Q8_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  std::size_t row_count,
  std::size_t vector_count,
  std::size_t blocks_per_row,
  std::size_t output_row_stride,
  const float * vector_scales,
  const float * matrix_scales) noexcept;
#endif

#if QWEN35X_Q8_0_HAS_AVX_VNNI_TU
[[nodiscard]] float q8_0_dot_avx_vnni(
  const Q8_0Block * lhs,
  const Q8_0Block * rhs,
  std::size_t block_count) noexcept;

void q8_0_matvec_avx_vnni(
  const Q8_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;

void q8_0_matmul_avx_vnni(
  const Q8_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  std::size_t row_count,
  std::size_t vector_count,
  std::size_t blocks_per_row,
  std::size_t output_row_stride,
  const float * vector_scales,
  const float * matrix_scales) noexcept;
#endif

} // namespace qwen35x::cpu::detail
