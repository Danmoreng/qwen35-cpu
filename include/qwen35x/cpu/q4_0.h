#pragma once

#include "qwen35x/cpu/q8_0.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace qwen35x::cpu {

[[nodiscard]] const char * q4_0_decode_kernel_name(Q8_0Backend backend) noexcept;
[[nodiscard]] const char * q4_0_prefill_kernel_name(Q8_0Backend backend) noexcept;


inline constexpr std::size_t q4_0_values_per_block = 32;

// Binary-compatible with GGML's block_q4_0: one IEEE-754 binary16 scale
// followed by 16 bytes. Byte i stores value i in its low nibble and value
// i + 16 in its high nibble; both are interpreted as unsigned_nibble - 8.
struct Q4_0Block {
  std::uint16_t d;
  std::uint8_t qs[q4_0_values_per_block / 2];
};

static_assert(std::is_standard_layout_v<Q4_0Block>);
static_assert(offsetof(Q4_0Block, d) == 0);
static_assert(offsetof(Q4_0Block, qs) == sizeof(std::uint16_t));
static_assert(sizeof(Q4_0Block) == 18);

// Eight Q4_0 rows for the same 32-column block. Scales are contiguous and
// quants are interleaved in eight-byte chunks. Nibbles retain the canonical
// unsigned offset-binary encoding (q + 8); the integer dot subtracts the
// prepared activation sum times eight. No per-block sign-bit XOR is needed.
// This runtime layout is also the q4_*_cpu_x8 artifact encoding. It is
// size-neutral: eight canonical blocks and one packed block are 144 bytes.
struct Q4_0BlockX8 {
  std::uint16_t d[8];
  std::uint8_t qs[8 * q4_0_values_per_block / 2];
};

static_assert(std::is_standard_layout_v<Q4_0BlockX8>);
static_assert(offsetof(Q4_0BlockX8, d) == 0);
static_assert(offsetof(Q4_0BlockX8, qs) == 16);
static_assert(sizeof(Q4_0BlockX8) == 8 * sizeof(Q4_0Block));

// Four token rows for the same 32-column activation block. Quant bytes are
// token-major so prefill consumes each 32-byte row
// with two contiguous 16-byte loads. This is transient scratch, not GGUF.
struct Q8_0BlockX4 {
  float scales[4];
  std::int16_t sums[4];
  std::int8_t qs[4 * q8_0_values_per_block];
};

static_assert(std::is_standard_layout_v<Q8_0BlockX4>);
static_assert(offsetof(Q8_0BlockX4, scales) == 0);
static_assert(offsetof(Q8_0BlockX4, sums) == 16);
static_assert(offsetof(Q8_0BlockX4, qs) == 24);
static_assert(sizeof(Q8_0BlockX4) == 152);

// Compact single-token preparation; arrays keep the SIMD accessors shared
// with X4 prefill. Two tail-padding bytes are not serialized or consumed.
struct Q8_0BlockX1 {
  float scales[1];
  std::int16_t sums[1];
  std::int8_t qs[q8_0_values_per_block];
};
static_assert(std::is_standard_layout_v<Q8_0BlockX1>);
static_assert(sizeof(Q8_0BlockX1) == 40);
static_assert(offsetof(Q8_0BlockX1, qs) == 6);

inline constexpr std::size_t q4_0_packed_rows = 8;
inline constexpr std::size_t q8_0_packed_vectors = 4;

struct Q4_0ArgmaxResult {
  float value;
  std::size_t index;
};

// row_count must be divisible by eight. packed contains
// (row_count / 8) * blocks_per_row elements.
void q4_0_pack_rows_8(
  const Q4_0Block * canonical,
  Q4_0BlockX8 * packed,
  std::size_t row_count,
  std::size_t blocks_per_row) noexcept;

// vector_count must be divisible by four. packed contains
// (vector_count / 4) * blocks_per_vector elements.
void q8_0_pack_vectors_4(
  const Q8_0Block * canonical,
  Q8_0BlockX4 * packed,
  std::size_t vector_count,
  std::size_t blocks_per_vector) noexcept;

// Directly quantizes token-major F32 activations into the transient 4-row
// layout, avoiding an intermediate 34-byte Q8_0 array and a repacking pass.
void q8_0_quantize_vectors_4(
  const float * input,
  Q8_0BlockX4 * packed,
  std::size_t vector_count,
  std::size_t blocks_per_vector,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// Prepares one decode vector in lane zero of each block. The FP16-rounded
// scale and exact int8 sum are computed once and reused across all output rows.
void q8_0_quantize_vector_1(
  const float * input,
  Q8_0BlockX1 * packed,
  std::size_t blocks_per_vector,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

void q4_0_dequantize(
  const Q4_0Block * input,
  float * output,
  std::size_t block_count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

void q4_0_scales_to_f32(
  const Q4_0Block * input,
  float * output,
  std::size_t block_count) noexcept;

[[nodiscard]] float q4_0_dot_q8_0(
  const Q4_0Block * weights,
  const Q8_0Block * activations,
  std::size_t block_count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

void q4_0_matvec_q8_0(
  const Q4_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// vectors and output are token-major. The optional F32 scale sidecars avoid
// repeatedly converting binary16 scales in batched prefill kernels.
void q4_0_matmul_q8_0(
  const Q4_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  std::size_t row_count,
  std::size_t vector_count,
  std::size_t blocks_per_row,
  std::size_t output_row_stride,
  Q8_0Backend backend = Q8_0Backend::auto_select,
  const float * vector_scales = nullptr,
  const float * matrix_scales = nullptr) noexcept;

// Packed 4-token by 8-output-row prefill path. row_count and vector_count
// must be divisible by eight and four respectively. Output remains
// token-major and may have a larger row stride for executor partitions.
void q4_0_packed_matmul_q8_0(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX4 * vectors,
  float * output,
  std::size_t row_count,
  std::size_t vector_count,
  std::size_t blocks_per_row,
  std::size_t output_row_stride,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

void q4_0_packed_matvec_q8_0(
  const Q4_0BlockX8 * matrix,
  const Q8_0Block * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

void q4_0_packed_matvec_prepared_q8_0(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX1 * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

[[nodiscard]] Q4_0ArgmaxResult q4_0_packed_matvec_prepared_q8_0_argmax(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX1 * vector,
  const int * token_counts,
  float repetition_penalty,
  std::size_t row_offset,
  std::size_t row_count,
  std::size_t blocks_per_row,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// Dequantizes one logical row from an eight-row packed matrix. This is mainly
// used for token-embedding gather, where only one short row is touched.
void q4_0_packed_dequantize_row(
  const Q4_0BlockX8 * matrix,
  std::size_t row_index,
  float * output,
  std::size_t blocks_per_row) noexcept;

} // namespace qwen35x::cpu
