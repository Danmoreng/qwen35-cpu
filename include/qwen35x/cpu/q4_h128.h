#pragma once

#include "qwen35x/cpu/q4_0.h"

#include <cstddef>
#include <cstdint>

namespace qwen35x::cpu {

struct Q4H128SignBlock { std::uint64_t words[2]; };

// Prepare once per seed/width. The immutable blocks can be shared by workers.
// Optional signs in the preparation functions must contain column_count/128
// blocks generated with the supplied seed; nullptr computes signs on demand.
void q4_h128_prepare_signs(Q4H128SignBlock * output, std::size_t block_count,
                          std::uint64_t sign_seed) noexcept;

inline constexpr std::size_t q4_h128_transform_size = 128;
inline constexpr std::size_t q4_h128_q4_blocks_per_transform =
  q4_h128_transform_size / q4_0_values_per_block;
inline constexpr float q4_h128_inverse_sqrt_size = 0.08838834764831844055F;
inline constexpr std::uint64_t q4_h128_default_sign_seed =
  UINT64_C(0x5148333548313238);

// Applies R = H_128 D / sqrt(128), where D is a deterministic randomized-sign
// diagonal. transform_block_index is the block index within an activation
// vector and therefore deliberately does not depend on a tensor name: all
// projections sharing an activation must use the same basis.
void q4_h128_transform_block(
  const float * input,
  float * output,
  std::size_t transform_block_index,
  std::uint64_t sign_seed = q4_h128_default_sign_seed,
  Q8_0Backend backend = Q8_0Backend::scalar) noexcept;

void q4_h128_transform_block_inplace(
  float * values,
  std::size_t transform_block_index,
  std::uint64_t sign_seed = q4_h128_default_sign_seed,
  Q8_0Backend backend = Q8_0Backend::scalar) noexcept;

// rows are row-major and columns must be a non-zero multiple of 128. The
// transform block index restarts at zero for each row.
[[nodiscard]] bool q4_h128_transform_rows(
  const float * input,
  float * output,
  std::size_t row_count,
  std::size_t column_count,
  std::uint64_t sign_seed = q4_h128_default_sign_seed,
  Q8_0Backend backend = Q8_0Backend::scalar) noexcept;

// Runtime-only variant that omits 1/sqrt(128). The caller must fold
// q4_h128_inverse_sqrt_size into the subsequent activation scales.
[[nodiscard]] bool q4_h128_transform_rows_unscaled(
  const float * input,
  float * output,
  std::size_t row_count,
  std::size_t column_count,
  std::uint64_t sign_seed = q4_h128_default_sign_seed,
  Q8_0Backend backend = Q8_0Backend::scalar) noexcept;

// Quantizes already-transformed values into the engine's signed Q4 nibble
// layout with one binary16 symmetric scale per 32 values.
void q4_h128_quantize_transformed(
  const float * input,
  Q4_0Block * output,
  std::size_t q4_block_count) noexcept;

// Direct BF16/F32 conversion primitive for an eligible projection matrix.
[[nodiscard]] bool q4_h128_quantize_matrix(
  const float * input,
  Q4_0Block * output,
  std::size_t row_count,
  std::size_t column_count,
  std::uint64_t sign_seed = q4_h128_default_sign_seed) noexcept;

// Transforms one activation vector and quantizes the result to the existing
// Q8_0 dot-product layout. transformed_scratch holds column_count floats.
[[nodiscard]] bool q4_h128_prepare_activation(
  const float * input,
  float * transformed_scratch,
  Q8_0Block * output,
  std::size_t column_count,
  std::uint64_t sign_seed = q4_h128_default_sign_seed,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// Fused single-token decode preparation; column_count must be divisible by
// 128. Quantization rounds the unnormalized Q8 scale to FP16 before folding
// the H128 normalization into its prepared FP32 scale.
[[nodiscard]] bool q4_h128_prepare_activation_1(
  const float * input, Q8_0BlockX1 * output, std::size_t column_count,
  std::uint64_t sign_seed = q4_h128_default_sign_seed,
  Q8_0Backend backend = Q8_0Backend::auto_select,
  const Q4H128SignBlock * signs = nullptr) noexcept;

// Transforms groups of four token-major activation rows and immediately
// quantizes them into the packed Q8_0 prefill layout. Keeping each transformed
// H128 block in a small local buffer avoids a batch-sized FP32 scratch pass.
// vector_count must be divisible by four and column_count by 128.
[[nodiscard]] bool q4_h128_prepare_activations_4(
  const float * input,
  Q8_0BlockX4 * output,
  std::size_t vector_count,
  std::size_t column_count,
  std::uint64_t sign_seed = q4_h128_default_sign_seed,
  Q8_0Backend backend = Q8_0Backend::auto_select,
  const Q4H128SignBlock * signs = nullptr) noexcept;

} // namespace qwen35x::cpu
