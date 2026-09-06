#pragma once

#include "qwen35x/cpu/q8_0.h"

#include <cstddef>

namespace qwen35x::cpu {

void add_f32(
  const float * lhs,
  const float * rhs,
  float * output,
  std::size_t count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// Applies split-half rotary position embedding to head_count rows. cosine and
// sine each contain rope_dim / 2 values for the requested absolute position.
void rope_f32(
  float * values,
  std::size_t head_count,
  std::size_t head_dim,
  std::size_t rope_dim,
  const float * cosine,
  const float * sine,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// Stateful depthwise causal convolution. State contains kernel_size - 1
// channel-major slots in ring order; ring_index identifies the oldest slot.
// Weights use [kernel][channel] layout. The output includes SiLU activation.
void causal_conv1d_silu_f32(
  float * state,
  std::size_t ring_index,
  const float * input,
  std::size_t input_stride,
  const float * kernel_major_weights,
  float * output,
  std::size_t batch_size,
  std::size_t channel_count,
  std::size_t kernel_size,
  std::size_t channel_begin,
  std::size_t channel_end,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// Applies RMS normalization to row_count rows. The width-element weight is
// shared by all rows and weight_offset supports Qwen's (1 + weight) form.
void rms_norm_f32(
  const float * input,
  const float * weight,
  float * output,
  std::size_t row_count,
  std::size_t width,
  float eps,
  float weight_offset,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

void l2_normalize_f32(
  float * values,
  std::size_t row_count,
  std::size_t width,
  float eps,
  float output_scale = 1.0F,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

void silu_f32(
  const float * input,
  float * output,
  std::size_t count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// Computes output[i] = silu(gate[i]) * up[i]. The AVX2 implementation uses
// the same bounded exp approximation as the optimized attention kernels.
void silu_mul_f32(
  const float * gate,
  const float * up,
  float * output,
  std::size_t count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

} // namespace qwen35x::cpu
