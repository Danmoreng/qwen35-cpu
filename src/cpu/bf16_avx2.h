#pragma once

#include <cstddef>
#include <cstdint>

namespace qwen35x::cpu {

void bf16_f32_matmul_rows_avx2(
  const std::uint16_t *weights, const float *input, float *output,
  std::size_t rows, std::size_t columns, std::size_t vectors,
  std::size_t begin, std::size_t end) noexcept;

} // namespace qwen35x::cpu
