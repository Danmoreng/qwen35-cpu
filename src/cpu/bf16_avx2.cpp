#include "bf16_avx2.h"

#include <immintrin.h>

namespace qwen35x::cpu {
namespace {

float sum_lanes(__m256 value) noexcept {
  const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
  const __m128 pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
  return _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 1)));
}

template<std::size_t Vectors>
void multiply_rows(const std::uint16_t *weights, const float *input, float *output,
                   std::size_t rows, std::size_t columns,
                   std::size_t begin, std::size_t end) noexcept {
  for (std::size_t row = begin; row < end; ++row) {
    __m256 sum[Vectors];
    for (auto &lanes : sum) lanes = _mm256_setzero_ps();
    const auto *weight = weights + row * columns;
    for (std::size_t col = 0; col < columns; col += 8) {
      const auto bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(weight + col));
      const auto value = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
      for (std::size_t v = 0; v < Vectors; ++v)
        sum[v] = _mm256_fmadd_ps(value, _mm256_loadu_ps(input + v * columns + col), sum[v]);
    }
    for (std::size_t v = 0; v < Vectors; ++v)
      output[v * rows + row] = sum_lanes(sum[v]);
  }
}

} // namespace

void bf16_f32_matmul_rows_avx2(const std::uint16_t *weights, const float *input, float *output,
                              std::size_t rows, std::size_t columns, std::size_t vectors,
                              std::size_t begin, std::size_t end) noexcept {
  while (vectors >= 4) {
    multiply_rows<4>(weights, input, output, rows, columns, begin, end);
    input += 4 * columns;
    output += 4 * rows;
    vectors -= 4;
  }
  switch (vectors) {
    case 3: multiply_rows<3>(weights, input, output, rows, columns, begin, end); break;
    case 2: multiply_rows<2>(weights, input, output, rows, columns, begin, end); break;
    case 1: multiply_rows<1>(weights, input, output, rows, columns, begin, end); break;
  }
}

} // namespace qwen35x::cpu
