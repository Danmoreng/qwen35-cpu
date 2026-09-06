#include "q4_dot4_internal.h"
#include <cmath>
#include <limits>
namespace qwen35x::cpu {
void q4_dot4_pack_rows_8(const Q4_0Block * source, Q4_0BlockX8 * output,
                        std::size_t rows, std::size_t blocks) noexcept {
  for (std::size_t rt = 0; rt < rows / 8; ++rt) for (std::size_t b = 0; b < blocks; ++b) {
    auto & dst = output[rt * blocks + b];
    for (std::size_t r = 0; r < 8; ++r) {
      const auto & src = source[(rt * 8 + r) * blocks + b];
      dst.d[r] = src.d;
      auto nibble = [&](std::size_t k) { return (src.qs[k % 16] >> (4 * (k / 16))) & 15; };
      for (std::size_t t = 0; t < 4; ++t) for (std::size_t j = 0; j < 4; ++j)
        dst.qs[32*t+4*r+j] = static_cast<std::uint8_t>(nibble(8*t+j) | (nibble(8*t+4+j) << 4));
    }
  }
}
void q4_dot4_unpack_rows_8(const Q4_0BlockX8 * source, Q4_0Block * output,
                          std::size_t rows, std::size_t blocks) noexcept {
  for (std::size_t row = 0; row < rows; ++row) for (std::size_t b = 0; b < blocks; ++b) {
    const auto & src = source[(row / 8) * blocks + b];
    auto & dst = output[row * blocks + b];
    dst = {}; dst.d = src.d[row % 8];
    for (std::size_t k = 0; k < 32; ++k) {
      const int u = (src.qs[32*(k/8)+4*(row%8)+k%4] >> (4*((k%8)/4))) & 15;
      dst.qs[k%16] |= static_cast<std::uint8_t>(u << (4*(k/16)));
    }
  }
}
void q4_dot4_dequantize_row(const Q4_0BlockX8 * matrix, std::size_t row, float * out,
                          std::size_t blocks) noexcept {
  for (std::size_t b = 0; b < blocks; ++b) {
    const auto & w = matrix[(row/8)*blocks+b];
    const float scale = detail::half_to_float(w.d[row%8]);
    for (std::size_t k = 0; k < 32; ++k) {
      const int u = (w.qs[32*(k/8)+4*(row%8)+k%4] >> (4*((k%8)/4))) & 15;
      out[b*32+k] = scale * static_cast<float>(u-8);
    }
  }
}
namespace {
template <typename Block>
float scalar_row(const Q4_0BlockX8 * matrix, const Block * vector, std::size_t lane,
                 std::size_t row, std::size_t blocks) noexcept {
  float result = 0;
  for (std::size_t b = 0; b < blocks; ++b) {
    const auto & w = matrix[(row/8)*blocks+b]; const auto & a = vector[b];
    int dot = 0;
    for (std::size_t k = 0; k < 32; ++k) {
      const int u = (w.qs[32*(k/8)+4*(row%8)+k%4] >> (4*((k%8)/4))) & 15;
      dot += (u-8) * static_cast<int>(a.qs[lane*32+k]);
    }
    result = std::fma(static_cast<float>(dot), detail::half_to_float(w.d[row%8])*a.scales[lane], result);
  }
  return result;
}
}
void q4_dot4_matvec(const Q4_0BlockX8 * matrix, const Q8_0BlockX1 * vector, float * output, std::size_t rows, std::size_t blocks, Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX512_VNNI_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx512_vnni) return detail::q4_dot4_matvec_evex(matrix, vector, output, rows, blocks);
#endif
#if QWEN35X_Q8_0_HAS_AVX_VNNI_TU
  if (q8_0_backend_uses_avx_vnni(backend)) return detail::q4_dot4_matvec_vex(matrix, vector, output, rows, blocks);
#endif
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) return detail::q4_dot4_matvec_avx2(matrix, vector, output, rows, blocks);
#endif
  for (std::size_t r = 0; r < rows; ++r) output[r] = scalar_row(matrix, vector, 0, r, blocks);
}
void q4_dot4_matmul(const Q4_0BlockX8 * matrix, const Q8_0BlockX4 * vectors, float * output, std::size_t rows, std::size_t count, std::size_t blocks, std::size_t stride, Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX512_VNNI_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx512_vnni) return detail::q4_dot4_matmul_evex(matrix, vectors, output, rows, count, blocks, stride);
#endif
#if QWEN35X_Q8_0_HAS_AVX_VNNI_TU
  if (q8_0_backend_uses_avx_vnni(backend)) return detail::q4_dot4_matmul_vex(matrix, vectors, output, rows, count, blocks, stride);
#endif
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) return detail::q4_dot4_matmul_avx2(matrix, vectors, output, rows, count, blocks, stride);
#endif
  for (std::size_t t = 0; t < count; ++t) for (std::size_t r = 0; r < rows; ++r)
    output[t*stride+r] = scalar_row(matrix, vectors+(t/4)*blocks, t%4, r, blocks);
}
Q4_0ArgmaxResult q4_dot4_argmax(const Q4_0BlockX8 * matrix, const Q8_0BlockX1 * vector, const int * counts, float penalty, std::size_t offset, std::size_t rows, std::size_t blocks, Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX512_VNNI_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx512_vnni) return detail::q4_dot4_argmax_evex(matrix, vector, counts, penalty, offset, rows, blocks);
#endif
#if QWEN35X_Q8_0_HAS_AVX_VNNI_TU
  if (q8_0_backend_uses_avx_vnni(backend)) return detail::q4_dot4_argmax_vex(matrix, vector, counts, penalty, offset, rows, blocks);
#endif
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) return detail::q4_dot4_argmax_avx2(matrix, vector, counts, penalty, offset, rows, blocks);
#endif
  Q4_0ArgmaxResult best{-std::numeric_limits<float>::infinity(), offset};
  for (std::size_t r = 0; r < rows; ++r) {
    float value = scalar_row(matrix, vector, 0, r, blocks);
    if (counts && counts[offset+r] > 0 && penalty > 1.0F) value = value > 0 ? value/penalty : value*penalty;
    if (value > best.value) best = {value, offset+r};
  }
  return best;
}
}
