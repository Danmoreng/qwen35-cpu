// SPDX-License-Identifier: MIT
// SIMD128 fallback plus an optional bounded integer relaxed-dot specialization.
#include "q4_dot4_internal.h"
#include "wasm_fma.h"
#include <wasm_simd128.h>
#include <cstring>

namespace qwen35x::cpu::detail {
namespace {
v128_t dot_four_rows(v128_t w, v128_t a) noexcept {
  const auto low = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(w), wasm_i16x8_extend_low_i8x16(a));
  const auto high = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(w), wasm_i16x8_extend_high_i8x16(a));
  return wasm_i32x4_add(wasm_i32x4_shuffle(low, high, 0, 2, 4, 6),
                        wasm_i32x4_shuffle(low, high, 1, 3, 5, 7));
}
v128_t broadcast_four(const std::int8_t *p) noexcept {
  std::int32_t bytes; std::memcpy(&bytes, p, sizeof(bytes)); return wasm_i32x4_splat(bytes);
}
template <int Tokens, typename Block>
void tile(const Q4_0BlockX8 *weights, const Block *vectors, std::size_t blocks,
          std::size_t first_lane, float *output, std::size_t stride) noexcept {
  v128_t acc[Tokens][2];
  for (auto &a : acc) for (auto &v : a) v = wasm_f32x4_splat(0);
  for (std::size_t b = 0; b < blocks; ++b) {
    const auto &w = weights[b]; const auto &a = vectors[b];
    v128_t dots[Tokens][2];
    for (auto &d : dots) for (auto &v : d) v = wasm_i32x4_splat(0);
    for (int group = 0; group < 4; ++group) for (int half = 0; half < 2; ++half) {
      const auto packed = wasm_v128_load(w.qs + 32 * group + 16 * half);
      auto low = wasm_v128_and(packed, wasm_i8x16_splat(15));
      auto high = wasm_u8x16_shr(packed, 4);
#if !QWEN35X_HAS_WASM_RELAXED_DOT
      low = wasm_i8x16_sub(low, wasm_i8x16_splat(8));
      high = wasm_i8x16_sub(high, wasm_i8x16_splat(8));
#endif
      for (int t = 0; t < Tokens; ++t) {
        const auto *q = a.qs + (first_lane + t) * 32 + group * 8;
#if QWEN35X_HAS_WASM_RELAXED_DOT
        // Second operand is unsigned 0..15, below the 7-bit limit. Products
        // of adjacent elements sum to at most 3840, so saturation is impossible.
        // Thus the relaxed integer instruction has a single exact result.
        auto dot = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(broadcast_four(q), low, dots[t][half]);
        dots[t][half] = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(broadcast_four(q + 4), high, dot);
#else
        dots[t][half] = wasm_i32x4_add(dots[t][half], wasm_i32x4_add(
          dot_four_rows(low, broadcast_four(q)), dot_four_rows(high, broadcast_four(q + 4))));
#endif
      }
    }
    float scales[8]; for (int r = 0; r < 8; ++r) scales[r] = half_to_float(w.d[r]);
    for (int half = 0; half < 2; ++half) {
      const auto ws = wasm_v128_load(scales + half * 4);
      for (int t = 0; t < Tokens; ++t) {
#if QWEN35X_HAS_WASM_RELAXED_DOT
        dots[t][half] = wasm_i32x4_sub(dots[t][half], wasm_i32x4_splat(8 * a.sums[first_lane + t]));
#endif
        const auto scale = wasm_f32x4_mul(ws, wasm_f32x4_splat(a.scales[first_lane + t]));
        acc[t][half] = wasm_fma_f32x4(wasm_f32x4_convert_i32x4(dots[t][half]), scale, acc[t][half]);
      }
    }
  }
  for (int t = 0; t < Tokens; ++t) for (int half = 0; half < 2; ++half)
    wasm_v128_store(output + t * stride + half * 4, acc[t][half]);
}
}
void q4_dot4_matvec_wasm(const Q4_0BlockX8 *w, const Q8_0BlockX1 *a, float *out,
                         std::size_t rows, std::size_t blocks) noexcept {
  for (std::size_t row = 0; row < rows; row += 8) tile<1>(w + row / 8 * blocks, a, blocks, 0, out + row, rows);
}
void q4_dot4_matmul_wasm(const Q4_0BlockX8 *w, const Q8_0BlockX4 *a, float *out,
                         std::size_t rows, std::size_t tokens, std::size_t blocks, std::size_t stride) noexcept {
  std::size_t t = 0;
  for (; t + 4 <= tokens; t += 4) for (std::size_t row = 0; row < rows; row += 8)
    tile<4>(w + row / 8 * blocks, a + t / 4 * blocks, blocks, 0, out + t * stride + row, stride);
  for (; t < tokens; ++t) for (std::size_t row = 0; row < rows; row += 8)
    tile<1>(w + row / 8 * blocks, a + t / 4 * blocks, blocks, t % 4, out + t * stride + row, stride);
}
}
