// SPDX-License-Identifier: MIT
#include "q4_h128_internal.h"
#include <wasm_simd128.h>
namespace qwen35x::cpu::detail {
// Same butterfly order as the portable reference; signs only flip sign bits.
void q4_h128_transform_block_wasm_signed(const float *input, float *output,
                                          const std::uint64_t *signs) noexcept {
  v128_t values[32];
  for (int i = 0; i < 32; ++i) {
    const auto bits = wasm_i32x4_splat(static_cast<int>(signs[i / 16] >> (i % 16 * 4)));
    const auto negative = wasm_i32x4_ne(wasm_v128_and(bits, wasm_i32x4_make(1, 2, 4, 8)), wasm_i32x4_splat(0));
    auto x = wasm_v128_xor(wasm_v128_load(input + i * 4),
      wasm_v128_and(negative, wasm_i32x4_splat(static_cast<int>(0x80000000U))));
    auto a = wasm_i32x4_shuffle(x, x, 0, 0, 2, 2);
    auto b = wasm_i32x4_shuffle(x, x, 1, 1, 3, 3);
    auto add = wasm_f32x4_add(a, b), sub = wasm_f32x4_sub(a, b);
    x = wasm_i32x4_shuffle(add, sub, 0, 4, 2, 6);
    a = wasm_i32x4_shuffle(x, x, 0, 1, 0, 1);
    b = wasm_i32x4_shuffle(x, x, 2, 3, 2, 3);
    add = wasm_f32x4_add(a, b); sub = wasm_f32x4_sub(a, b);
    values[i] = wasm_i32x4_shuffle(add, sub, 0, 1, 4, 5);
  }
  for (int stride = 1; stride < 32; stride *= 2)
    for (int base = 0; base < 32; base += stride * 2)
      for (int lane = 0; lane < stride; ++lane) {
        const auto a = values[base + lane], b = values[base + stride + lane];
        values[base + lane] = wasm_f32x4_add(a, b);
        values[base + stride + lane] = wasm_f32x4_sub(a, b);
      }
  for (int i = 0; i < 32; ++i) wasm_v128_store(output + i * 4, values[i]);
}
}
