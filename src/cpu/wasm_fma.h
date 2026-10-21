// SPDX-License-Identifier: MIT
#pragma once
#include <wasm_simd128.h>
#include <cmath>
#include <limits>
namespace qwen35x::cpu::detail {
// The optional direct path requires a fused-semantics probe in the browser.
// The portable path reproduces FP32 fused multiply/add through FP64.
// FP64 holds the full FP32 product. Exceptional double-rounding ties and
// subnormal results take the standard library's correctly rounded path.
inline v128_t wasm_fma_f32x4(v128_t x, v128_t y, v128_t z) noexcept {
#if QWEN35X_HAS_WASM_RELAXED_FMA
  return wasm_f32x4_relaxed_madd(x, y, z);
#else
  const auto high = [](v128_t v) { return wasm_i32x4_shuffle(v, v, 2, 3, 2, 3); };
  const auto low = wasm_f64x2_add(wasm_f64x2_mul(wasm_f64x2_promote_low_f32x4(x),
    wasm_f64x2_promote_low_f32x4(y)), wasm_f64x2_promote_low_f32x4(z));
  const auto hi = wasm_f64x2_add(wasm_f64x2_mul(wasm_f64x2_promote_low_f32x4(high(x)),
    wasm_f64x2_promote_low_f32x4(high(y))), wasm_f64x2_promote_low_f32x4(high(z)));
  const auto exceptional = [](v128_t v) {
    const auto tie = wasm_i64x2_eq(wasm_v128_and(v, wasm_i64x2_splat(0x1fffffff)), wasm_i64x2_splat(0x10000000));
    const auto a = wasm_f64x2_abs(v);
    const auto tiny = wasm_v128_and(wasm_f64x2_gt(a, wasm_f64x2_splat(0)),
      wasm_f64x2_lt(a, wasm_f64x2_splat(std::numeric_limits<float>::min())));
    return wasm_i64x2_bitmask(wasm_v128_or(tie, tiny)) != 0;
  };
  if (exceptional(low) || exceptional(hi)) {
    float a[4], b[4], c[4]; wasm_v128_store(a, x); wasm_v128_store(b, y); wasm_v128_store(c, z);
    for (int i = 0; i < 4; ++i) c[i] = std::fma(a[i], b[i], c[i]);
    return wasm_v128_load(c);
  }
  return wasm_i32x4_shuffle(wasm_f32x4_demote_f64x2_zero(low), wasm_f32x4_demote_f64x2_zero(hi), 0, 1, 4, 5);
#endif
}
}
