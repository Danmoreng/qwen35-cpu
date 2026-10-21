// SPDX-License-Identifier: MIT
#pragma once
#include <wasm_simd128.h>
namespace qwen35x::cpu::detail {
// Four widened FP16 bit patterns to FP32, including signed zero, subnormals,
// infinities and NaN payloads. All subnormal integer conversions are exact.
inline v128_t wasm_half4_to_float(v128_t h) noexcept {
  const auto magnitude = wasm_v128_and(h, wasm_i32x4_splat(0x7fff));
  const auto exponent = wasm_v128_and(h, wasm_i32x4_splat(0x7c00));
  const auto sign = wasm_i32x4_shl(wasm_v128_and(h, wasm_i32x4_splat(0x8000)), 16);
  const auto normal = wasm_i32x4_add(wasm_i32x4_shl(magnitude, 13), wasm_i32x4_splat(0x38000000));
  const auto zero_exp = wasm_i32x4_eq(exponent, wasm_i32x4_splat(0));
  const auto max_exp = wasm_i32x4_eq(exponent, wasm_i32x4_splat(0x7c00));
  if (!wasm_v128_any_true(wasm_v128_or(zero_exp, max_exp))) return wasm_v128_or(normal, sign);
  const auto mantissa = wasm_v128_and(h, wasm_i32x4_splat(0x03ff));
  const auto subnormal = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(mantissa), wasm_f32x4_splat(0x1p-24f));
  const auto special = wasm_v128_or(wasm_i32x4_shl(mantissa, 13), wasm_i32x4_splat(0x7f800000));
  auto result = wasm_v128_bitselect(subnormal, normal, zero_exp);
  result = wasm_v128_bitselect(special, result, max_exp);
  return wasm_v128_or(result, sign);
}
}
