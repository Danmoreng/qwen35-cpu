// SPDX-License-Identifier: MIT
#include "delta_net_wasm.h"
#include "wasm_fma.h"
#include <wasm_simd128.h>
namespace qwen35x::cpu::detail {
namespace {
float sum(v128_t x) noexcept {
  x = wasm_f32x4_add(x, wasm_i32x4_shuffle(x, x, 2, 3, 0, 1));
  return wasm_f32x4_extract_lane(x, 0) + wasm_f32x4_extract_lane(x, 1);
}
float update(float *state, const float *q, const float *k, float v, float alpha, float beta,
             std::size_t width) noexcept {
  auto dot = wasm_f32x4_splat(0), dot_hi = dot; const auto decay = wasm_f32x4_splat(alpha);
  std::size_t c = 0;
  for (; c + 8 <= width; c += 8) {
    const auto s = wasm_f32x4_mul(wasm_v128_load(state + c), decay);
    const auto sh = wasm_f32x4_mul(wasm_v128_load(state + c + 4), decay);
    wasm_v128_store(state + c, s);
    wasm_v128_store(state + c + 4, sh);
    dot = wasm_fma_f32x4(s, wasm_v128_load(k + c), dot);
    dot_hi = wasm_fma_f32x4(sh, wasm_v128_load(k + c + 4), dot_hi);
  }
  float sk = sum(wasm_f32x4_add(dot, dot_hi));
  for (; c < width; ++c) { state[c] *= alpha; sk += state[c] * k[c]; }
  const float delta = (v - sk) * beta; const auto d = wasm_f32x4_splat(delta);
  dot = dot_hi = wasm_f32x4_splat(0); c = 0;
  for (; c + 8 <= width; c += 8) {
    const auto s = wasm_fma_f32x4(d, wasm_v128_load(k + c), wasm_v128_load(state + c));
    const auto sh = wasm_fma_f32x4(d, wasm_v128_load(k + c + 4), wasm_v128_load(state + c + 4));
    wasm_v128_store(state + c, s);
    wasm_v128_store(state + c + 4, sh);
    dot = wasm_fma_f32x4(s, wasm_v128_load(q + c), dot);
    dot_hi = wasm_fma_f32x4(sh, wasm_v128_load(q + c + 4), dot_hi);
  }
  float sq = sum(wasm_f32x4_add(dot, dot_hi));
  for (; c < width; ++c) { state[c] += delta * k[c]; sq += state[c] * q[c]; }
  return sq;
}
}
void gated_delta_net_update_rows_wasm(float *state, const float *q, const float *k, const float *v,
  const float *alpha, const float *beta, float *output, std::size_t keys, std::size_t values,
  std::size_t begin, std::size_t end) noexcept {
  for (std::size_t row = begin; row < end; ++row) {
    const auto h = row / values;
    output[row] = update(state + row * keys, q + h * keys, k + h * keys, v[row], alpha[h], beta[h], keys);
  }
}
void gated_delta_net_update_batch_rows_wasm(float *state, const float *q, const float *k, const float *v,
  const float *alpha, const float *beta, float *output, std::size_t tokens, std::size_t heads,
  std::size_t keys, std::size_t values, std::size_t begin, std::size_t end) noexcept {
  for (std::size_t h = begin; h < end; ++h) for (std::size_t t = 0; t < tokens; ++t) {
    const auto ht = h * tokens + t;
    for (std::size_t r = 0; r < values; ++r) {
      const auto row = h * values + r;
      output[t * heads * values + row] = update(state + row * keys, q + ht * keys,
        k + ht * keys, v[ht * values + r], alpha[ht], beta[ht], keys);
    }
  }
}
}
