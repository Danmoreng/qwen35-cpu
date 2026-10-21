// Exercise the browser's auto-dispatch against the portable scalar reference.
#include "qwen35x/cpu/q4_dot4.h"
#include "qwen35x/cpu/q4_h128.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <bit>
#ifdef __wasm_simd128__
#include "../src/cpu/wasm_fma.h"
#include "../src/cpu/wasm_half.h"
#include "../src/cpu/q8_0_internal.h"
#endif
using namespace qwen35x::cpu;
int main() {
  std::mt19937 rng(7138);
  for (std::size_t width : {128U, 1024U, 3584U}) {
    std::vector<float> input(width * 4);
    for (auto &x : input) x = (static_cast<int>(rng() % 2001) - 1000) * .001f;
    std::vector<Q4H128SignBlock> signs(width / 128);
    q4_h128_prepare_signs(signs.data(), signs.size(), q4_h128_default_sign_seed);
    std::vector<Q8_0BlockX1> one(width / 32), one_ref(one.size());
    if (!q4_h128_prepare_activation_1(input.data(), one.data(), width, q4_h128_default_sign_seed, Q8_0Backend::auto_select, signs.data()) ||
        !q4_h128_prepare_activation_1(input.data(), one_ref.data(), width, q4_h128_default_sign_seed, Q8_0Backend::scalar)) return 4;
    for (std::size_t b = 0; b < one.size(); ++b)
      if (std::memcmp(&one[b], &one_ref[b], offsetof(Q8_0BlockX1, qs) + 32)) return 4;
    std::vector<Q8_0BlockX4> four(width / 32), four_ref(four.size());
    if (!q4_h128_prepare_activations_4(input.data(), four.data(), 4, width, q4_h128_default_sign_seed, Q8_0Backend::auto_select, signs.data()) ||
        !q4_h128_prepare_activations_4(input.data(), four_ref.data(), 4, width, q4_h128_default_sign_seed, Q8_0Backend::scalar)) return 5;
    if (std::memcmp(four.data(), four_ref.data(), four.size() * sizeof(four[0]))) return 5;
  }
#ifdef __wasm_simd128__
  for (std::uint32_t first = 0; first < 65536; ++first) {
    float actual[4];
    const std::uint16_t patterns[4] = {static_cast<std::uint16_t>(first),
      static_cast<std::uint16_t>(first + 0x400), static_cast<std::uint16_t>(first + 0x7c00),
      static_cast<std::uint16_t>(first + 0x8000)};
    const auto bits = wasm_i32x4_make(patterns[0], patterns[1], patterns[2], patterns[3]);
    wasm_v128_store(actual, detail::wasm_half4_to_float(bits));
    for (int i = 0; i < 4; ++i)
      if (std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(
          detail::half_to_float(patterns[i]))) return 6;
  }
  for (int n = 0; n < 10000; ++n) {
    float a[4], b[4], c[4], actual[4];
    for (int i = 0; i < 4; ++i) {
      a[i] = std::bit_cast<float>(static_cast<std::uint32_t>(rng()));
      b[i] = std::bit_cast<float>(static_cast<std::uint32_t>(rng()));
      c[i] = std::bit_cast<float>(static_cast<std::uint32_t>(rng()));
    }
    wasm_v128_store(actual, detail::wasm_fma_f32x4(wasm_v128_load(a), wasm_v128_load(b), wasm_v128_load(c)));
    for (int i = 0; i < 4; ++i) {
      const auto expected = std::fma(a[i], b[i], c[i]);
      if (std::isnan(expected) && std::isnan(actual[i])) continue;
      if (std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(expected)) return 3;
    }
  }
#endif
  for (int pattern = 0; pattern < 3; ++pattern)
  for (std::size_t blocks : {1U, 7U, 32U}) for (std::size_t rows : {8U, 24U})
    for (std::size_t tokens : {1U, 3U, 4U, 7U, 8U, 9U, 12U, 15U, 16U, 31U, 32U}) {
      std::vector<Q4_0BlockX8> w(rows / 8 * blocks);
      for (auto &b : w) { for (auto &d : b.d) d = pattern == 2 ? 0xbc00 : 0x2810; for (auto &q : b.qs) q = pattern == 0 ? rng() : pattern == 1 ? 255 : 0; }
      std::vector<Q8_0BlockX4> a(((tokens + 3) / 4) * blocks);
      for (auto &b : a) for (int t = 0; t < 4; ++t) {
        b.scales[t] = .003f; b.sums[t] = 0;
        for (int k = 0; k < 32; ++k) { const auto q = static_cast<std::int8_t>(pattern == 0 ? static_cast<int>(rng()) : pattern == 1 ? -128 : 127); b.qs[t*32+k] = q; b.sums[t] += q; }
      }
      const auto stride = rows + 5;
      std::vector<float> ref(stride * tokens, -123), got(ref);
      q4_dot4_matmul(w.data(), a.data(), ref.data(), rows, tokens, blocks, stride, Q8_0Backend::scalar);
      q4_dot4_matmul(w.data(), a.data(), got.data(), rows, tokens, blocks, stride);
      for (std::size_t i = 0; i < ref.size(); ++i)
        if (!std::isfinite(got[i]) || std::abs(got[i] - ref[i]) > 2e-5f * std::max(1.f, std::abs(ref[i]))) return 1;
      std::vector<Q8_0BlockX1> single(blocks);
      for (std::size_t b = 0; b < blocks; ++b) {
        single[b].scales[0] = a[b].scales[0]; single[b].sums[0] = a[b].sums[0];
        std::copy_n(a[b].qs, 32, single[b].qs);
      }
      q4_dot4_matvec(w.data(), single.data(), ref.data(), rows, blocks, Q8_0Backend::scalar);
      q4_dot4_matvec(w.data(), single.data(), got.data(), rows, blocks);
      for (std::size_t i = 0; i < rows; ++i)
        if (std::abs(got[i] - ref[i]) > 2e-5f * std::max(1.f, std::abs(ref[i]))) return 2;
    }
  std::cout << "WASM DOT4 dispatch, signed extremes, tails and strides passed\n";
}
