#include "qwen35x/cpu/q4_dot4.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace qwen35x::cpu;
namespace {
bool same_outputs(const float * actual, const float * expected, std::size_t count,
                  bool exact) {
  if (exact) return std::memcmp(actual, expected, count * sizeof(float)) == 0;
  // The old scalar kernel uses separate multiply/add, DOT4 explicitly uses
  // FMA to match SIMD. On machines without AVX2, compare that legacy reference
  // numerically; packing, embedding and argmax indices are still exact.
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
        std::fabs(actual[i] - expected[i]) >
          2.0e-5F * std::max({1.0F, std::fabs(actual[i]), std::fabs(expected[i])})) return false;
  }
  return true;
}
}
int main() {
  std::mt19937 random(3701);
  const auto reference_backend = q8_0_backend_available(Q8_0Backend::avx2)
    ? Q8_0Backend::avx2 : Q8_0Backend::scalar;
  const bool exact = reference_backend == Q8_0Backend::avx2;
  std::size_t cases = 0;
  for (std::size_t blocks : {1U, 2U, 4U, 32U, 64U, 112U}) {
    for (std::size_t rows : {8U, 16U, 32U, 64U}) {
      for (int pattern = 0; pattern < 3; ++pattern) {
        std::vector<Q4_0Block> canonical(rows*blocks), unpacked(rows*blocks);
        for (auto & w : canonical) {
          const std::uint16_t scales[] = {0, 0x3c00, 0xbc00, 0x2810};
          w.d = scales[random()%4];
          for (auto & q : w.qs) q = pattern == 0 ? 255 : pattern == 1 ? 0 : static_cast<std::uint8_t>(random());
        }
        std::vector<Q4_0BlockX8> legacy(rows/8*blocks), dot4(legacy.size());
        q4_0_pack_rows_8(canonical.data(), legacy.data(), rows, blocks);
        q4_dot4_pack_rows_8(canonical.data(), dot4.data(), rows, blocks);
        q4_dot4_unpack_rows_8(dot4.data(), unpacked.data(), rows, blocks);
        if (std::memcmp(canonical.data(), unpacked.data(), canonical.size()*sizeof(Q4_0Block))) return 1;
        std::vector<float> embedding(blocks*32), expected_embedding(embedding.size());
        q4_dot4_dequantize_row(dot4.data(), rows-1, embedding.data(), blocks);
        q4_0_packed_dequantize_row(legacy.data(), rows-1, expected_embedding.data(), blocks);
        if (embedding != expected_embedding) return 2;

        constexpr std::size_t tokens = 20; // 16-token tile plus 4-token tail.
        std::vector<Q8_0BlockX4> batch((tokens/4)*blocks);
        for (auto & a : batch) for (int t = 0; t < 4; ++t) {
          a.scales[t] = 0.001F * static_cast<float>(1 + random()%31);
          a.sums[t] = 0;
          for (int k = 0; k < 32; ++k) {
            const auto q = static_cast<std::int8_t>(pattern == 0 ? -128 : pattern == 1 ? 127 : static_cast<int>(random()%256)-128);
            a.qs[t*32+k] = q;
            a.sums[t] += q;
          }
        }
        std::vector<Q8_0BlockX1> vector(blocks);
        for (std::size_t b = 0; b < blocks; ++b) {
          vector[b].scales[0] = batch[b].scales[0];
          vector[b].sums[0] = batch[b].sums[0];
          std::copy_n(batch[b].qs, 32, vector[b].qs);
        }
        std::vector<float> expected(rows*tokens), actual(expected.size());
        q4_0_packed_matmul_q8_0(legacy.data(), batch.data(), expected.data(), rows, tokens, blocks, rows, reference_backend);
        std::vector<int> counts(rows+7, 0);
        for (std::size_t i = 0; i < counts.size(); i += 3) counts[i] = 2;
        const auto expected_best = q4_0_packed_matvec_prepared_q8_0_argmax(
          legacy.data(), vector.data(), counts.data(), 1.05F, 7, rows, blocks, reference_backend);
        for (auto backend : {Q8_0Backend::scalar, Q8_0Backend::avx2, Q8_0Backend::avx_vnni, Q8_0Backend::avx512_vnni}) {
          if (!q8_0_backend_available(backend)) continue;
          q4_dot4_matmul(dot4.data(), batch.data(), actual.data(), rows, tokens, blocks, rows, backend);
          if (!same_outputs(actual.data(), expected.data(), actual.size(), exact)) { std::cerr << "DOT4 prefill mismatch\n"; return 3; }
          q4_dot4_matvec(dot4.data(), vector.data(), actual.data(), rows, blocks, backend);
          if (!same_outputs(actual.data(), expected.data(), rows, exact)) return 4;
          const auto best = q4_dot4_argmax(dot4.data(), vector.data(), counts.data(), 1.05F, 7, rows, blocks, backend);
          if (!same_outputs(&best.value, &expected_best.value, 1, exact) || best.index != expected_best.index) return 5;
          ++cases;
        }
      }
    }
  }
  std::cout << "DOT4 roundtrip, embedding, decode, prefill/tail and argmax: " << cases << " cases passed\n";
}
