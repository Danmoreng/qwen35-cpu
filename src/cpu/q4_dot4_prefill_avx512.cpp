#include "q4_dot4_internal.h"
#include "f16c_compat.h"

#include <immintrin.h>
#include <cstring>
#include <utility>

namespace qwen35x::cpu::detail {
namespace {
#if defined(_MSC_VER)
#define PREFILL_INLINE __forceinline
#else
#define PREFILL_INLINE inline __attribute__((always_inline))
#endif

PREFILL_INLINE __m512i join(__m256i low, __m256i high) noexcept {
  return _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
}

struct Weights16 {
  __m512i low[4], high[4];
  __m512 scales;
};

PREFILL_INLINE Weights16 load(const Q4_0BlockX8 & a,
                              const Q4_0BlockX8 & b) noexcept {
  Weights16 w;
  const auto mask = _mm512_set1_epi32(0x0f0f0f0f);
  for (int part = 0; part < 4; ++part) {
    const auto bits = join(
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a.qs + 32 * part)),
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b.qs + 32 * part)));
    w.low[part] = _mm512_and_si512(bits, mask);
    w.high[part] = _mm512_and_si512(_mm512_srli_epi32(bits, 4), mask);
  }
  w.scales = _mm512_castsi512_ps(join(
    _mm256_castps_si256(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a.d)))),
    _mm256_castps_si256(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b.d))))));
  return w;
}

PREFILL_INLINE __m512i broadcast4(const std::int8_t * values) noexcept {
  int bytes;
  std::memcpy(&bytes, values, sizeof(bytes));
  return _mm512_set1_epi32(bytes);
}

template <std::size_t Token>
PREFILL_INLINE void accumulate(const Weights16 & w, const Q8_0BlockX4 * vectors,
                               std::size_t blocks, std::size_t block,
                               __m512 & accumulator) noexcept {
  const auto & a = vectors[(Token / 4) * blocks + block];
  const auto * qs = a.qs + (Token % 4) * 32;
  auto low = _mm512_setzero_si512(), high = _mm512_setzero_si512();
  for (int part = 0; part < 4; ++part) {
    low = _mm512_dpbusd_epi32(low, w.low[part], broadcast4(qs + 8 * part));
    high = _mm512_dpbusd_epi32(high, w.high[part], broadcast4(qs + 8 * part + 4));
  }
  auto dot = _mm512_sub_epi32(_mm512_add_epi32(low, high),
    _mm512_set1_epi32(8 * static_cast<int>(a.sums[Token % 4])));
  const auto scale = _mm512_mul_ps(w.scales, _mm512_set1_ps(a.scales[Token % 4]));
  // Preserve the original per-block integer result and floating-point FMA order.
  accumulator = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), scale, accumulator);
}

template <std::size_t... Token>
PREFILL_INLINE void tile(const Q4_0BlockX8 * matrix, const Q8_0BlockX4 * vectors,
                         float * output, std::size_t blocks, std::size_t stride,
                         std::index_sequence<Token...>) noexcept {
  __m512 accumulators[sizeof...(Token)] = {((void)Token, _mm512_setzero_ps())...};
  for (std::size_t block = 0; block < blocks; ++block) {
    const auto w = load(matrix[block], matrix[blocks + block]);
    (accumulate<Token>(w, vectors, blocks, block, accumulators[Token]), ...);
  }
  (_mm512_storeu_ps(output + Token * stride, accumulators[Token]), ...);
}
#undef PREFILL_INLINE
} // namespace

void q4_dot4_matmul_avx512(const Q4_0BlockX8 * matrix, const Q8_0BlockX4 * vectors,
                          float * output, std::size_t rows, std::size_t count,
                          std::size_t blocks, std::size_t stride) noexcept {
  // Pair two existing eight-row blocks in ZMM lanes without changing storage.
  // Eight tokens reuse the unpacked weights while limiting register pressure.
  std::size_t token = 0;
  for (; token + 8 <= count; token += 8) {
    std::size_t row = 0;
    for (; row + 16 <= rows; row += 16)
      tile(matrix + (row / 8) * blocks, vectors + (token / 4) * blocks,
           output + token * stride + row, blocks, stride, std::make_index_sequence<8>{});
    if (row < rows)
      q4_dot4_matmul_evex(matrix + (row / 8) * blocks, vectors + (token / 4) * blocks,
                         output + token * stride + row, rows - row, 8, blocks, stride);
  }
  if (token < count)
    q4_dot4_matmul_evex(matrix, vectors + (token / 4) * blocks,
                       output + token * stride, rows, count - token, blocks, stride);
}
} // namespace qwen35x::cpu::detail
