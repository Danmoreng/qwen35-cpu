#include "q4_dot4_internal.h"
#include "f16c_compat.h"
#include <immintrin.h>
#include <cstring>
#include <limits>
#include <utility>
#if defined(_MSC_VER)
#define DOT4_INLINE __forceinline
#else
#define DOT4_INLINE inline __attribute__((always_inline))
#endif
namespace qwen35x::cpu::detail {
namespace {
struct Dot4Weights { __m256i lo[4], hi[4]; __m256 scales; };
DOT4_INLINE Dot4Weights load_weights(const Q4_0BlockX8 & w) noexcept {
  Dot4Weights result;
  const __m256i mask = _mm256_set1_epi8(15);
  const __m256i raw0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w.qs + 0));
  result.lo[0] = _mm256_and_si256(raw0, mask);
  result.hi[0] = _mm256_and_si256(_mm256_srli_epi16(raw0, 4), mask);
  const __m256i raw1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w.qs + 32));
  result.lo[1] = _mm256_and_si256(raw1, mask);
  result.hi[1] = _mm256_and_si256(_mm256_srli_epi16(raw1, 4), mask);
  const __m256i raw2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w.qs + 64));
  result.lo[2] = _mm256_and_si256(raw2, mask);
  result.hi[2] = _mm256_and_si256(_mm256_srli_epi16(raw2, 4), mask);
  const __m256i raw3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w.qs + 96));
  result.lo[3] = _mm256_and_si256(raw3, mask);
  result.hi[3] = _mm256_and_si256(_mm256_srli_epi16(raw3, 4), mask);
  result.scales = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w.d)));
  return result;
}
DOT4_INLINE __m256i broadcast4(const std::int8_t * values) noexcept {
  int bytes;
  std::memcpy(&bytes, values, sizeof(bytes));
  return _mm256_set1_epi32(bytes);
}
template <std::size_t Token, typename Block>
DOT4_INLINE void accumulate(const Dot4Weights & w, const Block * vectors, std::size_t blocks,
                           std::size_t block, __m256 & accumulator) noexcept {
  const auto & a = vectors[(Token / 4) * blocks + block];
  const auto * qs = a.qs + (Token % 4) * 32;
#if DOT4_AVX2
  // Each int16 lane sums 16 products, bounded by 16*15*128=30720.
  __m256i pairs = _mm256_setzero_si256();
#else
  __m256i low = _mm256_setzero_si256(), high = _mm256_setzero_si256();
#endif
#if DOT4_AVX2
  pairs = _mm256_add_epi16(pairs, _mm256_maddubs_epi16(w.lo[0], broadcast4(qs + 0)));
  pairs = _mm256_add_epi16(pairs, _mm256_maddubs_epi16(w.hi[0], broadcast4(qs + 4)));
#else
  low = DOT4_DPBUSD(low, w.lo[0], broadcast4(qs + 0));
  high = DOT4_DPBUSD(high, w.hi[0], broadcast4(qs + 4));
#endif
#if DOT4_AVX2
  pairs = _mm256_add_epi16(pairs, _mm256_maddubs_epi16(w.lo[1], broadcast4(qs + 8)));
  pairs = _mm256_add_epi16(pairs, _mm256_maddubs_epi16(w.hi[1], broadcast4(qs + 12)));
#else
  low = DOT4_DPBUSD(low, w.lo[1], broadcast4(qs + 8));
  high = DOT4_DPBUSD(high, w.hi[1], broadcast4(qs + 12));
#endif
#if DOT4_AVX2
  pairs = _mm256_add_epi16(pairs, _mm256_maddubs_epi16(w.lo[2], broadcast4(qs + 16)));
  pairs = _mm256_add_epi16(pairs, _mm256_maddubs_epi16(w.hi[2], broadcast4(qs + 20)));
#else
  low = DOT4_DPBUSD(low, w.lo[2], broadcast4(qs + 16));
  high = DOT4_DPBUSD(high, w.hi[2], broadcast4(qs + 20));
#endif
#if DOT4_AVX2
  pairs = _mm256_add_epi16(pairs, _mm256_maddubs_epi16(w.lo[3], broadcast4(qs + 24)));
  pairs = _mm256_add_epi16(pairs, _mm256_maddubs_epi16(w.hi[3], broadcast4(qs + 28)));
#else
  low = DOT4_DPBUSD(low, w.lo[3], broadcast4(qs + 24));
  high = DOT4_DPBUSD(high, w.hi[3], broadcast4(qs + 28));
#endif
#if DOT4_AVX2
  __m256i dot = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
#else
  __m256i dot = _mm256_add_epi32(low, high);
#endif
  dot = _mm256_sub_epi32(dot, _mm256_set1_epi32(8 * static_cast<int>(a.sums[Token % 4])));
  const __m256 scale = _mm256_mul_ps(w.scales, _mm256_set1_ps(a.scales[Token % 4]));
  accumulator = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), scale, accumulator);
}
template <typename Block, std::size_t... Token>
DOT4_INLINE void tile(const Q4_0BlockX8 * matrix, const Block * vectors, float * output,
                     std::size_t blocks, std::size_t stride, std::index_sequence<Token...>) noexcept {
  __m256 accumulators[sizeof...(Token)] = {((void)Token, _mm256_setzero_ps())...};
  for (std::size_t block = 0; block < blocks; ++block) {
    const auto weights = load_weights(matrix[block]);
    (accumulate<Token>(weights, vectors, blocks, block, accumulators[Token]), ...);
  }
  (_mm256_storeu_ps(output + Token * stride, accumulators[Token]), ...);
}
}
void DOT4_FN(matvec)(const Q4_0BlockX8 * matrix, const Q8_0BlockX1 * vector, float * output,
                     std::size_t rows, std::size_t blocks) noexcept {
  for (std::size_t r = 0; r < rows / 8; ++r)
    tile(matrix + r * blocks, vector, output + r * 8, blocks, 8, std::make_index_sequence<1>{});
}
void DOT4_FN(matmul)(const Q4_0BlockX8 * matrix, const Q8_0BlockX4 * vectors, float * output,
                     std::size_t rows, std::size_t count, std::size_t blocks, std::size_t stride) noexcept {
  std::size_t token = 0;
#if DOT4_WIDE
  for (; token + 16 <= count; token += 16) {
    for (std::size_t r = 0; r < rows / 8; ++r)
      tile(matrix + r * blocks, vectors + (token / 4) * blocks, output + token * stride + r * 8,
           blocks, stride, std::make_index_sequence<16>{});
  }
#endif
  for (; token + 4 <= count; token += 4) {
    for (std::size_t r = 0; r < rows / 8; ++r)
      tile(matrix + r * blocks, vectors + (token / 4) * blocks, output + token * stride + r * 8,
           blocks, stride, std::make_index_sequence<4>{});
  }
}
Q4_0ArgmaxResult DOT4_FN(argmax)(const Q4_0BlockX8 * matrix, const Q8_0BlockX1 * vector,
    const int * counts, float penalty, std::size_t offset, std::size_t rows, std::size_t blocks) noexcept {
  Q4_0ArgmaxResult best{-std::numeric_limits<float>::infinity(), offset};
  alignas(32) float values[8];
  for (std::size_t r = 0; r < rows / 8; ++r) {
    tile(matrix + r * blocks, vector, values, blocks, 8, std::make_index_sequence<1>{});
    for (std::size_t lane = 0; lane < 8; ++lane) {
      const std::size_t index = offset + r * 8 + lane;
      float value = values[lane];
      if (counts != nullptr && counts[index] > 0 && penalty > 1.0F)
        value = value > 0.0F ? value / penalty : value * penalty;
      if (value > best.value) best = {value, index};
    }
  }
  return best;
}
}
#undef DOT4_INLINE
