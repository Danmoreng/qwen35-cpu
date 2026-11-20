#include "q4_dot4_internal.h"
#include "f16c_compat.h"

#include <immintrin.h>
#include <cstring>
#include <limits>
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

struct Nibbles16 {
  __m512i low, high;
};

PREFILL_INLINE Nibbles16 unpack_part(const Q4_0BlockX8 &a, const Q4_0BlockX8 &b,
                                    unsigned part) noexcept {
  const auto bits = join(
    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a.qs + 32 * part)),
    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b.qs + 32 * part)));
  const auto mask = _mm512_set1_epi8(15);
  return {_mm512_and_si512(bits, mask), _mm512_and_si512(_mm512_srli_epi32(bits, 4), mask)};
}

struct Weights16 {
  __m512i low[4], high[4];
  __m512 scales;
};

PREFILL_INLINE Weights16 load(const Q4_0BlockX8 & a,
                              const Q4_0BlockX8 & b) noexcept {
  Weights16 w;
  for (unsigned part = 0; part < 4; ++part) {
    const auto nibbles = unpack_part(a, b, part);
    w.low[part] = nibbles.low;
    w.high[part] = nibbles.high;
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

template <std::size_t Token, typename Block>
PREFILL_INLINE void accumulate(const Weights16 & w, const Block * vectors,
                               std::size_t blocks, std::size_t block,
                               __m512 & accumulator) noexcept {
  constexpr auto lanes = sizeof(Block::sums) / sizeof(std::int16_t);
  const auto & a = vectors[(Token / lanes) * blocks + block];
  const auto * qs = a.qs + (Token % lanes) * 32;
  auto low = _mm512_setzero_si512(), high = _mm512_setzero_si512();
  for (int part = 0; part < 4; ++part) {
    low = _mm512_dpbusd_epi32(low, w.low[part], broadcast4(qs + 8 * part));
    high = _mm512_dpbusd_epi32(high, w.high[part], broadcast4(qs + 8 * part + 4));
  }
  auto dot = _mm512_sub_epi32(_mm512_add_epi32(low, high),
    _mm512_set1_epi32(8 * static_cast<int>(a.sums[Token % lanes])));
  const auto scale = _mm512_mul_ps(w.scales, _mm512_set1_ps(a.scales[Token % lanes]));
  // Preserve the original per-block integer result and floating-point FMA order.
  accumulator = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), scale, accumulator);
}

template <typename Block, std::size_t... Token>
PREFILL_INLINE void tile(const Q4_0BlockX8 * matrix, const Block * vectors,
                         float * output, std::size_t blocks, std::size_t stride,
                         std::index_sequence<Token...>) noexcept {
  __m512 accumulators[sizeof...(Token)] = {((void)Token, _mm512_setzero_ps())...};
  for (std::size_t block = 0; block < blocks; ++block) {
    const auto w = load(matrix[block], matrix[blocks + block]);
    (accumulate<Token>(w, vectors, blocks, block, accumulators[Token]), ...);
  }
  (_mm512_storeu_ps(output + Token * stride, accumulators[Token]), ...);
}

// Eight prompt tokens share each weight load. Broadcasting the same four
// activation bytes to both row pairs covers 32 output rows per broadcast.
PREFILL_INLINE void tile_32x8(const Q4_0BlockX8 *matrix, const Q8_0BlockX4 *vectors,
                             float *output, std::size_t blocks, std::size_t stride) noexcept {
  constexpr unsigned pairs = 2;
  constexpr unsigned tokens = 8;
  __m512 accumulators[pairs][tokens]{};
  for (std::size_t block = 0; block < blocks; ++block) {
    __m512i dots[pairs][tokens];
    for (unsigned t = 0; t < tokens; ++t) {
      const auto &activation = vectors[(t / 4) * blocks + block];
      // Subtract the Q4 zero point once per block before adding unsigned-nibble dots.
      const auto correction = _mm512_set1_epi32(-8 * static_cast<int>(activation.sums[t % 4]));
      for (unsigned pair = 0; pair < pairs; ++pair) dots[pair][t] = correction;
    }
    for (unsigned part = 0; part < 4; ++part) {
      Nibbles16 weights[pairs];
      for (unsigned pair = 0; pair < pairs; ++pair) {
        const auto *rows = matrix + (2 * pair) * blocks + block;
        weights[pair] = unpack_part(rows[0], rows[blocks], part);
      }
      for (unsigned t = 0; t < tokens; ++t) {
        const auto &activation = vectors[(t / 4) * blocks + block];
        const auto *bytes = activation.qs + (t % 4) * 32 + 8 * part;
        const auto low = broadcast4(bytes);
        const auto high = broadcast4(bytes + 4);
        for (unsigned pair = 0; pair < pairs; ++pair) {
          dots[pair][t] = _mm512_dpbusd_epi32(dots[pair][t], weights[pair].low, low);
          dots[pair][t] = _mm512_dpbusd_epi32(dots[pair][t], weights[pair].high, high);
        }
      }
    }
    for (unsigned pair = 0; pair < pairs; ++pair) {
      const auto *rows = matrix + (2 * pair) * blocks + block;
      const auto low = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rows[0].d));
      const auto high = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rows[blocks].d));
      const auto weight_scales = _mm512_cvtph_ps(
        _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1));
      for (unsigned t = 0; t < tokens; ++t) {
        const auto &activation = vectors[(t / 4) * blocks + block];
        const auto scale = _mm512_mul_ps(weight_scales, _mm512_set1_ps(activation.scales[t % 4]));
        accumulators[pair][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dots[pair][t]),
                                                scale, accumulators[pair][t]);
      }
    }
  }
  for (unsigned pair = 0; pair < pairs; ++pair)
    for (unsigned t = 0; t < tokens; ++t)
      _mm512_storeu_ps(output + t * stride + pair * 16, accumulators[pair][t]);
}
#undef PREFILL_INLINE
} // namespace

void q4_dot4_matvec_avx512(const Q4_0BlockX8 * matrix, const Q8_0BlockX1 * vector,
                          float * output, std::size_t rows, std::size_t blocks) noexcept {
  std::size_t row = 0;
  for (; row + 16 <= rows; row += 16)
    tile(matrix + (row / 8) * blocks, vector, output + row,
         blocks, 16, std::make_index_sequence<1>{});
  if (row < rows)
    q4_dot4_matvec_evex(matrix + (row / 8) * blocks, vector, output + row,
                       rows - row, blocks);
}

Q4_0ArgmaxResult q4_dot4_argmax_avx512(const Q4_0BlockX8 * matrix,
    const Q8_0BlockX1 * vector, const int * counts, float penalty,
    std::size_t offset, std::size_t rows, std::size_t blocks) noexcept {
  Q4_0ArgmaxResult best{-std::numeric_limits<float>::infinity(), offset};
  alignas(64) float values[16];
  std::size_t row = 0;
  for (; row + 16 <= rows; row += 16) {
    tile(matrix + (row / 8) * blocks, vector, values,
         blocks, 16, std::make_index_sequence<1>{});
    for (std::size_t lane = 0; lane < 16; ++lane) {
      const std::size_t index = offset + row + lane;
      float value = values[lane];
      if (counts != nullptr && counts[index] > 0 && penalty > 1.0F)
        value = value > 0.0F ? value / penalty : value * penalty;
      if (value > best.value) best = {value, index};
    }
  }
  if (row < rows) {
    const auto tail = q4_dot4_argmax_evex(matrix + (row / 8) * blocks, vector,
      counts, penalty, offset + row, rows - row, blocks);
    if (tail.value > best.value) best = tail;
  }
  return best;
}

void q4_dot4_matmul_x1_avx512(const Q4_0BlockX8 *matrix, const Q8_0BlockX1 *vectors,
                             float *output, std::size_t rows, std::size_t count,
                             std::size_t blocks, std::size_t stride) noexcept {
  auto project = [&](auto tokens) {
    std::size_t row = 0;
    for (; row + 16 <= rows; row += 16)
      tile(matrix + (row / 8) * blocks, vectors, output + row, blocks, stride, tokens);
    if (row < rows)
      for (std::size_t i = 0; i < count; ++i)
        q4_dot4_matvec_evex(matrix + (row / 8) * blocks, vectors + i * blocks,
                            output + i * stride + row, rows - row, blocks);
  };
  switch (count) {
    case 2: project(std::make_index_sequence<2>{}); break;
    case 3: project(std::make_index_sequence<3>{}); break;
    case 5: project(std::make_index_sequence<5>{}); break;
  }
}

void q4_dot4_matmul_avx512(const Q4_0BlockX8 * matrix, const Q8_0BlockX4 * vectors,
                          float * output, std::size_t rows, std::size_t count,
                          std::size_t blocks, std::size_t stride) noexcept {
  std::size_t row = 0;
  if (count >= 64 && count % 8 == 0) {
    for (; row + 32 <= rows; row += 32) {
      for (std::size_t token = 0; token < count; token += 8)
        tile_32x8(matrix + (row / 8) * blocks, vectors + (token / 4) * blocks,
                  output + token * stride + row, blocks, stride);
    }
  }
  // Remaining row pairs and smaller batches use the existing token tiles.
  for (; row + 16 <= rows; row += 16) {
    std::size_t token = 0;
    if (count >= 64) {
      for (; token + 16 <= count; token += 16)
        tile(matrix + (row / 8) * blocks, vectors + (token / 4) * blocks,
             output + token * stride + row, blocks, stride, std::make_index_sequence<16>{});
    }
    for (; token + 8 <= count; token += 8)
      tile(matrix + (row / 8) * blocks, vectors + (token / 4) * blocks,
           output + token * stride + row, blocks, stride, std::make_index_sequence<8>{});
    for (; token + 4 <= count; token += 4)
      tile(matrix + (row / 8) * blocks, vectors + (token / 4) * blocks,
           output + token * stride + row, blocks, stride, std::make_index_sequence<4>{});
    if (token < count)
      q4_dot4_matmul_evex(matrix + (row / 8) * blocks, vectors + (token / 4) * blocks,
                         output + token * stride + row, 16, count - token, blocks, stride);
  }
  if (row < rows)
    q4_dot4_matmul_evex(matrix + (row / 8) * blocks, vectors,
                       output + row, rows - row, count, blocks, stride);
}
} // namespace qwen35x::cpu::detail
