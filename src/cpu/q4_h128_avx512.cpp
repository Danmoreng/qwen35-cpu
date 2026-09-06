#include "q4_h128_internal.h"
#include <immintrin.h>

namespace qwen35x::cpu::detail {
namespace {
inline __m512 signed_h16(const float * input, const __mmask16 signs) noexcept {
  const __m512i bits = _mm512_castps_si512(_mm512_loadu_ps(input));
  __m512 v = _mm512_castsi512_ps(_mm512_mask_xor_epi32(
    bits, signs, bits, _mm512_set1_epi32(static_cast<int>(0x80000000U))));
  __m512 shuffled = _mm512_permute_ps(v, 0xb1);
  __m512 sums = _mm512_add_ps(v, shuffled);
  __m512 differences = _mm512_sub_ps(v, shuffled);
  v = _mm512_mask_blend_ps(0xaaaa, sums, _mm512_permute_ps(differences, 0xb1));
  shuffled = _mm512_permute_ps(v, 0x4e);
  sums = _mm512_add_ps(v, shuffled);
  differences = _mm512_sub_ps(v, shuffled);
  v = _mm512_mask_blend_ps(0xcccc, sums, _mm512_permute_ps(differences, 0x4e));
  shuffled = _mm512_shuffle_f32x4(v, v, 0xb1);
  sums = _mm512_add_ps(v, shuffled);
  differences = _mm512_sub_ps(v, shuffled);
  v = _mm512_mask_blend_ps(0xf0f0, sums, _mm512_shuffle_f32x4(differences, differences, 0xb1));
  shuffled = _mm512_shuffle_f32x4(v, v, 0x4e);
  sums = _mm512_add_ps(v, shuffled);
  differences = _mm512_sub_ps(v, shuffled);
  return _mm512_mask_blend_ps(0xff00, sums, _mm512_shuffle_f32x4(differences, differences, 0x4e));
}
}

// Eight live ZMM vectors keep the outer three butterfly stages in registers.
// Each element follows the same add/subtract order as the scalar/AVX2 kernel.
void q4_h128_transform_block_avx512_signed(
  const float * input, float * output, const std::uint64_t * signs) noexcept {
  __m512 v0 = signed_h16(input + 0, static_cast<__mmask16>(signs[0] >> 0));
  __m512 v1 = signed_h16(input + 16, static_cast<__mmask16>(signs[0] >> 16));
  __m512 v2 = signed_h16(input + 32, static_cast<__mmask16>(signs[0] >> 32));
  __m512 v3 = signed_h16(input + 48, static_cast<__mmask16>(signs[0] >> 48));
  __m512 v4 = signed_h16(input + 64, static_cast<__mmask16>(signs[1] >> 0));
  __m512 v5 = signed_h16(input + 80, static_cast<__mmask16>(signs[1] >> 16));
  __m512 v6 = signed_h16(input + 96, static_cast<__mmask16>(signs[1] >> 32));
  __m512 v7 = signed_h16(input + 112, static_cast<__mmask16>(signs[1] >> 48));
  { const __m512 a = v0; v0 = _mm512_add_ps(a, v1); v1 = _mm512_sub_ps(a, v1); }
  { const __m512 a = v2; v2 = _mm512_add_ps(a, v3); v3 = _mm512_sub_ps(a, v3); }
  { const __m512 a = v4; v4 = _mm512_add_ps(a, v5); v5 = _mm512_sub_ps(a, v5); }
  { const __m512 a = v6; v6 = _mm512_add_ps(a, v7); v7 = _mm512_sub_ps(a, v7); }
  { const __m512 a = v0; v0 = _mm512_add_ps(a, v2); v2 = _mm512_sub_ps(a, v2); }
  { const __m512 a = v1; v1 = _mm512_add_ps(a, v3); v3 = _mm512_sub_ps(a, v3); }
  { const __m512 a = v4; v4 = _mm512_add_ps(a, v6); v6 = _mm512_sub_ps(a, v6); }
  { const __m512 a = v5; v5 = _mm512_add_ps(a, v7); v7 = _mm512_sub_ps(a, v7); }
  { const __m512 a = v0; v0 = _mm512_add_ps(a, v4); v4 = _mm512_sub_ps(a, v4); }
  { const __m512 a = v1; v1 = _mm512_add_ps(a, v5); v5 = _mm512_sub_ps(a, v5); }
  { const __m512 a = v2; v2 = _mm512_add_ps(a, v6); v6 = _mm512_sub_ps(a, v6); }
  { const __m512 a = v3; v3 = _mm512_add_ps(a, v7); v7 = _mm512_sub_ps(a, v7); }
  _mm512_storeu_ps(output + 0, v0);
  _mm512_storeu_ps(output + 16, v1);
  _mm512_storeu_ps(output + 32, v2);
  _mm512_storeu_ps(output + 48, v3);
  _mm512_storeu_ps(output + 64, v4);
  _mm512_storeu_ps(output + 80, v5);
  _mm512_storeu_ps(output + 96, v6);
  _mm512_storeu_ps(output + 112, v7);
}
} // namespace qwen35x::cpu::detail
