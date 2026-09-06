#include "k_quant_internal.h"
#include <immintrin.h>
namespace qwen35x::cpu::detail {
namespace {
float half(const std::uint8_t* p) noexcept {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(p[0] | (p[1]<<8))));
}
float sum8(__m256 a) noexcept {
  __m128 v=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
  v=_mm_hadd_ps(v,v);v=_mm_hadd_ps(v,v);return _mm_cvtss_f32(v);
}
__m256i shift_bytes(__m256i x,int shift) noexcept {
  return _mm256_srl_epi16(x,_mm_cvtsi32_si128(shift));
}
}
template<KQuantType type, std::size_t vectors>
void dot_rows(const std::uint8_t* w,
  const KQuantActivation* x,std::size_t stride,float* out,std::size_t out_stride,
  std::size_t blocks) noexcept {
  __m256 accum[4];float correction[4]{};
  for(std::size_t v=0;v<vectors;++v)accum[v]=_mm256_setzero_ps();
  const auto ones=_mm256_set1_epi16(1),mask=_mm256_set1_epi8(15);
  for(std::size_t b=0;b<blocks;++b,w+=k_quant_block_bytes(type)) {
    for(int g=0;g<8;++g) {
      __m256i codes;float d0,d1,bias=0;
      if constexpr(type==KQuantType::q8_0) {
        codes=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+g*34+2));
        d0=d1=half(w+g*34);
      } else if constexpr(type==KQuantType::q6_k) {
        auto lo=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+(g/4)*64+(g%2)*32));
        auto hi=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+128+(g/4)*32));
        lo=_mm256_and_si256(shift_bytes(lo,g%4>=2?4:0),mask);
        hi=_mm256_and_si256(shift_bytes(hi,2*(g%4)),_mm256_set1_epi8(3));
        codes=_mm256_or_si256(lo,_mm256_slli_epi16(hi,4));
        // Signed codes in [-32,31] avoid a separate zero-point correction.
        codes=_mm256_sub_epi8(codes,_mm256_set1_epi8(32));
        d0=half(w+208)*static_cast<std::int8_t>(w[192+g*2]);
        d1=half(w+208)*static_cast<std::int8_t>(w[193+g*2]);
      } else {
        int sc,m;k_scale_min(w,g,sc,m);
        d0=d1=half(w)*sc;bias=-half(w+2)*m;
        const auto* lo=w+(type==KQuantType::q5_k?48:16)+(g/2)*32;
        codes=_mm256_and_si256(shift_bytes(_mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(lo)),4*(g%2)),mask);
        if constexpr(type==KQuantType::q5_k) {
          auto hi=shift_bytes(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+16)),g);
          hi=_mm256_and_si256(hi,_mm256_set1_epi8(1));
          codes=_mm256_or_si256(codes,_mm256_slli_epi16(hi,4));
        }
      }
      const auto scale=_mm256_setr_ps(d0,d0,d0,d0,d1,d1,d1,d1);
      constexpr bool signed_codes=type==KQuantType::q6_k || type==KQuantType::q8_0;
      const auto magnitude=signed_codes?_mm256_abs_epi8(codes):codes;
      for(std::size_t v=0;v<vectors;++v) {
        const auto& a=x[v*stride+b];
        auto q=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.qs+g*32));
        if constexpr(signed_codes)q=_mm256_sign_epi8(q,codes);
        auto dot=_mm256_madd_epi16(_mm256_maddubs_epi16(magnitude,q),ones);
        const auto ds=_mm256_mul_ps(scale,_mm256_set1_ps(a.d[g]));
        accum[v]=_mm256_fmadd_ps(ds,_mm256_cvtepi32_ps(dot),accum[v]);
        correction[v]+=a.d[g]*bias*(a.sums[2*g]+a.sums[2*g+1]);
      }
    }
  }
  for(std::size_t v=0;v<vectors;++v)out[v*out_stride]=sum8(accum[v])+correction[v];
}
template<KQuantType type>
void dispatch(const std::uint8_t* w,const KQuantActivation* x,std::size_t stride,
              float* out,std::size_t out_stride,std::size_t vectors,std::size_t blocks) noexcept {
  switch(vectors) {
    case 1: return dot_rows<type,1>(w,x,stride,out,out_stride,blocks);
    case 2: return dot_rows<type,2>(w,x,stride,out,out_stride,blocks);
    case 3: return dot_rows<type,3>(w,x,stride,out,out_stride,blocks);
    case 4: return dot_rows<type,4>(w,x,stride,out,out_stride,blocks);
  }
}
void k_quant_dot_rows_avx2(const std::uint8_t* w,KQuantType type,
  const KQuantActivation* x,std::size_t stride,float* out,std::size_t out_stride,
  std::size_t vectors,std::size_t blocks) noexcept {
  switch(type) {
    case KQuantType::q4_k:return dispatch<KQuantType::q4_k>(w,x,stride,out,out_stride,vectors,blocks);
    case KQuantType::q5_k:return dispatch<KQuantType::q5_k>(w,x,stride,out,out_stride,vectors,blocks);
    case KQuantType::q6_k:return dispatch<KQuantType::q6_k>(w,x,stride,out,out_stride,vectors,blocks);
    case KQuantType::q8_0:return dispatch<KQuantType::q8_0>(w,x,stride,out,out_stride,vectors,blocks);
  }
}
}

#include <algorithm>
#include <cmath>
namespace qwen35x::cpu::detail {
void k_quant_prepare_avx2(const float* x,KQuantActivation* out,std::size_t blocks) noexcept {
  const auto sign=_mm256_set1_ps(-0.0f);
  for(std::size_t b=0;b<blocks;++b,x+=256) {
    __m256 maximum=_mm256_setzero_ps();
    for(int i=0;i<256;i+=8)maximum=_mm256_max_ps(maximum,_mm256_andnot_ps(sign,_mm256_loadu_ps(x+i)));
    float lanes[8];_mm256_storeu_ps(lanes,maximum);
    float amax=0;for(float value:lanes)amax=std::max(amax,value);
    float selected=0;
    if(amax)for(int i=0;i<256;++i)if(std::abs(x[i])==amax) {selected=x[i];break;}
    const float inv=amax?-127.0f/selected:0;
    const float d=amax?1.0f/inv:0;
    auto& a=out[b];_mm256_storeu_ps(a.d,_mm256_set1_ps(d));
    for(int i=0;i<256;i+=8) {
      auto values=_mm256_mul_ps(_mm256_loadu_ps(x+i),_mm256_set1_ps(inv));
      values=_mm256_min_ps(_mm256_set1_ps(127),_mm256_max_ps(_mm256_set1_ps(-127),values));
      auto integers=_mm256_cvtps_epi32(values);
      auto shorts=_mm_packs_epi32(_mm256_castsi256_si128(integers),_mm256_extracti128_si256(integers,1));
      auto bytes=_mm_packs_epi16(shorts,_mm_setzero_si128());
      _mm_storel_epi64(reinterpret_cast<__m128i*>(a.qs+i),bytes);
    }
    for(int g=0;g<16;++g) {
      auto q=_mm_loadu_si128(reinterpret_cast<const __m128i*>(a.qs+g*16));
      auto sums=_mm_madd_epi16(_mm_cvtepi8_epi16(q),_mm_set1_epi16(1));
      auto high=_mm_madd_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(q,8)),_mm_set1_epi16(1));
      sums=_mm_add_epi32(sums,high);sums=_mm_hadd_epi32(sums,sums);sums=_mm_hadd_epi32(sums,sums);
      a.sums[g]=static_cast<std::int16_t>(_mm_cvtsi128_si32(sums));
    }
  }
}
}
