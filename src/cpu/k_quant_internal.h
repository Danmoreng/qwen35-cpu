#pragma once
#include "qwen35x/cpu/k_quant.h"
#include "q8_0_internal.h"
#include <cstring>
namespace qwen35x::cpu::detail {
inline float k_half(const std::uint8_t* p) noexcept {
  return half_to_float(static_cast<std::uint16_t>(p[0] | (p[1]<<8)));
}
inline void k_scale_min(const std::uint8_t* p, int g, int& scale, int& min) noexcept {
  const auto* s=p+4;
  if(g<4) {scale=s[g]&63; min=s[g+4]&63;}
  else {scale=(s[g+4]&15)|((s[g-4]>>6)<<4); min=(s[g+4]>>4)|((s[g]>>6)<<4);}
}
// One 32-element group; Q6_K has distinct scales for its two halves.
inline void k_group(const std::uint8_t* p, KQuantType type, int g,
                    std::int16_t* codes, float& d0, float& d1, float& bias) noexcept {
  bias=0;
  if(type==KQuantType::q8_0) {
    p+=g*34; d0=d1=k_half(p);
    for(int i=0;i<32;++i) codes[i]=static_cast<std::int8_t>(p[2+i]);
  } else if(type==KQuantType::q6_k) {
    const auto* lo=p+(g/4)*64+(g%2)*32;
    const auto* hi=p+128+(g/4)*32;
    d0=k_half(p+208)*static_cast<std::int8_t>(p[192+2*g]);
    d1=k_half(p+208)*static_cast<std::int8_t>(p[193+2*g]);
    for(int i=0;i<32;++i) codes[i]=((lo[i]>>(g%4>=2?4:0))&15) |
                                                (((hi[i]>>(2*(g%4)))&3)<<4);
    for(int i=0;i<32;++i) codes[i]-=32;
  } else {
    int sc,m; k_scale_min(p,g,sc,m);
    d0=d1=k_half(p)*sc; bias=-k_half(p+2)*m;
    const auto* lo=p+(type==KQuantType::q5_k?48:16)+(g/2)*32;
    for(int i=0;i<32;++i) codes[i]=((lo[i]>>(4*(g%2)))&15) |
      (type==KQuantType::q5_k?((p[16+i]>>g)&1)<<4:0);
  }
}
void k_quant_dot_rows_avx2(const std::uint8_t*, KQuantType,
  const KQuantActivation*, std::size_t, float*, std::size_t,
  std::size_t, std::size_t) noexcept;
}
