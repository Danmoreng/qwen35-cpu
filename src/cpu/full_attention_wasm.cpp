// SPDX-License-Identifier: MIT
// SIMD lanes span independent query heads. FP32 QK/PV reduction order is fixed.
#include "qwen35x/cpu/full_attention.h"
#include <wasm_simd128.h>
#include <chrono>
#include <cmath>
#include <limits>

namespace qwen35x::cpu::detail {
namespace {
using Clock=std::chrono::steady_clock;
float sigmoid(float x) noexcept {
  if(x>=0.F)return 1.F/(1.F+std::exp(-x));
  const float e=std::exp(x);return e/(1.F+e);
}
void transpose_queries(const float *q, float *packed) noexcept {
  for(std::size_t c=0;c<256;c+=4) {
    const auto r0=wasm_v128_load(q+c),r1=wasm_v128_load(q+256+c);
    const auto r2=wasm_v128_load(q+512+c),r3=wasm_v128_load(q+768+c);
    const auto a=wasm_i32x4_shuffle(r0,r1,0,4,1,5);
    const auto b=wasm_i32x4_shuffle(r0,r1,2,6,3,7);
    const auto d=wasm_i32x4_shuffle(r2,r3,0,4,1,5);
    const auto e=wasm_i32x4_shuffle(r2,r3,2,6,3,7);
    wasm_v128_store(packed+4*c,wasm_i32x4_shuffle(a,d,0,1,4,5));
    wasm_v128_store(packed+4*c+4,wasm_i32x4_shuffle(a,d,2,3,6,7));
    wasm_v128_store(packed+4*c+8,wasm_i32x4_shuffle(b,e,0,1,4,5));
    wasm_v128_store(packed+4*c+12,wasm_i32x4_shuffle(b,e,2,3,6,7));
  }
}
}
void causal_attention_gqa4_rows_wasm(const Gqa4Attention &a,
    std::size_t begin,std::size_t end,Gqa4AttentionTimes *times) noexcept {
  const std::size_t qw=a.kv_heads*4*256,kw=a.kv_heads*256;
  alignas(16) float packed[256*4];
  for(std::size_t group=begin;group<end;++group) {
    auto mark=times?Clock::now():Clock::time_point{};
    const auto token=group/a.kv_heads,kv=group%a.kv_heads;
    const auto qoffset=token*qw+kv*4*256;
    const auto length=a.position+token+1;
    transpose_queries(a.queries+qoffset,packed);
    if(times){auto now=Clock::now();times->pack_ms+=std::chrono::duration<double,std::milli>(now-mark).count();mark=now;}
    auto maximum=wasm_f32x4_splat(-std::numeric_limits<float>::infinity());
    for(std::size_t context=0;context<length;++context) {
      const float *k=a.keys+context*kw+kv*256;
      auto dot=wasm_f32x4_splat(0);
      for(std::size_t c=0;c<256;++c)
        dot=wasm_f32x4_add(dot,wasm_f32x4_mul(wasm_v128_load(packed+4*c),wasm_f32x4_splat(k[c])));
      dot=wasm_f32x4_mul(dot,wasm_f32x4_splat(a.scale));
      wasm_v128_store(a.scores+4*context,dot);
      // Match std::max's left operand on equal values, including signed zero.
      maximum=wasm_v128_bitselect(dot,maximum,wasm_f32x4_gt(dot,maximum));
    }
    if(times){auto now=Clock::now();times->qk_ms+=std::chrono::duration<double,std::milli>(now-mark).count();mark=now;}
    auto denominator=wasm_f32x4_splat(0);
    alignas(16) float lanes[4];
    for(std::size_t context=0;context<length;++context) {
      wasm_v128_store(lanes,wasm_f32x4_sub(wasm_v128_load(a.scores+4*context),maximum));
      for(auto &x:lanes)x=std::exp(x);
      const auto probabilities=wasm_v128_load(lanes);
      wasm_v128_store(a.scores+4*context,probabilities);
      denominator=wasm_f32x4_add(denominator,probabilities);
    }
    const auto one=wasm_f32x4_splat(1);
    denominator=wasm_v128_bitselect(denominator,one,wasm_f32x4_gt(denominator,wasm_f32x4_splat(0)));
    const auto inverse=wasm_f32x4_div(one,denominator);
    if(times){auto now=Clock::now();times->softmax_ms+=std::chrono::duration<double,std::milli>(now-mark).count();mark=now;}
    float *o0=a.output+qoffset,*o1=o0+256,*o2=o1+256,*o3=o2+256;
    for(std::size_t c=0;c<1024;c+=4)wasm_v128_store(o0+c,wasm_f32x4_splat(0));
    for(std::size_t context=0;context<length;++context) {
      const auto p=wasm_f32x4_mul(wasm_v128_load(a.scores+4*context),inverse);
      const auto p0=wasm_i32x4_shuffle(p,p,0,0,0,0),p1=wasm_i32x4_shuffle(p,p,1,1,1,1);
      const auto p2=wasm_i32x4_shuffle(p,p,2,2,2,2),p3=wasm_i32x4_shuffle(p,p,3,3,3,3);
      const float *v=a.values+context*kw+kv*256;
      for(std::size_t c=0;c<256;c+=4) {
        const auto value=wasm_v128_load(v+c);
        wasm_v128_store(o0+c,wasm_f32x4_add(wasm_v128_load(o0+c),wasm_f32x4_mul(p0,value)));
        wasm_v128_store(o1+c,wasm_f32x4_add(wasm_v128_load(o1+c),wasm_f32x4_mul(p1,value)));
        wasm_v128_store(o2+c,wasm_f32x4_add(wasm_v128_load(o2+c),wasm_f32x4_mul(p2,value)));
        wasm_v128_store(o3+c,wasm_f32x4_add(wasm_v128_load(o3+c),wasm_f32x4_mul(p3,value)));
      }
    }
    if(times){auto now=Clock::now();times->pv_ms+=std::chrono::duration<double,std::milli>(now-mark).count();mark=now;}
    for(std::size_t c=0;c<1024;++c)o0[c]*=sigmoid(a.gates[qoffset+c]);
    if(times)times->gate_ms+=std::chrono::duration<double,std::milli>(Clock::now()-mark).count();
  }
}
} // namespace qwen35x::cpu::detail
