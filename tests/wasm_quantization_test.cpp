// SPDX-License-Identifier: MIT
#include "qwen35x/cpu/q4_0.h"
#include "qwen35x/cpu/q4_h128.h"
#include "qwen35x/cpu/activation.h"
#include "qwen35x/cpu/full_attention.h"
#include "../src/cpu/q4_0_internal.h"
#include "../src/cpu/q8_0_internal.h"
#include <bit>
#include "attention_partition_cases.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
using namespace qwen35x::cpu;
template<typename Block>
bool equal_fields(const Block &a,const Block &b,std::size_t lanes) {
  for (std::size_t t=0;t<lanes;++t)
    if (std::bit_cast<std::uint32_t>(a.scales[t])!=std::bit_cast<std::uint32_t>(b.scales[t]) || a.sums[t]!=b.sums[t] ||
        std::memcmp(a.qs+t*32,b.qs+t*32,32)) return false;
  return true;
}
int main() {
  if(!attention_partition_cases())return 9;
  std::mt19937 rng(91827);
  for (int pattern=0;pattern<10;++pattern) for (std::size_t blocks:{1U,4U,32U,64U,112U}) {
    std::vector<float> input(blocks*32*8);
    for (std::size_t i=0;i<input.size();++i) {
      float value=static_cast<int>(rng()%20001)-10000;
      switch(pattern) {
        case 0: value=0;break;
        case 1: value=-0.f;break;
        case 2: value=(i%32==0)?127.f:std::nextafter(float(i%127)+.5f,-std::numeric_limits<float>::infinity());break;
        case 3: value=(i%32==0)?127.f:-(float(i%127)+.5f);break;
        case 4: value*=0x1p-140f;break;
        case 5: value*=0x1p-120f;break;
        case 6: value*=0x1p110f;break;
        case 7: value=i%32==0?std::numeric_limits<float>::infinity():value;break;
        case 8: value=i%32==0?std::numeric_limits<float>::quiet_NaN():value;break;
        default: value*=.001f;
      }
      input[i]=value;
    }
    std::vector<Q8_0BlockX1> one(blocks),ref(blocks);
    for (int t=0;t<8;++t) {
      q8_0_quantize_vector_1(input.data()+t*blocks*32,one.data(),blocks);
      detail::q8_0_quantize_vector_1_scalar(input.data()+t*blocks*32,ref.data(),blocks);
      for (std::size_t b=0;b<blocks;++b) if(!equal_fields(one[b],ref[b],1)) return 1;
    }
    std::vector<Q8_0BlockX4> four(blocks*2),four_ref(four.size());
    q8_0_quantize_vectors_4(input.data(),four.data(),8,blocks);
    detail::q8_0_quantize_vectors_4_scalar(input.data(),four_ref.data(),8,blocks);
    for (std::size_t b=0;b<four.size();++b) if(!equal_fields(four[b],four_ref[b],4)) return 2;
  }
  for (auto seed:{std::uint64_t{0},q4_h128_default_sign_seed,std::uint64_t{918273}})
    for (std::size_t width:{128U,1024U,2048U,3584U}) {
      std::vector<float> input(width*4);
      for(auto &x:input)x=(static_cast<int>(rng()%20001)-10000)*.0003f;
      std::vector<Q8_0BlockX4> actual(width/32),reference(actual.size());
      if(!q4_h128_prepare_activations_4(input.data(),actual.data(),4,width,seed) ||
         !q4_h128_prepare_activations_4(input.data(),reference.data(),4,width,seed,Q8_0Backend::scalar))return 3;
      for(std::size_t b=0;b<actual.size();++b)if(!equal_fields(actual[b],reference[b],4))return 4;
  }
  for (std::size_t width : {128U, 1024U, 3584U})
    for (std::size_t count : {4U, 8U, 12U, 32U}) {
      std::vector<float> gate_up(width*count*2), hidden(width*count), reference(hidden.size());
      for(auto &x:gate_up)x=(static_cast<int>(rng()%20001)-10000)*.001f;
      std::vector<Q8_0BlockX4> actual(width/32*count/4), expected(actual.size());
      for(std::size_t t=0;t<count;++t)
        silu_mul_f32(gate_up.data()+t*2*width,gate_up.data()+t*2*width+width,
                     reference.data()+t*width,width);
      if(!q4_h128_prepare_activations_4(reference.data(),expected.data(),count,width) ||
         !q4_h128_prepare_silu_activations_4(gate_up.data(),hidden.data(),actual.data(),
           count,width,q4_h128_default_sign_seed,Q8_0Backend::auto_select)) return 5;
      if(std::memcmp(hidden.data(),reference.data(),hidden.size()*sizeof(float)))return 6;
      for(std::size_t b=0;b<actual.size();++b)if(!equal_fields(actual[b],expected[b],4))return 7;
    }
  // FP16 storage must match the same scalar attention arithmetic after exact
  // widening. This isolates storage conversion from intentional FP16 error.
  for (std::size_t count : {1U, 3U, 31U, 32U, 33U})
    for (int position : {0, 127, 1024}) {
      constexpr int heads=4, kv_heads=2, dim=256;
      const std::size_t length=position+count, qw=heads*dim, kw=kv_heads*dim;
      std::vector<float> q(count*qw),g(q.size()),k(length*kw),v(k.size());
      for(auto *a : {&q,&g,&k,&v})
        for(auto &x:*a)x=(static_cast<int>(rng()%20001)-10000)*.0001f;
      std::vector<std::uint16_t> kh(k.size()),vh(v.size());
      attention_cache_store_f16(k.data(),kh.data(),k.size());
      attention_cache_store_f16(v.data(),vh.data(),v.size());
      for(std::size_t i=0;i<k.size();++i) {
        k[i]=detail::half_to_float(kh[i]);v[i]=detail::half_to_float(vh[i]);
      }
      std::vector<float> a(q.size()),b(q.size()),scores(length);
      causal_attention_batch_rows(q.data(),g.data(),k.data(),v.data(),nullptr,nullptr,
        scores.data(),a.data(),length,qw,kw,position,heads,kv_heads,dim,1.f/16,
        0,count*heads,Q8_0Backend::scalar,true);
      causal_attention_batch_rows(q.data(),g.data(),nullptr,nullptr,kh.data(),vh.data(),
        scores.data(),b.data(),length,qw,kw,position,heads,kv_heads,dim,1.f/16,
        0,count*heads,Q8_0Backend::scalar,true);
      if(std::memcmp(a.data(),b.data(),a.size()*sizeof(float)))return 8;
    }
  std::cout<<"WASM Q8 exact scalar parity: rounding neighbours, exceptional inputs, X1/X4, H128 widths/seeds fused SiLU fields and FP16 KV widening passed\n";
}
