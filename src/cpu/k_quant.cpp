#include "k_quant_internal.h"
#include <algorithm>
#include <cmath>
namespace qwen35x::cpu {
void k_quant_prepare(const float* x, KQuantActivation* out, std::size_t blocks,
                     bool q8, Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if(!q8 && q8_0_backend_uses_avx2(backend)) {detail::k_quant_prepare_avx2(x,out,blocks);return;}
#endif
  for(std::size_t b=0;b<blocks;++b,x+=256) {
    auto& a=out[b];
    if(q8) {
      Q8_0Block tmp[8];
      q8_0_quantize_with_scales(x,tmp,a.d,8,backend);
      for(int g=0;g<8;++g) std::memcpy(a.qs+g*32,tmp[g].qs,32);
    } else {
      float max=0, amax=0;
      for(int i=0;i<256;++i) if(std::abs(x[i])>amax) {amax=std::abs(x[i]);max=x[i];}
      const float inv=amax ? -127.0f/max : 0;
      const float d=amax ? 1.0f/inv : 0;
      std::fill_n(a.d,8,d);
      // nearbyint uses round-to-nearest-even under the engine's default FP
      // environment, matching GGML's Q8_K reference quantizer.
      for(int i=0;i<256;++i) a.qs[i]=static_cast<std::int8_t>(
        std::clamp(std::nearbyint(inv*x[i]),-127.0f,127.0f));
    }
    for(int g=0;g<16;++g) {
      int sum=0;for(int i=0;i<16;++i)sum+=a.qs[g*16+i];
      a.sums[g]=static_cast<std::int16_t>(sum);
    }
  }
}
void k_quant_dequantize(const std::uint8_t* w,KQuantType type,float* out,
                        std::size_t blocks) noexcept {
  for(std::size_t b=0;b<blocks;++b,w+=k_quant_block_bytes(type))
    for(int g=0;g<8;++g) {
      std::int16_t codes[32];float d0,d1,bias;
      detail::k_group(w,type,g,codes,d0,d1,bias);
      for(int i=0;i<32;++i) out[b*256+g*32+i]=(i<16?d0:d1)*codes[i]+bias;
    }
}
void k_quant_dot_rows(const std::uint8_t* w,KQuantType type,
  const KQuantActivation* x,std::size_t stride,float* out,std::size_t out_stride,
  std::size_t vectors,std::size_t blocks,Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if(q8_0_backend_uses_avx2(backend)) {
    detail::k_quant_dot_rows_avx2(w,type,x,stride,out,out_stride,vectors,blocks);return;
  }
#endif
  float accum[4]{};
  for(std::size_t b=0;b<blocks;++b,w+=k_quant_block_bytes(type))
    for(int g=0;g<8;++g) {
      std::int16_t codes[32];float d0,d1,bias;
      detail::k_group(w,type,g,codes,d0,d1,bias);
      for(std::size_t v=0;v<vectors;++v) {
        const auto& a=x[v*stride+b];int dot0=0,dot1=0;
        for(int i=0;i<16;++i) {
          dot0+=codes[i]*a.qs[g*32+i];dot1+=codes[i+16]*a.qs[g*32+i+16];
        }
        accum[v]+=a.d[g]*(d0*dot0+d1*dot1+bias*(a.sums[g*2]+a.sums[g*2+1]));
      }
    }
  for(std::size_t v=0;v<vectors;++v)out[v*out_stride]=accum[v];
}
}
