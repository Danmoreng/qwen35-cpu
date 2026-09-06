#include "qwen35x/cpu/q4_g16.h"
#include "qwen35x/cpu/q4_quantizer.h"
#include "q8_0_internal.h"
#include <algorithm>
#include <cmath>
namespace qwen35x::cpu {
const char* q4_g16_batch_kernel_name(Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX512_VNNI_TU
  if(q8_0_resolve_backend(backend)==Q8_0Backend::avx512_vnni) return "q4-g16-vnni512-16rows-16vectors";
#endif
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if(q8_0_backend_uses_avx2(backend)) return "q4-g16-avx2-8rows-4vectors";
#endif
  return "q4-g16-scalar";
}
#if QWEN35X_Q8_0_HAS_AVX2_TU
void q4_g16_matvec_avx2(const Q4G16Tile*,const Q8_0BlockX1*,float*,std::size_t,std::size_t) noexcept;
void q4_g16_matmul_avx2(const Q4G16Tile*,const Q8_0BlockX1*,const std::int16_t*,float*,std::size_t,std::size_t,std::size_t,std::size_t) noexcept;
#endif
#if QWEN35X_Q8_0_HAS_AVX512_VNNI_TU
void q4_g16_matmul_evex(const Q4G16Tile*,const Q8_0BlockX1*,const std::int16_t*,float*,std::size_t,std::size_t,std::size_t,std::size_t) noexcept;
void q4_g16_matmul_wide(const Q4G16Tile*,const Q8_0BlockX1*,const std::int16_t*,float*,std::size_t,std::size_t,std::size_t,std::size_t) noexcept;
#endif
namespace {
int code(const Q4_0Block& b,int i) { return ((b.qs[i%16]>>(i<16?0:4))&15)-8; }
int packed_code(const Q4G16Tile& b,int row,int i) {
  return ((b.qs[32*(i/8)+4*row+i%4]>>(i%8<4?0:4))&15)-8;
}
}
bool q4_g16_pack(const float* input,Q4G16Tile* output,std::size_t rows,std::size_t cols,const float* importance) noexcept {
  if(!input || !output || !rows || rows%8 || !cols || cols%32) return false;
  const auto blocks=cols/32;
  for(std::size_t row=0;row<rows;++row) for(std::size_t b=0;b<blocks;++b) {
    const float* w=input+row*cols+b*32;
    const float* h=importance?importance+b*32:nullptr;
    Q4_0Block parent;
    if(!q4_quantize_offline(w,&parent,1,Q4Quantizer::mse16,h)) return false;
    auto& tile=output[(row/8)*blocks+b];
    for(int half=0;half<2;++half) {
      float duplicate[32];
      float duplicate_h[32];
      for(int i=0;i<32;++i) duplicate[i]=w[half*16+i%16];
      if(h) for(int i=0;i<32;++i) duplicate_h[i]=h[half*16+i%16];
      Q4_0Block child;
      if(!q4_quantize_offline(duplicate,&child,1,Q4Quantizer::mse16,h?duplicate_h:nullptr)) return false;
      double pe=0,ce=0;
      const double pd=detail::half_to_float(parent.d),cd=detail::half_to_float(child.d);
      for(int i=0;i<16;++i) {
        const double weight=h?h[half*16+i]:1.0;
        pe+=weight*std::pow(w[half*16+i]-pd*code(parent,half*16+i),2);
        ce+=weight*std::pow(w[half*16+i]-cd*code(child,i),2);
      }
      tile.d[half][row%8]=ce<pe?child.d:parent.d;
      for(int j=0;j<16;++j) {
        const int i=half*16+j;
        const int q=(ce<pe?code(child,j):code(parent,i))+8;
        auto& byte=tile.qs[32*(i/8)+4*(row%8)+i%4];
        const int shift=i%8<4?0:4;
        byte=static_cast<std::uint8_t>((byte&~(15<<shift))|(q<<shift));
      }
    }
  }
  return true;
}
void q4_g16_dequantize_row(const Q4G16Tile* w,std::size_t row,float* out,std::size_t blocks) noexcept {
  for(std::size_t b=0;b<blocks;++b) for(int i=0;i<32;++i)
    out[b*32+i]=detail::half_to_float(w[(row/8)*blocks+b].d[i/16][row%8])*
      packed_code(w[(row/8)*blocks+b],static_cast<int>(row%8),i);
}
void q4_g16_matvec(const Q4G16Tile* w,const Q8_0BlockX1* a,float* out,std::size_t rows,std::size_t blocks,Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if(q8_0_backend_uses_avx2(q8_0_resolve_backend(backend))) {
    q4_g16_matvec_avx2(w,a,out,rows,blocks); return;
  }
#endif
  for(std::size_t row=0;row<rows;++row) {
    float sum=0;
    for(std::size_t b=0;b<blocks;++b) for(int half=0;half<2;++half) {
      int dot=0;
      for(int i=half*16;i<half*16+16;++i) dot+=packed_code(w[(row/8)*blocks+b],static_cast<int>(row%8),i)*a[b].qs[i];
      sum+=dot*(detail::half_to_float(w[(row/8)*blocks+b].d[half][row%8])*a[b].scales[0]);
    }
    out[row]=sum;
  }
}
void q4_g16_matmul(const Q4G16Tile* w,const Q8_0BlockX1* a,const std::int16_t* sums,
    float* out,std::size_t rows,std::size_t count,std::size_t blocks,std::size_t stride,Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX512_VNNI_TU
  if(q8_0_resolve_backend(backend)==Q8_0Backend::avx512_vnni) {
    q4_g16_matmul_wide(w,a,sums,out,rows,count,blocks,stride);return;
  }
#endif
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if(q8_0_backend_uses_avx2(backend)) {
    q4_g16_matmul_avx2(w,a,sums,out,rows,count,blocks,stride);return;
  }
#endif
  for(std::size_t t=0;t<count;++t) q4_g16_matvec(w,a+t*blocks,out+t*stride,rows,blocks,Q8_0Backend::scalar);
}
}
