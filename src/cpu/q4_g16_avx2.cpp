#include "qwen35x/cpu/q4_g16.h"
#include "f16c_compat.h"
#include <immintrin.h>
#include <cstring>
namespace qwen35x::cpu {
void q4_g16_matvec_avx2(const Q4G16Tile* w,const Q8_0BlockX1* a,float* out,std::size_t rows,std::size_t blocks) noexcept {
  const auto mask=_mm256_set1_epi8(15);
  for(std::size_t tile=0;tile<rows/8;++tile) {
    auto sum=_mm256_setzero_ps();
    for(std::size_t b=0;b<blocks;++b) for(int half=0;half<2;++half) {
      auto pairs=_mm256_setzero_si256();
      int correction=0;
      for(int chunk=0;chunk<2;++chunk) {
        const int offset=half*16+chunk*8;
        auto raw=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[tile*blocks+b].qs+offset*4));
        int low,high;
        std::memcpy(&low,a[b].qs+offset,4); std::memcpy(&high,a[b].qs+offset+4,4);
        pairs=_mm256_add_epi16(pairs,_mm256_maddubs_epi16(_mm256_and_si256(raw,mask),_mm256_set1_epi32(low)));
        pairs=_mm256_add_epi16(pairs,_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(raw,4),mask),_mm256_set1_epi32(high)));
        for(int i=0;i<8;++i) correction+=a[b].qs[offset+i];
      }
      // Eight products per int16 lane: <=8*15*128=15360, no saturation/overflow.
      auto dot=_mm256_sub_epi32(_mm256_madd_epi16(pairs,_mm256_set1_epi16(1)),_mm256_set1_epi32(8*correction));
      auto scale=_mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w[tile*blocks+b].d[half]))),_mm256_set1_ps(a[b].scales[0]));
      sum=_mm256_fmadd_ps(_mm256_cvtepi32_ps(dot),scale,sum);
    }
    _mm256_storeu_ps(out+tile*8,sum);
  }
}
}

#define G16_VNNI 0
#define G16_BATCH_FN q4_g16_matmul_avx2
#include "q4_g16_batch_simd.inl"
