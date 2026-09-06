#include "qwen35x/cpu/q4_g16.h"
#include "f16c_compat.h"
#include <immintrin.h>
#include <cstring>
#include <utility>
#if defined(_MSC_VER)
#define G16_INLINE __forceinline
#else
#define G16_INLINE inline __attribute__((always_inline))
#endif
namespace qwen35x::cpu {
namespace {
struct G16Half { __m256i lo0,hi0,lo1,hi1; __m256 scales; };
G16_INLINE G16Half load_half(const Q4G16Tile& w,int half) noexcept {
  const auto mask=_mm256_set1_epi8(15);
  const auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.qs+half*64));
  const auto b=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.qs+half*64+32));
  return {_mm256_and_si256(a,mask),_mm256_and_si256(_mm256_srli_epi16(a,4),mask),
          _mm256_and_si256(b,mask),_mm256_and_si256(_mm256_srli_epi16(b,4),mask),
          _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w.d[half])))};
}
G16_INLINE __m256i g16_broadcast(const std::int8_t* x) noexcept {
  int bits;std::memcpy(&bits,x,4);return _mm256_set1_epi32(bits);
}
template<std::size_t T>
G16_INLINE void g16_accumulate(const G16Half& w,const Q8_0BlockX1* a,
    const std::int16_t* sums,std::size_t blocks,std::size_t block,int half,__m256& acc) noexcept {
  const auto& v=a[T*blocks+block];const auto* q=v.qs+half*16;
#if G16_VNNI
  auto dot=_mm256_dpbusd_epi32(_mm256_setzero_si256(),w.lo0,g16_broadcast(q));
  dot=_mm256_dpbusd_epi32(dot,w.hi0,g16_broadcast(q+4));
  dot=_mm256_dpbusd_epi32(dot,w.lo1,g16_broadcast(q+8));
  dot=_mm256_dpbusd_epi32(dot,w.hi1,g16_broadcast(q+12));
#else
  // Eight pair-lane products at most: 8*15*128=15360 fits int16.
  auto pairs=_mm256_maddubs_epi16(w.lo0,g16_broadcast(q));
  pairs=_mm256_add_epi16(pairs,_mm256_maddubs_epi16(w.hi0,g16_broadcast(q+4)));
  pairs=_mm256_add_epi16(pairs,_mm256_maddubs_epi16(w.lo1,g16_broadcast(q+8)));
  pairs=_mm256_add_epi16(pairs,_mm256_maddubs_epi16(w.hi1,g16_broadcast(q+12)));
  auto dot=_mm256_madd_epi16(pairs,_mm256_set1_epi16(1));
#endif
  dot=_mm256_sub_epi32(dot,_mm256_set1_epi32(8*sums[(T*blocks+block)*2+half]));
  const auto scale=_mm256_mul_ps(w.scales,_mm256_set1_ps(v.scales[0]));
  acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(dot),scale,acc);
}
template<std::size_t... T>
G16_INLINE void g16_tile(const Q4G16Tile* w,const Q8_0BlockX1* a,const std::int16_t* sums,
    float* out,std::size_t blocks,std::size_t stride,std::index_sequence<T...>) noexcept {
  __m256 acc[sizeof...(T)]={((void)T,_mm256_setzero_ps())...};
  for(std::size_t b=0;b<blocks;++b) {
    const auto low=load_half(w[b],0);
    (g16_accumulate<T>(low,a,sums,blocks,b,0,acc[T]),...);
    const auto high=load_half(w[b],1);
    (g16_accumulate<T>(high,a,sums,blocks,b,1,acc[T]),...);
  }
  (_mm256_storeu_ps(out+T*stride,acc[T]),...);
}
}
void G16_BATCH_FN(const Q4G16Tile* w,const Q8_0BlockX1* a,const std::int16_t* sums,
    float* out,std::size_t rows,std::size_t count,std::size_t blocks,std::size_t stride) noexcept {
  std::size_t t=0;
#if G16_VNNI
  for(;t+16<=count;t+=16) for(std::size_t r=0;r<rows/8;++r)
    g16_tile(w+r*blocks,a+t*blocks,sums+t*blocks*2,out+t*stride+r*8,blocks,stride,std::make_index_sequence<16>{});
#endif
  for(;t+4<=count;t+=4) for(std::size_t r=0;r<rows/8;++r)
    g16_tile(w+r*blocks,a+t*blocks,sums+t*blocks*2,out+t*stride+r*8,blocks,stride,std::make_index_sequence<4>{});
  for(;t+2<=count;t+=2) for(std::size_t r=0;r<rows/8;++r)
    g16_tile(w+r*blocks,a+t*blocks,sums+t*blocks*2,out+t*stride+r*8,blocks,stride,std::make_index_sequence<2>{});
  for(;t<count;++t) for(std::size_t r=0;r<rows/8;++r)
    g16_tile(w+r*blocks,a+t*blocks,sums+t*blocks*2,out+t*stride+r*8,blocks,stride,std::make_index_sequence<1>{});
}
}
