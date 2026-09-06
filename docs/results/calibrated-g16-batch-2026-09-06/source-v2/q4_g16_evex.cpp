#define G16_VNNI 1
#define G16_BATCH_FN q4_g16_matmul_evex
#include "q4_g16_batch_simd.inl"

namespace qwen35x::cpu {
namespace {
struct WideHalf { __m512i lo0,hi0,lo1,hi1; __m512 scales; };
G16_INLINE __m512i join_rows(const void* low,const void* high) noexcept {
  return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256(static_cast<const __m256i*>(low))),
    _mm256_loadu_si256(static_cast<const __m256i*>(high)),1);
}
G16_INLINE WideHalf wide_half(const Q4G16Tile& a,const Q4G16Tile& b,int half) noexcept {
  const auto mask=_mm512_set1_epi32(0x0f0f0f0f);
  const auto x=join_rows(a.qs+half*64,b.qs+half*64);
  const auto y=join_rows(a.qs+half*64+32,b.qs+half*64+32);
  const auto scales=_mm256_set_m128i(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b.d[half])),
                                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(a.d[half])));
  return {_mm512_and_si512(x,mask),_mm512_and_si512(_mm512_srli_epi32(x,4),mask),
    _mm512_and_si512(y,mask),_mm512_and_si512(_mm512_srli_epi32(y,4),mask),_mm512_cvtph_ps(scales)};
}
G16_INLINE __m512i wide_broadcast(const std::int8_t* q) noexcept {
  int x;std::memcpy(&x,q,4);return _mm512_set1_epi32(x);
}
template<std::size_t T>
G16_INLINE void wide_accumulate(const WideHalf& w,const Q8_0BlockX1* a,const std::int16_t* sums,
    std::size_t blocks,std::size_t b,int half,__m512& acc) noexcept {
  const auto& v=a[T*blocks+b];const auto* q=v.qs+half*16;
  auto dot=_mm512_dpbusd_epi32(_mm512_setzero_si512(),w.lo0,wide_broadcast(q));
  dot=_mm512_dpbusd_epi32(dot,w.hi0,wide_broadcast(q+4));
  dot=_mm512_dpbusd_epi32(dot,w.lo1,wide_broadcast(q+8));
  dot=_mm512_dpbusd_epi32(dot,w.hi1,wide_broadcast(q+12));
  dot=_mm512_sub_epi32(dot,_mm512_set1_epi32(8*sums[(T*blocks+b)*2+half]));
  const auto scale=_mm512_mul_ps(w.scales,_mm512_set1_ps(v.scales[0]));
  acc=_mm512_fmadd_ps(_mm512_cvtepi32_ps(dot),scale,acc);
}
template<std::size_t... T>
G16_INLINE void wide_tile(const Q4G16Tile* w,const Q8_0BlockX1* a,const std::int16_t* sums,
    float* out,std::size_t blocks,std::size_t stride,std::index_sequence<T...>) noexcept {
  __m512 acc[sizeof...(T)]={((void)T,_mm512_setzero_ps())...};
  for(std::size_t b=0;b<blocks;++b) {
    const auto low=wide_half(w[b],w[blocks+b],0);
    (wide_accumulate<T>(low,a,sums,blocks,b,0,acc[T]),...);
    const auto high=wide_half(w[b],w[blocks+b],1);
    (wide_accumulate<T>(high,a,sums,blocks,b,1,acc[T]),...);
  }
  (_mm512_storeu_ps(out+T*stride,acc[T]),...);
}
}
void q4_g16_matmul_wide(const Q4G16Tile* w,const Q8_0BlockX1* a,const std::int16_t* sums,
    float* out,std::size_t rows,std::size_t count,std::size_t blocks,std::size_t stride) noexcept {
  std::size_t t=0;
  const auto complete_rows=rows/16*16;
  for(;t+16<=count;t+=16) for(std::size_t r=0;r<complete_rows;r+=16)
    wide_tile(w+(r/8)*blocks,a+t*blocks,sums+t*blocks*2,out+t*stride+r,blocks,stride,std::make_index_sequence<16>{});
  for(;t+4<=count;t+=4) for(std::size_t r=0;r<complete_rows;r+=16)
    wide_tile(w+(r/8)*blocks,a+t*blocks,sums+t*blocks*2,out+t*stride+r,blocks,stride,std::make_index_sequence<4>{});
  for(;t+2<=count;t+=2) for(std::size_t r=0;r<complete_rows;r+=16)
    wide_tile(w+(r/8)*blocks,a+t*blocks,sums+t*blocks*2,out+t*stride+r,blocks,stride,std::make_index_sequence<2>{});
  for(;t<count;++t) for(std::size_t r=0;r<complete_rows;r+=16)
    wide_tile(w+(r/8)*blocks,a+t*blocks,sums+t*blocks*2,out+t*stride+r,blocks,stride,std::make_index_sequence<1>{});
  if(rows!=complete_rows)
    q4_g16_matmul_evex(w+(complete_rows/8)*blocks,a,sums,out+complete_rows,rows-complete_rows,count,blocks,stride);
}
}
