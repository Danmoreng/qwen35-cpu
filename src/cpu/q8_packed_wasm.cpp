// SPDX-License-Identifier: MIT
#include "q4_0_internal.h"
#include "q8_0_internal.h"
#include <wasm_simd128.h>
#include <cmath>
#include <cstring>
namespace qwen35x::cpu::detail {
namespace {
template<typename Block>
void quantize(const float *input, Block &output, std::size_t lane) noexcept {
  v128_t values[8];
  auto maximum = wasm_f32x4_splat(0);
  for (int i=0; i<8; ++i) {
    values[i]=wasm_v128_load(input+i*4);
    const auto magnitude=wasm_f32x4_abs(values[i]);
    if (wasm_v128_any_true(wasm_i32x4_ge(magnitude,wasm_i32x4_splat(0x7f800000)))) {
      Q8_0BlockX1 scalar;
      q8_0_quantize_vector_1_scalar(input,&scalar,1);
      output.scales[lane]=scalar.scales[0]; output.sums[lane]=scalar.sums[0];
      std::memcpy(output.qs+lane*32,scalar.qs,32); return;
    }
    maximum=wasm_f32x4_max(maximum,magnitude);
  }
  maximum=wasm_f32x4_max(maximum,wasm_i32x4_shuffle(maximum,maximum,2,3,0,1));
  maximum=wasm_f32x4_max(maximum,wasm_i32x4_shuffle(maximum,maximum,1,0,3,2));
  const float scale=wasm_f32x4_extract_lane(maximum,0)/127.f;
  const float inverse=scale==0 ? 0 : 1.f/scale;
  if (!std::isfinite(inverse)) {
    Q8_0BlockX1 scalar; q8_0_quantize_vector_1_scalar(input,&scalar,1);
    output.scales[lane]=scalar.scales[0]; output.sums[lane]=scalar.sums[0];
    std::memcpy(output.qs+lane*32,scalar.qs,32); return;
  }
  output.scales[lane]=half_to_float(float_to_half(scale));
  v128_t integers[8]; auto sum=wasm_i32x4_splat(0);
  for (int i=0; i<8; ++i) {
    const auto scaled=wasm_f32x4_mul(values[i],wasm_f32x4_splat(inverse));
    const auto truncated=wasm_i32x4_trunc_sat_f32x4(scaled);
    const auto residual=wasm_f32x4_sub(scaled,wasm_f32x4_convert_i32x4(truncated));
    // std::round semantics, including FP32 values immediately below a half.
    auto q=wasm_i32x4_add(truncated,wasm_v128_and(wasm_f32x4_ge(residual,wasm_f32x4_splat(.5f)),wasm_i32x4_splat(1)));
    q=wasm_i32x4_sub(q,wasm_v128_and(wasm_f32x4_le(residual,wasm_f32x4_splat(-.5f)),wasm_i32x4_splat(1)));
    integers[i]=wasm_i32x4_min(wasm_i32x4_max(q,wasm_i32x4_splat(-127)),wasm_i32x4_splat(127));
    sum=wasm_i32x4_add(sum,integers[i]);
  }
  sum=wasm_i32x4_add(sum,wasm_i32x4_shuffle(sum,sum,2,3,0,1));
  sum=wasm_i32x4_add(sum,wasm_i32x4_shuffle(sum,sum,1,0,3,2));
  output.sums[lane]=static_cast<std::int16_t>(wasm_i32x4_extract_lane(sum,0));
  for (int half=0; half<2; ++half) {
    const auto a=wasm_i16x8_narrow_i32x4(integers[half*4],integers[half*4+1]);
    const auto b=wasm_i16x8_narrow_i32x4(integers[half*4+2],integers[half*4+3]);
    wasm_v128_store(output.qs+lane*32+half*16,wasm_i8x16_narrow_i16x8(a,b));
  }
}
}
void q8_0_quantize_vector_1_wasm(const float *input,Q8_0BlockX1 *output,std::size_t blocks) noexcept {
  for (std::size_t b=0; b<blocks; ++b) quantize(input+b*32,output[b],0);
}
void q8_0_quantize_vectors_4_wasm(const float *input,Q8_0BlockX4 *output,std::size_t tokens,std::size_t blocks) noexcept {
  for (std::size_t tile=0; tile<tokens/4; ++tile) for (std::size_t b=0; b<blocks; ++b)
    for (std::size_t t=0; t<4; ++t) quantize(input+(tile*4+t)*blocks*32+b*32,output[tile*blocks+b],t);
}
}
