#include "qwen35x/cpu/q4_quantizer.h"
#include "qwen35x/cpu/q4_dot4.h"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>

using namespace qwen35x::cpu;
int main(int argc, char ** argv) {
  if (argc == 2 && std::string(argv[1]) == "--basis-fixture") {
    float x[128], y[128];
    for(int i=0;i<128;++i)x[i]=(i-64)*0.03125F;
    q4_h128_transform_block(x,y,0);
    std::cout.precision(9);
    for(float v:y)std::cout<<v<<' ';
    return 0;
  }
  std::mt19937 random(1234);
  std::normal_distribution<float> normal;
  for (int trial = 0; trial < 1000; ++trial) {
    std::array<float, 256> values{}, before{}, after{}, importance{}, weighted_values{};
    for (int i = 0; i < 256; ++i) {
      float v = normal(random);
      if (trial == 0) v = float(i % 16 - 8);
      if (trial == 1) v = -float(i % 16 - 8);
      if (trial == 2) v = 0;
      if (trial == 3) v = 1.25F;
      if (trial == 4) v = -1.25F;
      if (trial == 5) v *= 1.0e-7F;
      if (trial == 6) v = i % 2 ? 65504.0F : -65504.0F;
      if (trial > 6 && i % 13 == 0) v *= 30;
      values[i] = v;
      importance[i] = i%5 == 0 ? 0.0F : i%7 == 0 ? 20.0F : 1.0F;
    }
    std::array<Q4_0Block, 8> legacy{}, fit{}, again{}, unpacked{}, original{};
    Q4_0BlockX8 packed{};
    q4_h128_quantize_transformed(values.data(), original.data(), 8);
    if (!q4_quantize_offline(values.data(), legacy.data(), 8, Q4Quantizer::legacy_absmax15) ||
        std::memcmp(original.data(), legacy.data(), sizeof(legacy)) != 0 ||
        !q4_quantize_offline(values.data(), fit.data(), 8, Q4Quantizer::mse16) ||
        !q4_quantize_offline(values.data(), again.data(), 8, Q4Quantizer::mse16) ||
        std::memcmp(fit.data(), again.data(), sizeof(fit)) != 0) return 1;
    q4_dot4_pack_rows_8(fit.data(), &packed, 8, 1);
    q4_dot4_unpack_rows_8(&packed, unpacked.data(), 8, 1);
    if (std::memcmp(fit.data(), unpacked.data(), sizeof(fit)) != 0) return 2;
    q4_0_dequantize(legacy.data(), before.data(), 8);
    q4_0_dequantize(fit.data(), after.data(), 8);
    std::array<Q4_0Block,8> weighted{};
    if(!q4_quantize_offline(values.data(), weighted.data(), 8, Q4Quantizer::mse16, importance.data()))return 7;
    q4_0_dequantize(weighted.data(), weighted_values.data(), 8);
    for (int b = 0; b < 8; ++b) {
      double old_error = 0, new_error = 0;
      double weighted_error=0, retained_error=0;
      for (int i = b*32; i < (b+1)*32; ++i) {
        old_error += std::pow(double(values[i])-before[i], 2);
        new_error += std::pow(double(values[i])-after[i], 2);
        weighted_error += importance[i]*std::pow(double(values[i])-weighted_values[i],2);
        retained_error += importance[i]*std::pow(double(values[i])-after[i],2);
      }
      if (new_error > old_error + 1e-12 || (trial < 2 && new_error != 0)) return 3;
      if (trial == 0 && fit[b].d != 0x3c00) return 4;
      if (trial == 1 && fit[b].d != 0xbc00) return 5;
      if(weighted_error > retained_error+1e-12)return 8;
    }
  }
  float invalid[32]{};
  Q4_0Block block;
  float bad_importance[32]{};bad_importance[0]=-1;
  if(q4_quantize_offline(invalid,&block,1,Q4Quantizer::mse16,bad_importance))return 9;
  for (float value : {std::numeric_limits<float>::infinity(),
                     std::numeric_limits<float>::quiet_NaN(), 1e10F}) {
    invalid[0] = value;
    if (q4_quantize_offline(invalid, &block, 1, Q4Quantizer::mse16)) return 6;
  }
  std::cout << "Offline Q4 quantizer tests passed\n";
}
