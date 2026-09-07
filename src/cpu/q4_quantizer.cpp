#include "qwen35x/cpu/q4_quantizer.h"
#include "q8_0_internal.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace qwen35x::cpu {
namespace {
int code(const Q4_0Block & block, int i) {
  return ((block.qs[i % 16] >> (i < 16 ? 0 : 4)) & 15) - 8;
}
double objective(const float * w, const float * h, const Q4_0Block & block) {
  const double d = detail::half_to_float(block.d);
  double error = 0;
  for (int i = 0; i < 32; ++i) {
    const double delta = w[i] - d * code(block, i);
    error += (h ? h[i] : 1.0) * delta * delta;
  }
  return error;
}
Q4_0Block assign(const float * w, float scale) {
  Q4_0Block block{};
  block.d = detail::float_to_half(scale);
  const double d = detail::half_to_float(block.d);
  for (int i = 0; i < 32; ++i) {
    const int q = d == 0 ? 0 : static_cast<int>(std::clamp(std::round(w[i] / d), -8.0, 7.0));
    block.qs[i % 16] |= static_cast<unsigned char>((q + 8) << (i < 16 ? 0 : 4));
  }
  return block;
}
}

bool q4_quantize_offline(const float * input, Q4_0Block * output,
    std::size_t blocks, Q4Quantizer recipe, const float * importance) noexcept {
  if (!input || !output) return false;
  if (recipe != Q4Quantizer::legacy_absmax15 && recipe != Q4Quantizer::mse16) return false;
  for (std::size_t b = 0; b < blocks; ++b) {
    const float * w = input + 32*b;
    const float * h = importance ? importance + 32*b : nullptr;
    float maximum = 0;
    for (int i = 0; i < 32; ++i) {
      if (!std::isfinite(w[i]) || (h && (!std::isfinite(h[i]) || h[i] < 0))) return false;
      maximum = std::max(maximum, std::fabs(w[i]));
    }
    if (!std::isfinite(detail::half_to_float(detail::float_to_half(maximum / 7)))) return false;
    q4_h128_quantize_transformed(w, output + b, 1);
    if (recipe == Q4Quantizer::legacy_absmax15 || maximum == 0) continue;
    double best = objective(w, h, output[b]);
    if (h) {
      Q4_0Block unweighted;
      if (!q4_quantize_offline(w, &unweighted, 1, recipe)) return false;
      const double error = objective(w, h, unweighted);
      if (error < best) { best = error; output[b] = unweighted; }
    }
    // Search both scale orientations, including the exact [-8,7] grid.
    for (float sign : {1.0F, -1.0F}) {
      for (float divisor : {7.0F, 7.5F, 8.0F, 8.5F, 9.0F}) {
        float scale = sign * maximum / divisor;
        for (int iteration = 0; iteration < 4; ++iteration) {
          const auto candidate = assign(w, scale);
          const double error = objective(w, h, candidate);
          if (error < best) { best = error; output[b] = candidate; }
          double numerator = 0, denominator = 0;
          for (int i = 0; i < 32; ++i) {
            const int q = code(candidate, i);
            const double weight = h ? h[i] : 1.0;
            numerator += weight * w[i] * q;
            denominator += weight * q * q;
          }
          if (denominator == 0) break;
          scale = static_cast<float>(numerator / denominator);
          const auto half = detail::float_to_half(scale);
          if (!std::isfinite(detail::half_to_float(half)) || half == candidate.d) break;
        }
      }
    }
  }
  return true;
}

bool q4_quantize_covariance128(const float * input, Q4_0Block * output,
    std::size_t columns, const float * importance, const float * covariance) noexcept {
  if (!input || !output || !importance || !covariance || !columns || columns % 128) return false;
  for (std::size_t base=0; base<columns; base+=128) {
    const float * h=covariance+(base/128)*32768;
    const float * u=h+16384;
    Q4_0Block baseline[4], candidate[4]{};
    if(!q4_quantize_offline(input+base,baseline,4,Q4Quantizer::mse16,importance+base)) return false;
    float work[128]; std::copy_n(input+base,128,work);
    for(int group=0; group<4; ++group) {
      // Choose the actual stored scale before compensating codes within G32.
      Q4_0Block fit;
      if(!q4_quantize_offline(work+32*group,&fit,1,Q4Quantizer::mse16,importance+base+32*group)) return false;
      candidate[group].d=fit.d;
      const double d=detail::half_to_float(fit.d);
      for(int lane=0;lane<32;++lane) {
        const int i=32*group+lane;
        if(!std::isfinite(work[i]) || !std::isfinite(u[i*128+i]) || u[i*128+i]<=0) return false;
        const int q=d==0?0:static_cast<int>(std::clamp(std::round(work[i]/d),-8.0,7.0));
        candidate[group].qs[lane%16] |= static_cast<unsigned char>((q+8)<<(lane<16?0:4));
        const double error=(work[i]-d*q)/u[i*128+i];
        for(int j=i+1;j<128;++j) work[j]-=static_cast<float>(error*u[i*128+j]);
      }
    }
    const auto loss=[&](const Q4_0Block * quantized) {
      double error[128];
      for(int i=0;i<128;++i) error[i]=input[base+i]-
          double(detail::half_to_float(quantized[i/32].d))*code(quantized[i/32],i%32);
      double sum=0;
      for(int i=0;i<128;++i) {
        double dot=0;
        for(int j=0;j<128;++j) dot+=h[i*128+j]*error[j];
        sum+=error[i]*dot;
      }
      return sum;
    };
    const double old_loss=loss(baseline), new_loss=loss(candidate);
    if(!std::isfinite(old_loss) || !std::isfinite(new_loss)) return false;
    std::copy_n(new_loss<old_loss?candidate:baseline,4,output+base/32);
  }
  return true;
}
}
