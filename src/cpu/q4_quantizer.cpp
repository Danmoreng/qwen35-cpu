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
}
