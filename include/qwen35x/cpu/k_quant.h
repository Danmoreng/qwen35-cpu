#pragma once
#include "qwen35x/cpu/q8_0.h"
#include <cstddef>
#include <cstdint>

namespace qwen35x::cpu {
// GGML on-disk type IDs. Q4_K_M is a recipe, not a tensor type.
enum class KQuantType : std::uint32_t { q8_0=8, q4_k=12, q5_k=13, q6_k=14 };
constexpr std::size_t k_quant_block_bytes(KQuantType type) noexcept {
  switch(type) {
    case KQuantType::q4_k: return 144;
    case KQuantType::q5_k: return 176;
    case KQuantType::q6_k: return 210;
    case KQuantType::q8_0: return 272; // eight original 34-byte blocks
  }
  return 0;
}
struct KQuantActivation {
  float d[8];
  std::int8_t qs[256];
  std::int16_t sums[16];
};
// K weights use one signed FP32 activation scale per 256 values. Q8_0
// companions retain the existing eight FP16-rounded 32-value scales.
void k_quant_prepare(const float*, KQuantActivation*, std::size_t blocks,
                     bool q8_0, Q8_0Backend) noexcept;
void k_quant_dequantize(const std::uint8_t*, KQuantType, float*,
                        std::size_t blocks) noexcept;
// One immutable compressed row, one to four activation rows. Strides are in
// activation blocks and output floats. No allocation or executor recursion.
void k_quant_dot_rows(const std::uint8_t*, KQuantType,
                      const KQuantActivation*, std::size_t activation_stride,
                      float*, std::size_t output_stride, std::size_t vectors,
                      std::size_t blocks, Q8_0Backend) noexcept;
}
