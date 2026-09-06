#pragma once
#include "qwen35x/cpu/q4_h128.h"

namespace qwen35x::cpu {
enum class Q4Quantizer { legacy_absmax15, mse16 };

// Offline only. Rejects non-finite inputs and blocks whose legacy scale
// overflows binary16. Zero/subnormal scales use the stored binary16 value.
// Code ties round away from zero; equal objective candidates retain the first.
// Optional importance is in the SAME basis as input, one weight per value.
[[nodiscard]] bool q4_quantize_offline(const float * input, Q4_0Block * output,
    std::size_t blocks, Q4Quantizer recipe, const float * importance = nullptr) noexcept;
}
