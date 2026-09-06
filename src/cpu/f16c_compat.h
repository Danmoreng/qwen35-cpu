#pragma once

#include <immintrin.h>

#include <cstdint>

namespace qwen35x::cpu::detail {

[[nodiscard]] inline float f16c_half_to_float(const std::uint16_t value) noexcept {
  const __m128i packed = _mm_cvtsi32_si128(static_cast<int>(value));
  return _mm_cvtss_f32(_mm_cvtph_ps(packed));
}

[[nodiscard]] inline std::uint16_t f16c_float_to_half(const float value) noexcept {
  const __m128i packed = _mm_cvtps_ph(
    _mm_set_ss(value), _MM_FROUND_TO_NEAREST_INT);
  return static_cast<std::uint16_t>(_mm_extract_epi16(packed, 0));
}

} // namespace qwen35x::cpu::detail
