#pragma once
#include "qwen35x/cpu/q4_dot4.h"
#include "q8_0_internal.h"
namespace qwen35x::cpu::detail {
#if QWEN35X_Q8_0_HAS_AVX2_TU
void q4_dot4_matvec_avx2(const Q4_0BlockX8 *, const Q8_0BlockX1 *, float *, std::size_t, std::size_t) noexcept;
void q4_dot4_matmul_avx2(const Q4_0BlockX8 *, const Q8_0BlockX4 *, float *, std::size_t, std::size_t, std::size_t, std::size_t) noexcept;
Q4_0ArgmaxResult q4_dot4_argmax_avx2(const Q4_0BlockX8 *, const Q8_0BlockX1 *, const int *, float, std::size_t, std::size_t, std::size_t) noexcept;
#endif
#if QWEN35X_Q8_0_HAS_AVX_VNNI_TU
void q4_dot4_matvec_vex(const Q4_0BlockX8 *, const Q8_0BlockX1 *, float *, std::size_t, std::size_t) noexcept;
void q4_dot4_matmul_vex(const Q4_0BlockX8 *, const Q8_0BlockX4 *, float *, std::size_t, std::size_t, std::size_t, std::size_t) noexcept;
Q4_0ArgmaxResult q4_dot4_argmax_vex(const Q4_0BlockX8 *, const Q8_0BlockX1 *, const int *, float, std::size_t, std::size_t, std::size_t) noexcept;
#endif
#if QWEN35X_Q8_0_HAS_AVX512_VNNI_TU
void q4_dot4_matvec_evex(const Q4_0BlockX8 *, const Q8_0BlockX1 *, float *, std::size_t, std::size_t) noexcept;
void q4_dot4_matmul_evex(const Q4_0BlockX8 *, const Q8_0BlockX4 *, float *, std::size_t, std::size_t, std::size_t, std::size_t) noexcept;
Q4_0ArgmaxResult q4_dot4_argmax_evex(const Q4_0BlockX8 *, const Q8_0BlockX1 *, const int *, float, std::size_t, std::size_t, std::size_t) noexcept;
#endif
}
