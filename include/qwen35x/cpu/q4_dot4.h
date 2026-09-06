#pragma once
#include "qwen35x/cpu/q4_0.h"

namespace qwen35x::cpu {
// Explicit DOT4 layout, same 144-byte storage type as X8, different encoding.
// qs[32*t+4*r+j] = u[r][8*t+j] | (u[r][8*t+4+j] << 4).
void q4_dot4_pack_rows_8(const Q4_0Block *, Q4_0BlockX8 *, std::size_t rows, std::size_t blocks) noexcept;
void q4_dot4_unpack_rows_8(const Q4_0BlockX8 *, Q4_0Block *, std::size_t rows, std::size_t blocks) noexcept;
void q4_dot4_dequantize_row(const Q4_0BlockX8 *, std::size_t row, float *, std::size_t blocks) noexcept;
void q4_dot4_matvec(const Q4_0BlockX8 *, const Q8_0BlockX1 *, float *, std::size_t rows, std::size_t blocks,
                    Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;
void q4_dot4_matmul(const Q4_0BlockX8 *, const Q8_0BlockX4 *, float *, std::size_t rows, std::size_t vectors,
                    std::size_t blocks, std::size_t stride, Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;
Q4_0ArgmaxResult q4_dot4_argmax(const Q4_0BlockX8 *, const Q8_0BlockX1 *, const int *, float penalty,
                    std::size_t offset, std::size_t rows, std::size_t blocks,
                    Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;
}
