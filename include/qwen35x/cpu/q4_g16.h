#pragma once
#include "qwen35x/cpu/q4_0.h"
namespace qwen35x::cpu {
// Eight rows x 32 columns. DOT4 nibble ordering, independent half scales.
struct Q4G16Tile { std::uint16_t d[2][8]; std::uint8_t qs[128]; };
static_assert(sizeof(Q4G16Tile)==160);
const char* q4_g16_batch_kernel_name(Q8_0Backend backend) noexcept;
bool q4_g16_pack(const float*, Q4G16Tile*, std::size_t rows, std::size_t columns,
                 const float* column_importance = nullptr) noexcept;
void q4_g16_dequantize_row(const Q4G16Tile*,std::size_t row,float*,std::size_t blocks) noexcept;
void q4_g16_matvec(const Q4G16Tile*,const Q8_0BlockX1*,float*,std::size_t rows,
                  std::size_t blocks,Q8_0Backend backend) noexcept;
// Sixteen vectors for one G32 input block. Transient block-major scratch;
// no new activation rounding or persistent weight expansion.
struct Q4G16ActivationTile {
  float scales[16];
  std::int16_t sums[2][16];
  std::int8_t qs[16][32];
};
static_assert(sizeof(Q4G16ActivationTile)==640);
void q4_g16_pack_activations(const Q8_0BlockX1*,Q4G16ActivationTile*,
                            std::size_t count,std::size_t blocks) noexcept;
void q4_g16_matmul(const Q4G16Tile*,const Q4G16ActivationTile*,float*,
                  std::size_t rows,std::size_t count,std::size_t blocks,
                  std::size_t output_stride,Q8_0Backend backend) noexcept;
}
