#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace qwen35x::cpu {

inline constexpr std::size_t q8_0_values_per_block = 32;

// Binary-compatible with GGML's block_q8_0: one IEEE-754 binary16 scale
// followed by 32 signed quants. In particular, there is no tail padding.
struct Q8_0Block {
  std::uint16_t d;
  std::int8_t qs[q8_0_values_per_block];
};

static_assert(std::is_standard_layout_v<Q8_0Block>);
static_assert(offsetof(Q8_0Block, d) == 0);
static_assert(offsetof(Q8_0Block, qs) == sizeof(std::uint16_t));
static_assert(sizeof(Q8_0Block) == 34);

enum class Q8_0Backend : std::uint8_t {
  auto_select = 0,
  scalar = 1,
  avx2 = 2,
  avx_vnni = 3,
  avx512 = 4,
  avx512_vnni = 5,
};

// Explicit ISA requests remain safe on unsupported CPUs and resolve to the
// next supported implementation. Call q8_0_backend_available() when a hard
// requirement (for example, an ISA-specific test) must be enforced.
[[nodiscard]] bool q8_0_backend_available(Q8_0Backend backend) noexcept;
// Usable capabilities include compiler support and enabled OS register state.
// These flags are independent: EVEX VNNI does not imply VEX AVX-VNNI.
struct CpuCapabilities {
  bool avx2 = false;
  bool avx_vnni = false;
  bool avx512 = false;
  bool avx512_vnni = false;
};
[[nodiscard]] CpuCapabilities cpu_capabilities() noexcept;
[[nodiscard]] Q8_0Backend q8_0_resolve_backend_for_capabilities(
  Q8_0Backend requested, CpuCapabilities capabilities) noexcept;
[[nodiscard]] Q8_0Backend q8_0_dot_backend_for_capabilities(
  Q8_0Backend requested, CpuCapabilities capabilities) noexcept;
[[nodiscard]] bool q8_0_backend_uses_avx_vnni(Q8_0Backend backend) noexcept;

[[nodiscard]] Q8_0Backend q8_0_resolve_backend(Q8_0Backend requested) noexcept;
[[nodiscard]] const char * q8_0_backend_name(Q8_0Backend backend) noexcept;

// AVX-VNNI specializes integer Q8 and packed Q4 projection dots but reuses the
// AVX2 implementation for FP32 activation, attention, and DeltaNet kernels.
[[nodiscard]] bool q8_0_backend_uses_avx2(Q8_0Backend backend) noexcept;

// FP32 kernels can use AVX-512 independently of the integer-dot selection.
// Q8 dots use VEX VNNI when available, otherwise AVX2; Q4 also supports EVEX.
[[nodiscard]] bool q8_0_backend_uses_avx512(Q8_0Backend backend) noexcept;

// Each operation works on complete 32-value Q8_0 blocks. Source and destination
// ranges must not overlap. A block_count of zero is valid.
void q8_0_quantize(
  const float * input,
  Q8_0Block * output,
  std::size_t block_count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// Quantizes blocks and emits the exact binary16-rounded block scale as F32 in
// the same pass. This avoids rereading Q8_0 blocks before batched matrix
// multiplication. scales contains block_count elements.
void q8_0_quantize_with_scales(
  const float * input,
  Q8_0Block * output,
  float * scales,
  std::size_t block_count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

void q8_0_dequantize(
  const Q8_0Block * input,
  float * output,
  std::size_t block_count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// Expands only the binary16 scale of each block. This is useful for batched
// kernels where activation scales are reused across thousands of weight rows.
void q8_0_scales_to_f32(
  const Q8_0Block * input,
  float * output,
  std::size_t block_count) noexcept;

[[nodiscard]] float q8_0_dot(
  const Q8_0Block * lhs,
  const Q8_0Block * rhs,
  std::size_t block_count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// matrix is row-major with blocks_per_row consecutive Q8_0 blocks per row.
// vector contains blocks_per_row blocks and output contains row_count floats.
void q8_0_matvec(
  const Q8_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// Batched variant for vector_count independently quantized input rows.
// vectors and optional vector_scales use token-major layout
// [vector_count, blocks_per_row]. Output is token-major:
// output[vector * row_count + row].
void q8_0_matmul(
  const Q8_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  std::size_t row_count,
  std::size_t vector_count,
  std::size_t blocks_per_row,
  std::size_t output_row_stride,
  Q8_0Backend backend = Q8_0Backend::auto_select,
  const float * vector_scales = nullptr,
  const float * matrix_scales = nullptr) noexcept;

} // namespace qwen35x::cpu
