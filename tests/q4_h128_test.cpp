#include "qwen35x/cpu/q4_h128.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

bool expect(const bool condition, const char * message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

bool near(const float lhs, const float rhs, const float relative = 2.0e-5F) {
  const float tolerance = relative * std::max({1.0F, std::fabs(lhs), std::fabs(rhs)});
  return std::fabs(lhs - rhs) <= tolerance;
}

std::vector<float> make_values(const std::size_t count, const float phase) {
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    values[index] = std::sin(static_cast<float>(index) * 0.137F + phase) * 2.75F +
      std::cos(static_cast<float>(index) * 0.047F - phase) * 0.625F;
  }
  return values;
}

float dot(const float * lhs, const float * rhs, const std::size_t count) {
  float result = 0.0F;
  for (std::size_t index = 0; index < count; ++index) {
    result += lhs[index] * rhs[index];
  }
  return result;
}

bool test_orthogonality() {
  const std::vector<float> lhs = make_values(256, 0.31F);
  const std::vector<float> rhs = make_values(256, -0.73F);
  std::vector<float> transformed_lhs(256);
  std::vector<float> transformed_rhs(256);
  bool ok = true;
  ok = expect(qwen35x::cpu::q4_h128_transform_rows(
                lhs.data(), transformed_lhs.data(), 1, 256),
              "lhs transform rejected valid dimensions") && ok;
  ok = expect(qwen35x::cpu::q4_h128_transform_rows(
                rhs.data(), transformed_rhs.data(), 1, 256),
              "rhs transform rejected valid dimensions") && ok;
  ok = expect(near(dot(lhs.data(), rhs.data(), 256),
                   dot(transformed_lhs.data(), transformed_rhs.data(), 256)),
              "orthogonal transform changed dot product") && ok;
  ok = expect(near(dot(lhs.data(), lhs.data(), 256),
                   dot(transformed_lhs.data(), transformed_lhs.data(), 256)),
              "orthogonal transform changed norm") && ok;

  std::vector<float> repeated(256);
  ok = expect(qwen35x::cpu::q4_h128_transform_rows(
                lhs.data(), repeated.data(), 1, 256),
              "repeat transform failed") && ok;
  ok = expect(repeated == transformed_lhs, "transform is not deterministic") && ok;
  ok = expect(!std::equal(transformed_lhs.begin(), transformed_lhs.begin() + 128,
                          transformed_lhs.begin() + 128),
              "different input blocks reused an identical transform result") && ok;
  return ok;
}

bool test_rows_and_rejection() {
  const std::vector<float> input = make_values(3 * 256, 0.13F);
  std::vector<float> rows(input.size());
  std::vector<float> block(128);
  bool ok = expect(qwen35x::cpu::q4_h128_transform_rows(
                     input.data(), rows.data(), 3, 256),
                   "multi-row transform failed");
  qwen35x::cpu::q4_h128_transform_block(input.data() + 256, block.data(), 0);
  ok = expect(std::equal(block.begin(), block.end(), rows.begin() + 256),
              "transform block index did not restart for each row") && ok;
  ok = expect(!qwen35x::cpu::q4_h128_transform_rows(
                input.data(), rows.data(), 1, 192),
              "invalid transform width was accepted") && ok;
  ok = expect(!qwen35x::cpu::q4_h128_transform_rows(
                nullptr, rows.data(), 1, 128),
              "null transform input was accepted") && ok;
  return ok;
}

bool test_avx2_parity() {
  if (!qwen35x::cpu::q8_0_backend_available(
        qwen35x::cpu::Q8_0Backend::avx2)) {
    return true;
  }
  const std::vector<float> input = make_values(4 * 256, -0.41F);
  std::vector<float> scalar(input.size());
  std::vector<float> avx2(input.size());
  bool ok = expect(qwen35x::cpu::q4_h128_transform_rows(
                     input.data(), scalar.data(), 4, 256,
                     qwen35x::cpu::q4_h128_default_sign_seed,
                     qwen35x::cpu::Q8_0Backend::scalar),
                   "scalar parity transform failed");
  ok = expect(qwen35x::cpu::q4_h128_transform_rows(
                input.data(), avx2.data(), 4, 256,
                qwen35x::cpu::q4_h128_default_sign_seed,
                qwen35x::cpu::Q8_0Backend::avx2),
              "AVX2 parity transform failed") && ok;
  ok = expect(scalar == avx2, "AVX2 transform differs from scalar ABI") && ok;
  return ok;
}

bool test_unscaled_q8_preparation() {
  constexpr std::size_t columns = 3584;
  const std::vector<float> input = make_values(columns, 0.77F);
  std::vector<float> normalized(columns);
  std::vector<float> unscaled(columns);
  const std::size_t block_count = columns / qwen35x::cpu::q8_0_values_per_block;
  std::vector<qwen35x::cpu::Q8_0BlockX1> reference(block_count);
  std::vector<qwen35x::cpu::Q8_0BlockX1> folded(block_count);
  bool ok = expect(qwen35x::cpu::q4_h128_transform_rows(
                     input.data(), normalized.data(), 1, columns,
                     qwen35x::cpu::q4_h128_default_sign_seed,
                     qwen35x::cpu::Q8_0Backend::avx2),
                   "normalized preparation transform failed");
  ok = expect(qwen35x::cpu::q4_h128_transform_rows_unscaled(
                input.data(), unscaled.data(), 1, columns,
                qwen35x::cpu::q4_h128_default_sign_seed,
                qwen35x::cpu::Q8_0Backend::avx2),
              "unscaled preparation transform failed") && ok;
  qwen35x::cpu::q8_0_quantize_vector_1(
    normalized.data(), reference.data(), block_count,
    qwen35x::cpu::Q8_0Backend::avx2);
  qwen35x::cpu::q8_0_quantize_vector_1(
    unscaled.data(), folded.data(), block_count,
    qwen35x::cpu::Q8_0Backend::avx2);
  for (std::size_t block = 0; block < block_count; ++block) {
    folded[block].scales[0] *= qwen35x::cpu::q4_h128_inverse_sqrt_size;
    ok = expect(reference[block].sums[0] == folded[block].sums[0] &&
                  std::memcmp(reference[block].qs, folded[block].qs, 32) == 0,
                "folded normalization changed Q8 integers") && ok;
    ok = expect(near(reference[block].scales[0], folded[block].scales[0], 6.0e-4F),
                "folded normalization changed Q8 scale excessively") && ok;
  }
  return ok;
}

bool test_quantized_projection() {
  constexpr std::size_t rows = 8;
  constexpr std::size_t columns = 256;
  const std::vector<float> weights = make_values(rows * columns, 0.53F);
  const std::vector<float> activation = make_values(columns, -0.29F);
  std::vector<qwen35x::cpu::Q4_0Block> quantized(
    rows * columns / qwen35x::cpu::q4_0_values_per_block);
  std::vector<float> scratch(columns);
  std::vector<qwen35x::cpu::Q8_0Block> prepared(
    columns / qwen35x::cpu::q8_0_values_per_block);
  bool ok = expect(qwen35x::cpu::q4_h128_quantize_matrix(
                     weights.data(), quantized.data(), rows, columns),
                   "matrix quantization failed");
  ok = expect(qwen35x::cpu::q4_h128_prepare_activation(
                activation.data(), scratch.data(), prepared.data(), columns,
                qwen35x::cpu::q4_h128_default_sign_seed,
                qwen35x::cpu::Q8_0Backend::scalar),
              "activation preparation failed") && ok;

  std::vector<float> output(rows);
  qwen35x::cpu::q4_0_matvec_q8_0(
    quantized.data(), prepared.data(), output.data(), rows,
    columns / qwen35x::cpu::q4_0_values_per_block,
    qwen35x::cpu::Q8_0Backend::scalar);
  for (std::size_t row = 0; row < rows; ++row) {
    const float reference = dot(weights.data() + row * columns, activation.data(), columns);
    ok = expect(std::isfinite(output[row]), "quantized projection is not finite") && ok;
    ok = expect(std::fabs(output[row] - reference) <=
                  0.12F * std::max(1.0F, std::fabs(reference)),
                "quantized projection error exceeded smoke threshold") && ok;
  }

  std::vector<float> transformed(columns);
  ok = expect(qwen35x::cpu::q4_h128_transform_rows(
                activation.data(), transformed.data(), 1, columns),
              "comparison transform failed") && ok;
  ok = expect(std::equal(transformed.begin(), transformed.end(), scratch.begin()),
              "prepared activation used a different transform") && ok;
  return ok;
}

bool test_fused_packed_activation() {
  using namespace qwen35x::cpu;
  bool ok = true;
  for (auto backend : {Q8_0Backend::scalar, Q8_0Backend::avx2,
                       Q8_0Backend::avx512_vnni}) {
    for (std::size_t columns : {128U, 256U, 3584U}) {
      for (std::uint64_t seed : {UINT64_C(0), q4_h128_default_sign_seed}) {
        constexpr std::size_t vectors = 8;
        const auto input = make_values(vectors * columns, 0.77F);
        std::vector<float> transformed(input.size());
        std::vector<Q8_0BlockX4> reference((vectors / 4) * (columns / 32));
        std::vector<Q8_0BlockX4> fused(reference.size());
        std::vector<Q4H128SignBlock> signs(columns / 128);
        q4_h128_prepare_signs(signs.data(), signs.size(), seed);
        ok = expect(q4_h128_transform_rows_unscaled(input.data(), transformed.data(),
                      vectors, columns, seed, backend), "packed reference transform failed") && ok;
        q8_0_quantize_vectors_4(transformed.data(), reference.data(), vectors, columns / 32, backend);
        for (auto & block : reference) {
          for (float & scale : block.scales) scale *= q4_h128_inverse_sqrt_size;
        }
        for (bool cached : {false, true}) {
          ok = expect(q4_h128_prepare_activations_4(input.data(), fused.data(), vectors,
                        columns, seed, backend, cached ? signs.data() : nullptr),
                      "fused packed preparation failed") && ok;
          ok = expect(std::equal(
                        reinterpret_cast<const unsigned char *>(reference.data()),
                        reinterpret_cast<const unsigned char *>(reference.data() + reference.size()),
                        reinterpret_cast<const unsigned char *>(fused.data())),
                      "fused packed differs from two-pass preparation") && ok;
        }
        ok = expect(!q4_h128_prepare_activations_4(input.data(), fused.data(), 6, columns),
                    "invalid packed vector count accepted") && ok;
      }
    }
  }
  return ok;
}

bool test_fused_decode_activation() {
  using namespace qwen35x::cpu;
  bool ok = true;
  for (auto backend : {Q8_0Backend::scalar, Q8_0Backend::avx2,
                       Q8_0Backend::avx_vnni, Q8_0Backend::avx512_vnni}) {
    for (std::size_t columns : {128U, 256U, 1024U, 3584U}) {
      for (std::uint64_t seed : {UINT64_C(0), q4_h128_default_sign_seed}) {
        for (bool zero : {false, true}) {
          auto input = make_values(columns, 0.31F);
          if (zero) std::fill(input.begin(), input.end(), 0.0F);
          std::vector<float> transformed(columns);
          std::vector<Q8_0BlockX1> reference(columns / 32), fused(columns / 32);
          std::vector<Q4H128SignBlock> signs(columns / 128);
          q4_h128_prepare_signs(signs.data(), signs.size(), seed);
          ok = expect(q4_h128_transform_rows_unscaled(input.data(), transformed.data(),
                        1, columns, seed, backend), "decode reference transform failed") && ok;
          q8_0_quantize_vector_1(transformed.data(), reference.data(), columns / 32, backend);
          for (auto & block : reference) block.scales[0] *= q4_h128_inverse_sqrt_size;
          for (bool cached : {false, true}) {
            ok = expect(q4_h128_prepare_activation_1(input.data(), fused.data(), columns,
                          seed, backend, cached ? signs.data() : nullptr), "fused decode preparation failed") && ok;
            for (std::size_t b = 0; b < reference.size(); ++b) {
              ok = expect(reference[b].scales[0] == fused[b].scales[0] &&
                            reference[b].sums[0] == fused[b].sums[0] &&
                            std::equal(reference[b].qs, reference[b].qs + 32, fused[b].qs),
                          "fused decode differs from two-pass preparation") && ok;
            }
          }
          ok = expect(!q4_h128_prepare_activation_1(input.data(), fused.data(), columns - 1,
                        seed, backend), "invalid decode width accepted") && ok;
        }
      }
    }
  }
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok = test_orthogonality() && ok;
  ok = test_rows_and_rejection() && ok;
  ok = test_avx2_parity() && ok;
  ok = test_unscaled_q8_preparation() && ok;
  ok = test_quantized_projection() && ok;
  ok = test_fused_packed_activation() && ok;
  ok = test_fused_decode_activation() && ok;
  if (ok) {
    std::cout << "Q4_H128 tests passed\n";
    return 0;
  }
  return 1;
}
