#include "qwen35x/cpu/activation.h"
#include "qwen35x/cpu/full_attention.h"
#include "qwen35x/cpu/q8_0.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using qwen35x::cpu::Q8_0Backend;
using qwen35x::cpu::Q8_0Block;
using qwen35x::cpu::q8_0_values_per_block;

bool expect(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << "q8_0 test failure: " << message << '\n';
  }
  return condition;
}

bool near(const float lhs, const float rhs) {
  const float tolerance = 1.0e-4F + 2.0e-5F * std::max(std::fabs(lhs), std::fabs(rhs));
  return std::fabs(lhs - rhs) <= tolerance;
}

bool near_attention(const float lhs, const float rhs) {
  const float tolerance = 5.0e-4F + 2.0e-4F * std::max(std::fabs(lhs), std::fabs(rhs));
  return std::fabs(lhs - rhs) <= tolerance;
}

std::vector<float> make_values(const std::size_t block_count, const float phase) {
  std::vector<float> values(block_count * q8_0_values_per_block);
  for (std::size_t index = 0; index < values.size(); ++index) {
    const float x = static_cast<float>(index) + phase;
    values[index] = 3.75F * std::sin(x * 0.37F) + 0.625F * std::cos(x * 0.11F);
  }
  if (!values.empty()) {
    values[0] = 0.0F;
  }
  return values;
}

bool test_known_layout() {
  std::array<float, q8_0_values_per_block> input{};
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(static_cast<int>(index) - 16);
  }
  input.back() = 127.0F;

  Q8_0Block block{};
  qwen35x::cpu::q8_0_quantize(input.data(), &block, 1, Q8_0Backend::scalar);
  bool ok = expect(block.d == 0x3c00U, "GGML binary16 scale must encode 1.0") &&
    expect(block.qs[0] == -16, "first quant must begin immediately after scale") &&
    expect(block.qs[16] == 0, "zero quant mismatch") &&
    expect(block.qs[31] == 127, "positive Q8_0 endpoint mismatch");

  std::array<float, q8_0_values_per_block> output{};
  qwen35x::cpu::q8_0_dequantize(&block, output.data(), 1, Q8_0Backend::scalar);
  for (std::size_t index = 0; index < output.size(); ++index) {
    ok = expect(output[index] == static_cast<float>(block.qs[index]), "known block dequant mismatch") && ok;
  }
  return ok;
}

bool test_zero_block() {
  std::array<float, q8_0_values_per_block> input{};
  Q8_0Block block{};
  std::memset(&block, 0x55, sizeof(block));
  qwen35x::cpu::q8_0_quantize(input.data(), &block, 1, Q8_0Backend::scalar);
  bool ok = expect(block.d == 0, "zero block scale must be zero");
  for (const std::int8_t value : block.qs) {
    ok = expect(value == 0, "zero block quant must be zero") && ok;
  }
  return ok;
}

bool test_empty_ranges() {
  qwen35x::cpu::q8_0_quantize(nullptr, nullptr, 0, Q8_0Backend::auto_select);
  qwen35x::cpu::q8_0_quantize_with_scales(
    nullptr, nullptr, nullptr, 0, Q8_0Backend::auto_select);
  qwen35x::cpu::q8_0_dequantize(nullptr, nullptr, 0, Q8_0Backend::auto_select);
  bool ok = expect(
    qwen35x::cpu::q8_0_dot(nullptr, nullptr, 0, Q8_0Backend::auto_select) == 0.0F,
    "empty dot product must be zero");
  float output = 1.0F;
  qwen35x::cpu::q8_0_matvec(nullptr, nullptr, &output, 1, 0, Q8_0Backend::auto_select);
  return expect(output == 0.0F, "zero-width matvec must produce zero") && ok;
}

bool test_silu_mul(const Q8_0Backend backend) {
  constexpr std::size_t count = 19;
  std::array<float, count> gate{};
  std::array<float, count> up{};
  for (std::size_t index = 0; index < count; ++index) {
    gate[index] = (static_cast<float>(index) - 9.0F) * 1.375F;
    up[index] = std::sin(static_cast<float>(index) * 0.7F);
  }
  std::array<float, count> expected{};
  std::array<float, count> actual{};
  std::array<float, count> expected_silu{};
  std::array<float, count> actual_silu{};
  qwen35x::cpu::silu_f32(
    gate.data(), expected_silu.data(), count, Q8_0Backend::scalar);
  qwen35x::cpu::silu_f32(
    gate.data(), actual_silu.data(), count, backend);
  qwen35x::cpu::silu_mul_f32(
    gate.data(), up.data(), expected.data(), count, Q8_0Backend::scalar);
  qwen35x::cpu::silu_mul_f32(
    gate.data(), up.data(), actual.data(), count, backend);
  bool ok = true;
  for (std::size_t index = 0; index < count; ++index) {
    ok = expect(near(expected_silu[index], actual_silu[index]), "SiLU mismatch") && ok;
    ok = expect(near(expected[index], actual[index]), "SiLU multiply mismatch") && ok;
  }
  return ok;
}

bool test_rms_norm(const Q8_0Backend backend) {
  constexpr std::size_t rows = 3;
  constexpr std::size_t width = 35;
  std::array<float, rows * width> input{};
  std::array<float, width> weight{};
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = std::sin(static_cast<float>(index) * 0.17F) * 3.0F;
  }
  for (std::size_t index = 0; index < weight.size(); ++index) {
    weight[index] = std::cos(static_cast<float>(index) * 0.11F) * 0.25F;
  }
  std::array<float, rows * width> expected{};
  std::array<float, rows * width> actual{};
  qwen35x::cpu::rms_norm_f32(
    input.data(), weight.data(), expected.data(), rows, width, 1.0e-6F, 1.0F,
    Q8_0Backend::scalar);
  qwen35x::cpu::rms_norm_f32(
    input.data(), weight.data(), actual.data(), rows, width, 1.0e-6F, 1.0F,
    backend);
  bool ok = true;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    ok = expect(near(expected[index], actual[index]), "RMS norm mismatch") && ok;
  }
  return ok;
}

bool test_add(const Q8_0Backend backend) {
  constexpr std::size_t count = 39;
  std::array<float, count> lhs{};
  std::array<float, count> rhs{};
  std::array<float, count> output{};
  for (std::size_t index = 0; index < count; ++index) {
    lhs[index] = static_cast<float>(index) * 0.25F;
    rhs[index] = std::sin(static_cast<float>(index));
  }
  qwen35x::cpu::add_f32(lhs.data(), rhs.data(), output.data(), count, backend);
  bool ok = true;
  for (std::size_t index = 0; index < count; ++index) {
    ok = expect(output[index] == lhs[index] + rhs[index], "vector add mismatch") && ok;
  }
  return ok;
}

bool test_rope(const Q8_0Backend backend) {
  constexpr std::size_t heads = 3;
  constexpr std::size_t head_dim = 23;
  constexpr std::size_t rope_dim = 18;
  constexpr std::size_t half = rope_dim / 2;
  std::array<float, heads * head_dim> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    expected[index] = std::sin(static_cast<float>(index) * 0.19F) * 2.0F;
  }
  auto actual = expected;
  std::array<float, half> cosine{};
  std::array<float, half> sine{};
  for (std::size_t index = 0; index < half; ++index) {
    const float angle = static_cast<float>(index + 1) * 0.13F;
    cosine[index] = std::cos(angle);
    sine[index] = std::sin(angle);
  }
  qwen35x::cpu::rope_f32(
    expected.data(), heads, head_dim, rope_dim,
    cosine.data(), sine.data(), Q8_0Backend::scalar);
  qwen35x::cpu::rope_f32(
    actual.data(), heads, head_dim, rope_dim,
    cosine.data(), sine.data(), backend);
  bool ok = true;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    ok = expect(near(expected[index], actual[index]), "RoPE mismatch") && ok;
  }
  return ok;
}

bool test_causal_conv1d_silu(const Q8_0Backend backend) {
  constexpr std::size_t batch = 5;
  constexpr std::size_t channels = 19;
  constexpr std::size_t input_stride = 23;
  constexpr std::size_t kernel = 4;
  constexpr std::size_t history = kernel - 1;
  constexpr std::size_t ring_index = 2;
  std::array<float, batch * input_stride> input{};
  std::array<float, kernel * channels> weights{};
  std::array<float, history * channels> expected_state{};
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = std::sin(static_cast<float>(index) * 0.07F) * 1.5F;
  }
  for (std::size_t index = 0; index < weights.size(); ++index) {
    weights[index] = std::cos(static_cast<float>(index) * 0.09F) * 0.25F;
  }
  for (std::size_t index = 0; index < expected_state.size(); ++index) {
    expected_state[index] = std::sin(static_cast<float>(index) * 0.05F);
  }
  auto actual_state = expected_state;
  std::array<float, batch * channels> expected{};
  std::array<float, batch * channels> actual{};
  qwen35x::cpu::causal_conv1d_silu_f32(
    expected_state.data(), ring_index, input.data(), input_stride,
    weights.data(), expected.data(), batch, channels, kernel, 0, channels,
    Q8_0Backend::scalar);
  qwen35x::cpu::causal_conv1d_silu_f32(
    actual_state.data(), ring_index, input.data(), input_stride,
    weights.data(), actual.data(), batch, channels, kernel, 0, channels,
    backend);
  bool ok = true;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    ok = expect(near(expected[index], actual[index]), "causal convolution output mismatch") && ok;
  }
  for (std::size_t index = 0; index < actual_state.size(); ++index) {
    ok = expect(
      expected_state[index] == actual_state[index],
      "causal convolution ring state mismatch") && ok;
  }
  return ok;
}

bool test_full_attention(const Q8_0Backend backend) {
  constexpr int heads = 2;
  constexpr int kv_heads = 1;
  constexpr int head_dim = 17;
  constexpr std::array<int, 11> contexts{
    1, 7, 8, 9, 63, 64, 65, 255, 256, 257, 2048,
  };
  bool ok = true;
  for (const int context : contexts) {
    const std::size_t query_width = static_cast<std::size_t>(heads * head_dim);
    const std::size_t kv_width = static_cast<std::size_t>(kv_heads * head_dim);
    std::vector<float> query(query_width);
    std::vector<float> gate(query_width);
    std::vector<float> key_cache(static_cast<std::size_t>(context) * kv_width);
    std::vector<float> value_cache(static_cast<std::size_t>(context) * kv_width);
    for (std::size_t index = 0; index < query.size(); ++index) {
      query[index] = std::sin(static_cast<float>(index) * 0.17F);
      gate[index] = std::cos(static_cast<float>(index) * 0.23F) * 3.0F;
    }
    for (std::size_t index = 0; index < key_cache.size(); ++index) {
      key_cache[index] = std::cos(static_cast<float>(index) * 0.013F);
      value_cache[index] = std::sin(static_cast<float>(index) * 0.019F);
    }
    std::vector<float> scalar_scores(static_cast<std::size_t>(heads * context));
    std::vector<float> test_scores(static_cast<std::size_t>(heads * context));
    std::vector<float> expected(query_width);
    std::vector<float> actual(query_width);
    qwen35x::cpu::causal_attention_batch_rows(
      query.data(), gate.data(), key_cache.data(), value_cache.data(), nullptr,
      nullptr, scalar_scores.data(), expected.data(), static_cast<std::size_t>(context),
      query_width, kv_width, context - 1, heads, kv_heads, head_dim,
      1.0F / std::sqrt(static_cast<float>(head_dim)), 0,
      static_cast<std::size_t>(heads), Q8_0Backend::scalar);
    qwen35x::cpu::causal_attention_batch_rows(
      query.data(), gate.data(), key_cache.data(), value_cache.data(), nullptr,
      nullptr, test_scores.data(), actual.data(), static_cast<std::size_t>(context),
      query_width, kv_width, context - 1, heads, kv_heads, head_dim,
      1.0F / std::sqrt(static_cast<float>(head_dim)), 0,
      static_cast<std::size_t>(heads), backend);
    std::vector<float> compact_scores(static_cast<std::size_t>(context) + 2, -12345.0F);
    std::vector<float> compact_output(query_width, 0.0F);
    for (const std::size_t begin : {std::size_t{0}, std::size_t{1}}) {
      qwen35x::cpu::causal_attention_batch_rows(
        query.data(), gate.data(), key_cache.data(), value_cache.data(), nullptr,
        nullptr, compact_scores.data() + 1, compact_output.data(),
        static_cast<std::size_t>(context), query_width, kv_width, context - 1,
        heads, kv_heads, head_dim, 1.0F / std::sqrt(static_cast<float>(head_dim)),
        begin, static_cast<std::size_t>(heads), backend, true);
    }
    ok = expect(std::memcmp(actual.data(), compact_output.data(), query_width * sizeof(float)) == 0,
      "private score-row reuse changed attention output") && ok;
    ok = expect(compact_scores.front() == -12345.0F && compact_scores.back() == -12345.0F,
      "private score-row scratch overrun") && ok;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      ok = expect(
        near_attention(expected[index], actual[index]),
        "full-attention output mismatch") && ok;
    }
  }
  return ok;
}

bool test_full_attention_decode_pairs(const Q8_0Backend backend) {
  constexpr int heads = 8;
  constexpr int kv_heads = 2;
  constexpr std::array<int, 2> head_dims{17, 256};
  constexpr std::array<int, 11> contexts{
    1, 7, 8, 9, 63, 64, 65, 255, 256, 257, 2048,
  };
  bool ok = true;
  for (const int head_dim : head_dims) {
    for (const int context : contexts) {
      const std::size_t query_width = static_cast<std::size_t>(heads * head_dim);
      const std::size_t kv_width = static_cast<std::size_t>(kv_heads * head_dim);
      std::vector<float> query(query_width);
      std::vector<float> gate(query_width);
      std::vector<float> key_cache(static_cast<std::size_t>(context) * kv_width);
      std::vector<float> value_cache(static_cast<std::size_t>(context) * kv_width);
      for (std::size_t index = 0; index < query.size(); ++index) {
        query[index] = std::sin(static_cast<float>(index) * 0.017F);
        gate[index] = std::cos(static_cast<float>(index) * 0.023F) * 3.0F;
      }
      for (std::size_t index = 0; index < key_cache.size(); ++index) {
        key_cache[index] = std::cos(static_cast<float>(index) * 0.0013F);
        value_cache[index] = std::sin(static_cast<float>(index) * 0.0019F);
      }
      std::vector<std::uint16_t> key_cache_f16(key_cache.size());
      std::vector<std::uint16_t> value_cache_f16(value_cache.size());
      qwen35x::cpu::attention_cache_store_f16(
        key_cache.data(), key_cache_f16.data(), key_cache.size(), backend);
      qwen35x::cpu::attention_cache_store_f16(
        value_cache.data(), value_cache_f16.data(), value_cache.size(), backend);
      std::vector<float> reference_scores(static_cast<std::size_t>(heads * context));
      std::vector<float> paired_scores(static_cast<std::size_t>(heads * context));
      std::vector<float> reference(query_width);
      std::vector<float> paired(query_width);
      const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
      for (const bool use_f16 : {false, true}) {
        qwen35x::cpu::causal_attention_batch_rows(
          query.data(), gate.data(),
          use_f16 ? nullptr : key_cache.data(),
          use_f16 ? nullptr : value_cache.data(),
          use_f16 ? key_cache_f16.data() : nullptr,
          use_f16 ? value_cache_f16.data() : nullptr, reference_scores.data(),
          reference.data(), static_cast<std::size_t>(context), query_width, kv_width,
          context - 1, heads, kv_heads, head_dim, scale, 0,
          static_cast<std::size_t>(heads), backend);
        qwen35x::cpu::causal_attention_decode_gqa_pairs(
          query.data(), gate.data(),
          use_f16 ? nullptr : key_cache.data(),
          use_f16 ? nullptr : value_cache.data(),
          use_f16 ? key_cache_f16.data() : nullptr,
          use_f16 ? value_cache_f16.data() : nullptr, paired_scores.data(),
          paired.data(), static_cast<std::size_t>(context), query_width, kv_width,
          context, heads, kv_heads, head_dim, scale, 0,
          static_cast<std::size_t>(heads / 2), backend);
        for (std::size_t index = 0; index < paired.size(); ++index) {
          ok = expect(
            near_attention(reference[index], paired[index]),
            "paired GQA decode output differs from row kernel") && ok;
        }
      }
    }
  }
  return ok;
}

bool test_l2_normalize(const Q8_0Backend backend) {
  constexpr std::size_t rows = 3;
  constexpr std::size_t width = 37;
  std::array<float, rows * width> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    expected[index] = std::cos(static_cast<float>(index) * 0.13F) * 2.0F;
  }
  auto actual = expected;
  qwen35x::cpu::l2_normalize_f32(
    expected.data(), rows, width, 1.0e-6F, 0.125F, Q8_0Backend::scalar);
  qwen35x::cpu::l2_normalize_f32(
    actual.data(), rows, width, 1.0e-6F, 0.125F, backend);
  bool ok = true;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    ok = expect(near(expected[index], actual[index]), "L2 normalization mismatch") && ok;
  }
  return ok;
}

bool test_backend(const Q8_0Backend backend) {
  constexpr std::size_t blocks = 5;
  constexpr std::size_t rows = 4;
  constexpr std::size_t vectors = 13;
  const std::vector<float> lhs_values = make_values(blocks * rows, 0.25F);
  const std::vector<float> rhs_values = make_values(blocks, 1.75F);
  std::vector<Q8_0Block> lhs_scalar(blocks * rows);
  std::vector<Q8_0Block> rhs_scalar(blocks);
  std::vector<Q8_0Block> lhs_test(blocks * rows);
  std::vector<Q8_0Block> rhs_test(blocks);
  qwen35x::cpu::q8_0_quantize(lhs_values.data(), lhs_scalar.data(), lhs_scalar.size(), Q8_0Backend::scalar);
  qwen35x::cpu::q8_0_quantize(rhs_values.data(), rhs_scalar.data(), rhs_scalar.size(), Q8_0Backend::scalar);
  qwen35x::cpu::q8_0_quantize(lhs_values.data(), lhs_test.data(), lhs_test.size(), backend);
  qwen35x::cpu::q8_0_quantize(rhs_values.data(), rhs_test.data(), rhs_test.size(), backend);

  std::vector<Q8_0Block> lhs_with_scales(lhs_test.size());
  std::vector<float> fused_scales(lhs_test.size());
  std::vector<float> expanded_scales(lhs_test.size());
  qwen35x::cpu::q8_0_quantize_with_scales(
    lhs_values.data(),
    lhs_with_scales.data(),
    fused_scales.data(),
    lhs_with_scales.size(),
    backend);
  qwen35x::cpu::q8_0_scales_to_f32(
    lhs_with_scales.data(), expanded_scales.data(), lhs_with_scales.size());

  bool ok = expect(
    std::memcmp(lhs_scalar.data(), lhs_test.data(), lhs_scalar.size() * sizeof(Q8_0Block)) == 0,
    "quantized matrix differs from deterministic scalar Q8_0") &&
    expect(
      std::memcmp(rhs_scalar.data(), rhs_test.data(), rhs_scalar.size() * sizeof(Q8_0Block)) == 0,
      "quantized vector differs from deterministic scalar Q8_0") &&
    expect(
      std::memcmp(lhs_test.data(), lhs_with_scales.data(), lhs_test.size() * sizeof(Q8_0Block)) == 0,
      "quantize-with-scales changed Q8_0 output") &&
    expect(
      std::memcmp(fused_scales.data(), expanded_scales.data(), fused_scales.size() * sizeof(float)) == 0,
      "fused scale sidecar differs from expanded binary16 scales");

  std::vector<float> dequant_scalar(rhs_values.size());
  std::vector<float> dequant_test(rhs_values.size());
  qwen35x::cpu::q8_0_dequantize(rhs_scalar.data(), dequant_scalar.data(), blocks, Q8_0Backend::scalar);
  qwen35x::cpu::q8_0_dequantize(rhs_test.data(), dequant_test.data(), blocks, backend);
  for (std::size_t index = 0; index < dequant_scalar.size(); ++index) {
    ok = expect(dequant_scalar[index] == dequant_test[index], "dequantized value mismatch") && ok;
  }

  const float scalar_dot = qwen35x::cpu::q8_0_dot(lhs_scalar.data(), rhs_scalar.data(), blocks, Q8_0Backend::scalar);
  const float test_dot = qwen35x::cpu::q8_0_dot(lhs_test.data(), rhs_test.data(), blocks, backend);
  ok = expect(near(scalar_dot, test_dot), "dot product mismatch") && ok;

  std::vector<float> dequant_lhs(blocks * q8_0_values_per_block);
  qwen35x::cpu::q8_0_dequantize(
    lhs_scalar.data(), dequant_lhs.data(), blocks, Q8_0Backend::scalar);
  float reference_dot = 0.0F;
  for (std::size_t index = 0; index < dequant_lhs.size(); ++index) {
    reference_dot += dequant_lhs[index] * dequant_scalar[index];
  }
  ok = expect(near(scalar_dot, reference_dot), "scalar dot differs from dequantized reference") && ok;

  std::array<float, rows> scalar_output{};
  std::array<float, rows> test_output{};
  qwen35x::cpu::q8_0_matvec(
    lhs_scalar.data(), rhs_scalar.data(), scalar_output.data(), rows, blocks, Q8_0Backend::scalar);
  qwen35x::cpu::q8_0_matvec(
    lhs_test.data(), rhs_test.data(), test_output.data(), rows, blocks, backend);
  for (std::size_t row = 0; row < rows; ++row) {
    ok = expect(near(scalar_output[row], test_output[row]), "matvec row mismatch") && ok;
  }

  const std::vector<float> batch_values = make_values(blocks * vectors, 3.25F);
  std::vector<Q8_0Block> batch_scalar(blocks * vectors);
  std::vector<Q8_0Block> batch_test(blocks * vectors);
  qwen35x::cpu::q8_0_quantize(
    batch_values.data(), batch_scalar.data(), batch_scalar.size(), Q8_0Backend::scalar);
  qwen35x::cpu::q8_0_quantize(
    batch_values.data(), batch_test.data(), batch_test.size(), backend);
  constexpr std::size_t output_stride = rows + 3;
  std::vector<float> scalar_batch(vectors * output_stride, -91.0F);
  std::vector<float> test_batch(vectors * output_stride, -91.0F);
  std::vector<float> batch_scales(batch_test.size());
  std::vector<float> matrix_scales(lhs_test.size());
  qwen35x::cpu::q8_0_scales_to_f32(
    batch_test.data(), batch_scales.data(), batch_test.size());
  qwen35x::cpu::q8_0_scales_to_f32(
    lhs_test.data(), matrix_scales.data(), lhs_test.size());
  qwen35x::cpu::q8_0_matmul(
    lhs_scalar.data(),
    batch_scalar.data(),
    scalar_batch.data(),
    rows,
    vectors,
    blocks,
    output_stride,
    Q8_0Backend::scalar);
  qwen35x::cpu::q8_0_matmul(
    lhs_test.data(),
    batch_test.data(),
    test_batch.data(),
    rows,
    vectors,
    blocks,
    output_stride,
    backend,
    batch_scales.data(),
    matrix_scales.data());
  for (std::size_t vector_index = 0; vector_index < vectors; ++vector_index) {
    for (std::size_t row = 0; row < rows; ++row) {
      const std::size_t index = vector_index * output_stride + row;
      ok = expect(near(scalar_batch[index], test_batch[index]), "matmul output mismatch") && ok;
    }
    for (std::size_t padding = rows; padding < output_stride; ++padding) {
      ok = expect(
        test_batch[vector_index * output_stride + padding] == -91.0F,
        "matmul overwrote output stride padding") && ok;
    }
  }
  return ok;
}

} // namespace

int main() {
  using namespace qwen35x::cpu;
  for (bool vex : {false, true}) {
    for (bool wide : {false, true}) {
      for (bool evex : {false, true}) {
        const CpuCapabilities caps{true, vex, wide, evex};
        const auto expected = wide && evex ? Q8_0Backend::avx512_vnni
          : wide ? Q8_0Backend::avx512 : vex ? Q8_0Backend::avx_vnni : Q8_0Backend::avx2;
        if (!expect(q8_0_resolve_backend_for_capabilities(Q8_0Backend::auto_select, caps) == expected,
                    "synthetic independent capability dispatch failed")) return 1;
        if (!expect(q8_0_dot_backend_for_capabilities(Q8_0Backend::auto_select, caps) ==
                      (vex ? Q8_0Backend::avx_vnni : Q8_0Backend::avx2),
                    "Q8 chose unsupported VEX instructions")) return 1;
        if (!expect(q8_0_resolve_backend_for_capabilities(Q8_0Backend::avx2, caps) == Q8_0Backend::avx2,
                    "explicit AVX2 request was widened")) return 1;
      }
    }
  }
  if (!expect(q8_0_resolve_backend_for_capabilities(Q8_0Backend::auto_select, {}) == Q8_0Backend::scalar,
              "missing AVX state did not select scalar")) return 1;

  bool ok = test_known_layout() && test_zero_block() && test_empty_ranges();
  ok = test_silu_mul(Q8_0Backend::auto_select) && ok;
  ok = test_rms_norm(Q8_0Backend::auto_select) && ok;
  ok = test_add(Q8_0Backend::auto_select) && ok;
  ok = test_rope(Q8_0Backend::auto_select) && ok;
  ok = test_causal_conv1d_silu(Q8_0Backend::auto_select) && ok;
  ok = test_full_attention(Q8_0Backend::auto_select) && ok;
  ok = test_full_attention_decode_pairs(Q8_0Backend::auto_select) && ok;
  ok = test_l2_normalize(Q8_0Backend::auto_select) && ok;
  ok = test_backend(Q8_0Backend::scalar) && ok;
  ok = test_backend(Q8_0Backend::auto_select) && ok;

  if (qwen35x::cpu::q8_0_backend_available(Q8_0Backend::avx2)) {
    ok = expect(
      qwen35x::cpu::q8_0_resolve_backend(Q8_0Backend::avx2) == Q8_0Backend::avx2,
      "available AVX2 backend did not resolve to AVX2") && ok;
    ok = test_backend(Q8_0Backend::avx2) && ok;
    ok = test_rope(Q8_0Backend::avx2) && ok;
    ok = test_causal_conv1d_silu(Q8_0Backend::avx2) && ok;
    ok = test_full_attention(Q8_0Backend::avx2) && ok;
    ok = test_full_attention_decode_pairs(Q8_0Backend::avx2) && ok;
  } else {
    ok = expect(
      qwen35x::cpu::q8_0_resolve_backend(Q8_0Backend::avx2) == Q8_0Backend::scalar,
      "unavailable AVX2 backend did not safely fall back") && ok;
  }

  if (qwen35x::cpu::q8_0_backend_available(Q8_0Backend::avx_vnni)) {
    ok = expect(
      qwen35x::cpu::q8_0_resolve_backend(Q8_0Backend::avx_vnni) ==
        Q8_0Backend::avx_vnni,
      "available AVX-VNNI backend did not resolve to AVX-VNNI") && ok;
    ok = test_backend(Q8_0Backend::avx_vnni) && ok;
    ok = test_rope(Q8_0Backend::avx_vnni) && ok;
    ok = test_causal_conv1d_silu(Q8_0Backend::avx_vnni) && ok;
    ok = test_full_attention(Q8_0Backend::avx_vnni) && ok;
    ok = test_full_attention_decode_pairs(Q8_0Backend::avx_vnni) && ok;
  } else {
    const Q8_0Backend expected_fallback =
      qwen35x::cpu::q8_0_backend_available(Q8_0Backend::avx2)
      ? Q8_0Backend::avx2
      : Q8_0Backend::scalar;
    ok = expect(
      qwen35x::cpu::q8_0_resolve_backend(Q8_0Backend::avx_vnni) ==
        expected_fallback,
      "unavailable AVX-VNNI backend did not safely fall back") && ok;
  }

  if (qwen35x::cpu::q8_0_backend_available(Q8_0Backend::avx512)) {
    ok = expect(
      qwen35x::cpu::q8_0_resolve_backend(Q8_0Backend::avx512) ==
        Q8_0Backend::avx512,
      "available AVX-512 backend did not resolve to AVX-512") && ok;
    ok = test_backend(Q8_0Backend::avx512) && ok;
    ok = test_silu_mul(Q8_0Backend::avx512) && ok;
    ok = test_rms_norm(Q8_0Backend::avx512) && ok;
    ok = test_add(Q8_0Backend::avx512) && ok;
    ok = test_rope(Q8_0Backend::avx512) && ok;
    ok = test_causal_conv1d_silu(Q8_0Backend::avx512) && ok;
    ok = test_full_attention(Q8_0Backend::avx512) && ok;
    ok = test_full_attention_decode_pairs(Q8_0Backend::avx512) && ok;
    ok = test_l2_normalize(Q8_0Backend::avx512) && ok;
  }

  if (qwen35x::cpu::q8_0_backend_available(Q8_0Backend::avx512_vnni)) {
    ok = expect(
      qwen35x::cpu::q8_0_resolve_backend(Q8_0Backend::avx512_vnni) ==
        Q8_0Backend::avx512_vnni,
      "available AVX-512 VNNI backend did not resolve to AVX-512 VNNI") && ok;
    ok = test_backend(Q8_0Backend::avx512_vnni) && ok;
    ok = test_silu_mul(Q8_0Backend::avx512_vnni) && ok;
    ok = test_rms_norm(Q8_0Backend::avx512_vnni) && ok;
    ok = test_add(Q8_0Backend::avx512_vnni) && ok;
    ok = test_rope(Q8_0Backend::avx512_vnni) && ok;
    ok = test_causal_conv1d_silu(Q8_0Backend::avx512_vnni) && ok;
    ok = test_full_attention(Q8_0Backend::avx512_vnni) && ok;
    ok = test_full_attention_decode_pairs(Q8_0Backend::avx512_vnni) && ok;
    ok = test_l2_normalize(Q8_0Backend::avx512_vnni) && ok;
  }

  if (ok) {
    std::cout << "q8_0 scalar/dispatch tests passed; auto selected "
              << qwen35x::cpu::q8_0_backend_name(
                   qwen35x::cpu::q8_0_resolve_backend(Q8_0Backend::auto_select))
              << '\n';
    return 0;
  }
  return 1;
}
