#include "qwen35x/cpu/delta_net.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using qwen35x::cpu::Q8_0Backend;

bool expect(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << "delta_net test failure: " << message << '\n';
  }
  return condition;
}

bool near(const float lhs, const float rhs) {
  const float tolerance = 2.0e-4F + 2.0e-4F * std::max(std::fabs(lhs), std::fabs(rhs));
  return std::fabs(lhs - rhs) <= tolerance;
}

std::vector<float> values(const std::size_t count, const float phase) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const float x = static_cast<float>(index) + phase;
    result[index] = 0.2F * std::sin(0.17F * x) + 0.1F * std::cos(0.07F * x);
  }
  return result;
}

bool test_backend(const Q8_0Backend backend) {
  constexpr std::size_t heads = 3;
  constexpr std::size_t key_dim = 19;
  constexpr std::size_t value_dim = 11;
  std::vector<float> expected_state = values(heads * value_dim * key_dim, 0.3F);
  std::vector<float> actual_state = expected_state;
  const std::vector<float> q = values(heads * key_dim, 1.1F);
  const std::vector<float> k = values(heads * key_dim, 2.7F);
  const std::vector<float> v = values(heads * value_dim, 4.2F);
  const std::vector<float> alpha{0.91F, 0.87F, 0.95F};
  const std::vector<float> beta{0.33F, 0.71F, 0.52F};
  std::vector<float> expected_output(heads * value_dim);
  std::vector<float> actual_output(heads * value_dim);

  qwen35x::cpu::gated_delta_net_update_rows(
    expected_state.data(), q.data(), k.data(), v.data(), alpha.data(), beta.data(),
    expected_output.data(), heads, key_dim, value_dim, 0, heads * value_dim,
    Q8_0Backend::scalar);
  // Exercise disjoint partitions and a non-vector-width tail.
  qwen35x::cpu::gated_delta_net_update_rows(
    actual_state.data(), q.data(), k.data(), v.data(), alpha.data(), beta.data(),
    actual_output.data(), heads, key_dim, value_dim, 0, 13, backend);
  qwen35x::cpu::gated_delta_net_update_rows(
    actual_state.data(), q.data(), k.data(), v.data(), alpha.data(), beta.data(),
    actual_output.data(), heads, key_dim, value_dim, 13, heads * value_dim, backend);

  bool ok = true;
  for (std::size_t index = 0; index < expected_state.size(); ++index) {
    ok = expect(near(expected_state[index], actual_state[index]), "state mismatch") && ok;
  }
  for (std::size_t index = 0; index < expected_output.size(); ++index) {
    ok = expect(near(expected_output[index], actual_output[index]), "output mismatch") && ok;
  }
  return ok;
}

bool test_batch_backend(const Q8_0Backend backend) {
  constexpr std::size_t batch = 5;
  constexpr std::size_t heads = 3;
  constexpr std::size_t key_dim = 19;
  constexpr std::size_t value_dim = 11;
  constexpr std::size_t q_stride = heads * key_dim;
  constexpr std::size_t v_stride = heads * value_dim;
  std::vector<float> expected_state = values(heads * value_dim * key_dim, 0.9F);
  std::vector<float> actual_state = expected_state;
  const std::vector<float> q = values(batch * q_stride, 1.7F);
  const std::vector<float> k = values(batch * q_stride, 2.9F);
  const std::vector<float> v = values(batch * v_stride, 4.7F);
  std::vector<float> alpha(batch * heads);
  std::vector<float> beta(batch * heads);
  for (std::size_t index = 0; index < alpha.size(); ++index) {
    alpha[index] = 0.84F + 0.01F * static_cast<float>(index % 7);
    beta[index] = 0.21F + 0.03F * static_cast<float>(index % 5);
  }
  std::vector<float> expected_output(batch * v_stride);
  std::vector<float> actual_output(batch * v_stride);
  std::vector<float> q_head_major(q.size());
  std::vector<float> k_head_major(k.size());
  std::vector<float> v_head_major(v.size());
  std::vector<float> alpha_head_major(alpha.size());
  std::vector<float> beta_head_major(beta.size());
  for (std::size_t head = 0; head < heads; ++head) {
    for (std::size_t token = 0; token < batch; ++token) {
      const std::size_t head_token = head * batch + token;
      std::copy_n(
        q.data() + token * q_stride + head * key_dim,
        key_dim,
        q_head_major.data() + head_token * key_dim);
      std::copy_n(
        k.data() + token * q_stride + head * key_dim,
        key_dim,
        k_head_major.data() + head_token * key_dim);
      std::copy_n(
        v.data() + token * v_stride + head * value_dim,
        value_dim,
        v_head_major.data() + head_token * value_dim);
      alpha_head_major[head_token] = alpha[token * heads + head];
      beta_head_major[head_token] = beta[token * heads + head];
    }
  }
  for (std::size_t token = 0; token < batch; ++token) {
    qwen35x::cpu::gated_delta_net_update_rows(
      expected_state.data(),
      q.data() + token * q_stride,
      k.data() + token * q_stride,
      v.data() + token * v_stride,
      alpha.data() + token * heads,
      beta.data() + token * heads,
      expected_output.data() + token * v_stride,
      heads,
      key_dim,
      value_dim,
      0,
      heads * value_dim,
      backend);
  }
  qwen35x::cpu::gated_delta_net_update_batch_rows(
    actual_state.data(), q_head_major.data(), k_head_major.data(), v_head_major.data(),
    alpha_head_major.data(), beta_head_major.data(),
    actual_output.data(), batch, heads, key_dim, value_dim, 0, 1, backend);
  qwen35x::cpu::gated_delta_net_update_batch_rows(
    actual_state.data(), q_head_major.data(), k_head_major.data(), v_head_major.data(),
    alpha_head_major.data(), beta_head_major.data(),
    actual_output.data(), batch, heads, key_dim, value_dim, 1, heads, backend);

  bool ok = true;
  for (std::size_t index = 0; index < expected_state.size(); ++index) {
    ok = expect(near(expected_state[index], actual_state[index]), "batched state mismatch") && ok;
  }
  for (std::size_t index = 0; index < expected_output.size(); ++index) {
    ok = expect(near(expected_output[index], actual_output[index]), "batched output mismatch") && ok;
  }
  return ok;
}

} // namespace

int main() {
  bool ok = test_backend(Q8_0Backend::scalar) && test_backend(Q8_0Backend::auto_select) &&
    test_batch_backend(Q8_0Backend::scalar) && test_batch_backend(Q8_0Backend::auto_select);
  if (qwen35x::cpu::q8_0_backend_available(Q8_0Backend::avx2)) {
    ok = test_backend(Q8_0Backend::avx2) && test_batch_backend(Q8_0Backend::avx2) && ok;
  }
  if (qwen35x::cpu::q8_0_backend_available(Q8_0Backend::avx_vnni)) {
    ok = test_backend(Q8_0Backend::avx_vnni) &&
      test_batch_backend(Q8_0Backend::avx_vnni) && ok;
  }
  if (qwen35x::cpu::q8_0_backend_available(Q8_0Backend::avx512)) {
    ok = test_backend(Q8_0Backend::avx512) &&
      test_batch_backend(Q8_0Backend::avx512) && ok;
  }
  if (qwen35x::cpu::q8_0_backend_available(Q8_0Backend::avx512_vnni)) {
    ok = test_backend(Q8_0Backend::avx512_vnni) &&
      test_batch_backend(Q8_0Backend::avx512_vnni) && ok;
  }
  if (!ok) {
    return 1;
  }
  std::cout << "gated DeltaNet scalar/runtime-dispatch tests passed\n";
  return 0;
}
