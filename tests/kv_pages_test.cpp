#include "qwen35x/cpu/full_attention_tiled.h"
#include "qwen35x/cpu/kv_pages.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace qwen35x::cpu;
int main() try {
  auto check = [](bool ok) {
    if (!ok)
      throw std::runtime_error("KV page check failed");
  };
  std::vector<float> k(512, 1), v(512, 2);
  KvPagesF16 cache(512, 1024);
  for (int i = 0; i < 513; ++i)
    cache.append(k.data(), v.data(), Q8_0Backend::scalar);
  for (std::size_t n : {1U, 255U, 256U, 257U, 511U, 512U, 513U}) {
    auto prefix = cache.prefix(n);
    auto a = prefix.branch(1024), b = prefix.branch(1024);
    check(a.pages()[0] == b.pages()[0]);
    k[0] = 3;
    v[0] = 4;
    a.append(k.data(), v.data(), Q8_0Backend::scalar);
    const auto page = (n - 1) / 256;
    if (n % 256)
      check(a.pages()[page] != prefix.pages()[page]);
    else
      check(a.pages()[page] == prefix.pages()[page]);
    check(prefix.keys()[n - 1][0] == 0x3c00 && b.keys()[n - 1][0] == 0x3c00);
    check(a.keys()[n][0] == 0x4200 && a.values()[n][0] == 0x4400);
    for (std::size_t i = 0; i < n; ++i)
      check(a.keys()[i][0] == prefix.keys()[i][0]);
    prefix = {};
    check(b.keys()[n - 1][0] == 0x3c00);
  }
  bool rejected = false;
  try {
    cache.prefix(0);
  } catch (const std::out_of_range &) {
    rejected = true;
  }
  check(rejected);
  KvPagesF16 attention(512, 513);
  std::vector<std::uint16_t> keys(513 * 512), values(keys.size());
  for (std::size_t t = 0; t < 513; ++t) {
    for (std::size_t d = 0; d < 512; ++d) {
      k[d] = std::sin(float(t * 7 + d) * .017f);
      v[d] = std::cos(float(t + d * 3) * .013f);
    }
    attention.append(k.data(), v.data(), Q8_0Backend::scalar);
    attention_cache_store_f16(k.data(), keys.data() + t * 512, 512,
                              Q8_0Backend::scalar);
    attention_cache_store_f16(v.data(), values.data() + t * 512, 512,
                              Q8_0Backend::scalar);
  }
  AttentionKvRows pages{attention.keys(), attention.values()};
  std::vector<float> q(3 * 2048), gate(q.size()), a(q.size()), b(q.size()),
      scores(24 * 513);
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = std::sin(float(i) * .023f);
    gate[i] = std::cos(float(i) * .009f);
  }
  for (auto backend : {Q8_0Backend::scalar, Q8_0Backend::auto_select}) {
    causal_attention_batch_rows(q.data(), gate.data(), nullptr, nullptr,
                                keys.data(), values.data(), scores.data(),
                                a.data(), 513, 2048, 512, 510, 8, 2, 256,
                                .0625f, 0, 24, backend);
    causal_attention_batch_rows(q.data(), gate.data(), nullptr, nullptr,
                                nullptr, nullptr, scores.data(), b.data(), 513,
                                2048, 512, 510, 8, 2, 256, .0625f, 0, 24,
                                backend, false, &pages);
    check(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
    causal_attention_decode_gqa_pairs(q.data(), gate.data(), nullptr, nullptr,
                                      keys.data(), values.data(), scores.data(),
                                      a.data(), 513, 2048, 512, 513, 8, 2, 256,
                                      .0625f, 0, 4, backend);
    causal_attention_decode_gqa_pairs(q.data(), gate.data(), nullptr, nullptr,
                                      nullptr, nullptr, scores.data(), b.data(),
                                      513, 2048, 512, 513, 8, 2, 256, .0625f, 0,
                                      4, backend, &pages);
    check(std::memcmp(a.data(), b.data(), 2048 * sizeof(float)) == 0);
  }
  if (q8_0_backend_uses_avx2(Q8_0Backend::auto_select)) {
    TiledAttention args{
        q.data(), gate.data(), nullptr, nullptr, keys.data(), values.data(),
        a.data(), 3,           510,     2048,    512,         8,
        2,        256,         .0625f,  4,       32,          true};
    std::vector<float> scratch(tiled_attention_scratch_floats(args));
    for (std::size_t task = 0; task < tiled_attention_tasks(args); ++task)
      check(causal_attention_tiled(args, task, scratch.data(),
                                   Q8_0Backend::auto_select));
    args.keys_f16 = nullptr;
    args.values_f16 = nullptr;
    args.pages = &pages;
    args.output = b.data();
    for (std::size_t task = 0; task < tiled_attention_tasks(args); ++task)
      check(causal_attention_tiled(args, task, scratch.data(),
                                   Q8_0Backend::auto_select));
    check(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
    for (std::size_t cut : {1U, 255U, 256U, 511U, 513U}) {
      std::vector<float> left(a.size()), right(a.size()), lm(24), ls(24),
          rm(24), rs(24);
      args.gates = nullptr;
      args.output = left.data();
      args.kv_begin = 0;
      args.kv_end = cut;
      args.partial_maxima = lm.data();
      args.partial_sums = ls.data();
      for (std::size_t task = 0; task < tiled_attention_tasks(args); ++task)
        check(causal_attention_tiled(args, task, scratch.data(),
                                     Q8_0Backend::auto_select));
      args.output = right.data();
      args.kv_begin = cut;
      args.kv_end = 513;
      args.partial_maxima = rm.data();
      args.partial_sums = rs.data();
      for (std::size_t task = 0; task < tiled_attention_tasks(args); ++task)
        check(causal_attention_tiled(args, task, scratch.data(),
                                     Q8_0Backend::auto_select));
      for (std::size_t row = 0; row < 24; ++row) {
        merge_attention_partial(lm[row], ls[row], left.data() + row * 256,
                                rm[row], rs[row], right.data() + row * 256,
                                256);
        for (std::size_t d = 0; d < 256; ++d) {
          const auto index = row * 256 + d;
          const float g = gate[index], e = std::exp(-std::abs(g));
          const float gating = g >= 0 ? 1.f / (1.f + e) : e / (1.f + e);
          check(std::abs(left[index] / ls[row] * gating - a[index]) < 2e-5f);
        }
      }
    }
  }
  std::cout << "KV page boundaries, immutable branches, tail copy-on-write, "
               "lifetime and exact paged attention passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
