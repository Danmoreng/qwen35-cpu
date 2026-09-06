#include "../src/cpu/q8_0_internal.h"
#include "qwen35x/cpu/executor.h"
#include "qwen35x/cpu/full_attention.h"
#include "qwen35x/cpu/full_attention_tiled.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace qwen35x::cpu;
struct Job {
  TiledAttention a;
  Q8_0Backend backend;
  std::vector<float> scratch;
  std::size_t stride;
};
void execute(void *p, std::size_t begin, std::size_t end) noexcept {
  auto &j = *static_cast<Job *>(p);
  for (auto t = begin; t < end; ++t)
    causal_attention_tiled(j.a, t, j.scratch.data() + t * j.stride + 1, j.backend);
}
int main() {
  std::mt19937 rng(1701);
  std::uniform_real_distribution<float> random(-1.f, 1.f);
  double worst = 0;
  int cases = 0;
  for (auto backend : {Q8_0Backend::avx2, Q8_0Backend::avx512}) {
    if (std::string(tiled_attention_kernel(backend)) == "rows")
      continue;
    for (bool share : {false, true})
      for (bool half : {false, true})
        for (int tokens : {1, 7, 17, 65})
          for (int pos : {0, 37, 129, 4032})
            for (int bq : {4, 8, 16})
              for (int bk : {32, 64, 128}) {
                if (pos == 4032 && (tokens != 65 || bq != 8 || bk != 32))
                  continue;
                constexpr int H = 8, HK = 2, D = 256;
                const int count = pos + tokens;
                std::vector<float> q(tokens * H * D), g(q.size()), k(count * HK * D), v(k.size()),
                    out(q.size() + 2, -999.f);
                const float magnitude = cases % 7 == 0   ? 0.f
                                        : cases % 7 == 1 ? 1e-5f
                                        : cases % 7 == 2 ? 20.f
                                                         : 1.f;
                for (auto &x : q)
                  x = random(rng) * magnitude;
                for (auto &x : g)
                  x = random(rng) * 10;
                for (auto &x : k)
                  x = random(rng);
                for (auto &x : v)
                  x = random(rng);
                if (cases % 11 == 0) {
                  std::fill(q.begin(), q.end(), 50.f);
                  std::fill(k.begin(), k.end(), 50.f);
                }
                std::vector<std::uint16_t> kh(k.size()), vh(v.size());
                if (half) {
                  attention_cache_store_f16(k.data(), kh.data(), k.size(), backend);
                  attention_cache_store_f16(v.data(), vh.data(), v.size(), backend);
                  for (std::size_t i = 0; i < k.size(); ++i) {
                    k[i] = detail::half_to_float(kh[i]);
                    v[i] = detail::half_to_float(vh[i]);
                  }
                }
                TiledAttention a{q.data(),
                                 g.data(),
                                 half ? nullptr : k.data(),
                                 half ? nullptr : v.data(),
                                 half ? kh.data() : nullptr,
                                 half ? vh.data() : nullptr,
                                 out.data() + 1,
                                 static_cast<std::size_t>(tokens),
                                 static_cast<std::size_t>(pos),
                                 H * D,
                                 HK * D,
                                 H,
                                 HK,
                                 D,
                                 1.f / 16,
                                 bq,
                                 bk,
                                 share};
                const auto tasks = tiled_attention_tasks(a),
                           stride = tiled_attention_scratch_floats(a) + 2;
                Job job{a, backend, std::vector<float>(tasks * stride, -999.f), stride};
                std::error_code ec;
                auto executor = CpuExecutor::create({3, 1, 0}, ec);
                if (!executor ||
                    executor->parallel_for_rows(tasks, execute, &job) != CpuExecutorStatus::ok)
                  return 1;
                for (std::size_t t = 0; t < tasks; ++t)
                  if (job.scratch[t * stride] != -999.f ||
                      job.scratch[(t + 1) * stride - 1] != -999.f)
                    return 2;
                if (out.front() != -999.f || out.back() != -999.f)
                  return 3;
                for (int t = 0; t < tokens; ++t)
                  for (int h = 0; h < H; ++h) {
                    std::vector<double> scores(pos + t + 1);
                    double maximum = -INFINITY, total = 0;
                    for (int c = 0; c <= pos + t; ++c) {
                      double dot = 0;
                      for (int d = 0; d < D; ++d)
                        dot += double(q[(t * H + h) * D + d]) * k[(c * HK + h / (H / HK)) * D + d];
                      scores[c] = dot / 16;
                      maximum = std::max(maximum, scores[c]);
                    }
                    for (auto &s : scores) {
                      s = std::exp(s - maximum);
                      total += s;
                    }
                    for (int d = 0; d < D; ++d) {
                      double value = 0;
                      for (int c = 0; c <= pos + t; ++c)
                        value += scores[c] * v[(c * HK + h / (H / HK)) * D + d];
                      auto offset = (t * H + h) * D + d;
                      value = value / total / (1 + std::exp(-double(g[offset])));
                      const double error = std::abs(value - out[offset + 1]);
                      worst = std::max(worst, error);
                      if (!std::isfinite(out[offset + 1]) || error > 2e-5) {
                        std::cerr << "oracle error " << error << " pos " << pos << " tokens "
                                  << tokens << '\n';
                        return 4;
                      }
                    }
                  }
                // Whole-chunk KV is populated before execution. Altering future tokens
                // must never affect outputs for query zero, including diagonal tiles.
                const auto first_output = out;
                for (std::size_t i = (pos + 1) * HK * D; i < k.size(); ++i) {
                  k[i] = 123.f;
                  v[i] = -321.f;
                  kh[i] = 0x57b0;
                  vh[i] = 0xdd04;
                }
                execute(&job, 0, tasks);
                for (int d = 0; d < H * D; ++d)
                  if (out[d + 1] != first_output[d + 1])
                    return 5;
                ++cases;
              }
  }
  std::cout << cases << " tiled attention cases; maximum absolute FP64 error " << worst << '\n';
}
