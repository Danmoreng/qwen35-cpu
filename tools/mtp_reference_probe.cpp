#include "../src/runtime/cpu_runtime.cpp"
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) try {
  using namespace qwen35x;
  if (argc != 3) throw std::runtime_error("Usage: qwen35_mtp_probe SIDECAR OUTPUT");
  std::string error;
  const auto check = [&](bool ok) { if (!ok) throw std::runtime_error(error); };
  const auto backend = cpu::q8_0_resolve_backend(cpu::Q8_0Backend::auto_select);
  MtpWeights weights;
  check(load_mtp_weights(argv[1], backend, weights, error));
  auto context = make_cpu_execution_context(6, error);
  check(bool(context));

  RuntimeDims dims;
  dims.hidden = 2560;
  dims.intermediate = 9216;
  dims.n_heads = 16;
  dims.n_kv_heads = 4;
  dims.head_dim = 256;
  dims.rope_dim = 64;
  constexpr std::array<std::int32_t, 7> tokens{0, 1, 2, 3, 4, 5, 6};
  const auto count = tokens.size();
  const auto hidden = static_cast<std::size_t>(dims.hidden);
  ModelState target;
  for (std::size_t position = 0; position < count; ++position) {
    for (int i = 0; i < dims.rope_dim / 2; ++i) {
      const float angle = static_cast<float>(position) *
          std::pow(dims.rope_theta, -static_cast<float>(2 * i) / dims.rope_dim);
      target.rope_cosine.push_back(std::cos(angle));
      target.rope_sine.push_back(std::sin(angle));
    }
  }

  std::vector<float> embeddings(count * hidden), preceding(count * hidden);
  for (std::size_t i = 0; i < embeddings.size(); ++i) {
    embeddings[i] = (static_cast<int>((i * 19 + 17) % 127) - 63) / 64.0F;
    preceding[i] = i < hidden ? 0.0F :
        (static_cast<int>((i * 23 + i / 11) % 193) - 96) / 96.0F;
  }
  TensorData embedding;
  embedding.q8_0_backend = backend;
  embedding.q8_0_blocks.resize(embeddings.size() / 32);
  cpu::q8_0_quantize(embeddings.data(), embedding.q8_0_blocks.data(),
                     embedding.q8_0_blocks.size(), cpu::Q8_0Backend::scalar);
  for (auto token : tokens) dequantize_embedding_row(embedding, token, embeddings.data() + token * hidden, hidden);

  MtpState sequential;
  sequential.attention.k_cache_f16.resize(count * dims.n_kv_heads * dims.head_dim);
  sequential.attention.v_cache_f16.resize(sequential.attention.k_cache_f16.size());
  auto batched = sequential;
  MtpWorkspace scratch;
  std::ofstream output(argv[2], std::ios::binary);
  const auto write = [&](std::span<const float> values) {
    output.write(reinterpret_cast<const char *>(values.data()), values.size_bytes());
    if (!output) throw std::runtime_error("Could not write MTP probe output.");
  };
  write(embeddings);
  write(preceding);
  for (std::size_t position = 0; position < count; ++position) {
    check(mtp_forward(context.get(), weights, embedding, dims, target, sequential, scratch,
          position, tokens[position], std::span(preceding).subspan(position * hidden, hidden), error));
    write(scratch.head_hidden);
  }
  check(mtp_store_kv(context.get(), weights, embedding, dims, target, batched, scratch, 0,
        std::span(tokens).first(count - 1), std::span(preceding).first(hidden),
        std::span(preceding).subspan(hidden), error));
  check(mtp_forward(context.get(), weights, embedding, dims, target, batched, scratch, count - 1,
        tokens.back(), std::span(preceding).last(hidden), error));
  write(scratch.head_hidden);
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
