namespace {

struct MtpWeights {
  TensorData projection, embedding_norm, hidden_norm, output_norm;
  LayerWeights layer;
};

bool load_mtp_weights(const std::string &path, cpu::Q8_0Backend backend,
                      MtpWeights &weights, std::string &error) {
  Q4H128ArtifactReader reader;
  if (!reader.open(path, error)) return false;
  const auto &metadata = reader.metadata();
  if (metadata.num_hidden_layers != 32 || metadata.hidden_size != 2560 ||
      metadata.intermediate_size != 9216 || metadata.vocabulary_size != 248320 ||
      reader.tensors().size() != qwen35_4b_mtp_tensors.size()) {
    error = "Expected a Qwen3.5-4B MTP sidecar.";
    return false;
  }
  const auto *projection = reader.find_tensor(qwen35_4b_mtp_tensors.front().name);
  if (!projection) {
    error = "MTP sidecar lacks its input projection.";
    return false;
  }
  const auto encoding = projection->encoding;
  using Encoding = Q4H128TensorEncoding;
  if (encoding != Encoding::bf16 && encoding != Encoding::q8_0 &&
      encoding != Encoding::q4_h128_cpu_dot4) {
    error = "MTP matrices must use BF16, Q8_0 or H128/Q4-G32-DOT4.";
    return false;
  }
  auto &layer = weights.layer;
  const std::array targets{
    &weights.projection, &weights.embedding_norm, &weights.hidden_norm, &layer.input_layernorm,
    &layer.full.q_proj, &layer.full.k_proj, &layer.full.v_proj, &layer.full.q_norm,
    &layer.full.k_norm, &layer.full.o_proj, &layer.post_attention_layernorm,
    &layer.mlp_gate, &layer.mlp_up, &layer.mlp_down, &weights.output_norm,
  };
  static_assert(targets.size() == qwen35_4b_mtp_tensors.size());
  for (std::size_t index = 0; index < targets.size(); ++index) {
    const auto &shape = qwen35_4b_mtp_tensors[index];
    const std::string name(shape.name);
    const auto rows = static_cast<std::int64_t>(shape.rows);
    const auto cols = static_cast<std::int64_t>(shape.columns);
    auto &tensor = *targets[index];
    if (!cols) {
      if (!load_q4_h128_f32_checked(reader, name, {rows}, tensor, error)) return false;
      if (std::any_of(tensor.data.begin(), tensor.data.end(), [](float value) { return !std::isfinite(value); })) {
        error = "Non-finite MTP normalization weight: " + name;
        return false;
      }
      continue;
    }
    const auto *info = reader.find_tensor(name);
    if (!info || info->encoding != encoding || !q4_h128_shape_matches(info->shape, {rows, cols})) {
      error = "Missing or incompatible MTP matrix: " + name;
      return false;
    }
    if (encoding == Encoding::bf16) {
      tensor.shape = {rows, cols};
      tensor.q8_0_backend = backend;
      tensor.bf16.resize(static_cast<std::size_t>(rows * cols));
      if (!reader.read_tensor_into(name, tensor.bf16.data(), tensor.bf16.size() * sizeof(std::uint16_t), error))
        return false;
      if (std::any_of(tensor.bf16.begin(), tensor.bf16.end(), [](auto bits) { return (bits & 0x7f80U) == 0x7f80U; })) {
        error = "Non-finite MTP BF16 weight: " + name;
        return false;
      }
    } else if (!load_q4_h128_quantized_checked(reader, name, rows, cols, encoding == Encoding::q4_h128_cpu_dot4,
                                               backend, tensor, error)) {
      return false;
    }
  }
  return true;
}

struct MtpState {
  FullAttentionState attention;
  std::vector<float> target_hidden;
};

struct MtpWorkspace {
  std::vector<float> embedding, concat, projected, normed, keys, values, key_norm;
  std::vector<float> attention, residual, gate, up, mlp, head_hidden;
};

bool mtp_project(CpuExecutionContext *context, const TensorData &weight,
                 const std::vector<float> &input, std::size_t count,
                 std::vector<float> &output, std::string &error) {
  if (count == 1) return matvec_2d(context, weight, input, output, error);
  if (!weight.bf16.empty()) return bf16_matmul(context, weight, input, count, output, error);
  return matmul_2d_quantized_batch(context, weight, input, count, output, error);
}

bool mtp_prepare(CpuExecutionContext *context, const MtpWeights &weights,
                 const TensorData &embedding, const RuntimeDims &dims,
                 MtpWorkspace &scratch, std::span<const std::int32_t> tokens,
                 std::span<const float> previous, std::span<const float> preceding_rows,
                 std::string &error) {
  const auto hidden = static_cast<std::size_t>(dims.hidden);
  const auto count = tokens.size();
  scratch.embedding.resize(hidden);
  scratch.concat.resize(count * 2 * hidden);
  // At position p the MTP input pairs token p with the target's normalized
  // hidden state at p-1. Position zero uses a zero hidden row.
  for (std::size_t row = 0; row < count; ++row) {
    const float *hidden_input = row ? preceding_rows.data() + (row - 1) * hidden : previous.data();
    const auto input = scratch.concat.data() + row * 2 * hidden;
    dequantize_embedding_row(embedding, tokens[row], scratch.embedding.data(), hidden);
    cpu::rms_norm_f32(scratch.embedding.data(), weights.embedding_norm.data.data(),
                     input, 1, hidden, dims.rms_eps, 1.0F);
    cpu::rms_norm_f32(hidden_input, weights.hidden_norm.data.data(),
                     input + hidden, 1, hidden, dims.rms_eps, 1.0F);
  }
  if (!mtp_project(context, weights.projection, scratch.concat, count, scratch.projected, error))
    return false;
  rms_norm_qwen3next_batch(scratch.projected, count, hidden, weights.layer.input_layernorm,
                           dims.rms_eps, scratch.normed);
  return true;
}

bool mtp_store_kv(CpuExecutionContext *context, const MtpWeights &weights,
                  const TensorData &embedding, const RuntimeDims &dims, const ModelState &target,
                  MtpState &state, MtpWorkspace &scratch, std::size_t position,
                  std::span<const std::int32_t> tokens, std::span<const float> previous,
                  std::span<const float> preceding_rows, std::string &error) {
  const auto count = tokens.size();
  if (!count) return true;
  const auto &full = weights.layer.full;
  if (!mtp_prepare(context, weights, embedding, dims, scratch, tokens, previous, preceding_rows, error) ||
      !mtp_project(context, full.k_proj, scratch.normed, count, scratch.keys, error) ||
      !mtp_project(context, full.v_proj, scratch.normed, count, scratch.values, error))
    return false;
  const auto kv_width = static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim);
  const auto backend = full.o_proj.q8_0_backend;
  for (std::size_t row = 0; row < count; ++row) {
    rms_norm_per_head_qwen3next({scratch.keys.data() + row * kv_width, kv_width}, dims.n_kv_heads,
        dims.head_dim, full.k_norm, dims.rms_eps, scratch.key_norm);
    const auto rope = (position + row) * (dims.rope_dim / 2);
    cpu::rope_f32(scratch.key_norm.data(), dims.n_kv_heads, dims.head_dim, dims.rope_dim,
        target.rope_cosine.data() + rope, target.rope_sine.data() + rope, backend);
    const auto offset = (position + row) * kv_width;
    cpu::attention_cache_store_f16(scratch.key_norm.data(), state.attention.k_cache_f16.data() + offset,
                                  kv_width, backend);
    cpu::attention_cache_store_f16(scratch.values.data() + row * kv_width,
                                  state.attention.v_cache_f16.data() + offset, kv_width, backend);
  }
  return true;
}

bool mtp_forward(CpuExecutionContext *context, const MtpWeights &weights,
                 const TensorData &embedding, const RuntimeDims &dims, const ModelState &target,
                 MtpState &state, MtpWorkspace &scratch, std::size_t position,
                 std::int32_t token, std::span<const float> previous, std::string &error) {
  if (!mtp_prepare(context, weights, embedding, dims, scratch, {&token, 1}, previous, {}, error))
    return false;
  const auto &layer = weights.layer;
  const auto rope = position * (dims.rope_dim / 2);
  if (!run_full_attention_step(context, layer, dims, state.attention, scratch.normed,
        static_cast<int>(position), target.rope_cosine.data() + rope, target.rope_sine.data() + rope,
        scratch.attention, error))
    return false;
  scratch.residual.resize(dims.hidden);
  cpu::add_f32(scratch.projected.data(), scratch.attention.data(), scratch.residual.data(), dims.hidden);
  rms_norm_qwen3next(scratch.residual, layer.post_attention_layernorm, dims.rms_eps, scratch.normed);
  if (!matvec_2d(context, layer.mlp_gate, scratch.normed, scratch.gate, error) ||
      !matvec_2d(context, layer.mlp_up, scratch.normed, scratch.up, error))
    return false;
  cpu::silu_mul_f32(scratch.gate.data(), scratch.up.data(), scratch.gate.data(),
                   dims.intermediate, layer.mlp_down.q8_0_backend);
  if (!matvec_2d(context, layer.mlp_down, scratch.gate, scratch.mlp, error))
    return false;
  cpu::add_f32(scratch.residual.data(), scratch.mlp.data(), scratch.residual.data(), dims.hidden);
  rms_norm_qwen3next(scratch.residual, weights.output_norm, dims.rms_eps, scratch.head_hidden);
  return true;
}

} // namespace
