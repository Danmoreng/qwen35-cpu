struct GatedDeltaNetCpuJob {
  float * state = nullptr;
  const float * q = nullptr;
  const float * k = nullptr;
  const float * v = nullptr;
  const float * alpha = nullptr;
  const float * beta = nullptr;
  float * output = nullptr;
  std::size_t head_count = 0;
  std::size_t key_dim = 0;
  std::size_t value_dim = 0;
  cpu::Q8_0Backend backend = cpu::Q8_0Backend::auto_select;
};

void run_gated_delta_net_cpu_rows(
  void * opaque_context,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  auto & job = *static_cast<GatedDeltaNetCpuJob *>(opaque_context);
  cpu::gated_delta_net_update_rows(
    job.state,
    job.q,
    job.k,
    job.v,
    job.alpha,
    job.beta,
    job.output,
    job.head_count,
    job.key_dim,
    job.value_dim,
    row_begin,
    row_end,
    job.backend);
}

bool run_linear_attention_step(CpuExecutionContext *cpu_context,
  const LayerWeights & layer,
  const RuntimeDims & dims,
  LinearAttentionState & state,
  const std::vector<float> & x,
  std::vector<float> & out,
  std::string & error_message,
  const std::span<const float> projected = {},
  const bool return_core = false,
  CpuDecodeWorkspace::Linear * row_workspace = nullptr) {
  if ((!projected.empty() && (projected.size() != static_cast<std::size_t>(dims.linear_conv_channels + dims.linear_v_dim + 2 * dims.linear_num_v_heads))) ||
      (return_core && projected.empty())) {
    error_message = "Invalid preprojected CPU attention row.";
    return false;
  }
  CpuDecodeWorkspace::Linear local_workspace;
  auto & workspace = row_workspace != nullptr ? *row_workspace : cpu_context != nullptr
    ? cpu_context->decode.linear : local_workspace;
  auto & mixed_qkv = workspace.mixed_qkv;
  auto & z_vec = workspace.z_vec;
  auto & b_vec = workspace.b_vec;
  auto & a_vec = workspace.a_vec;


  bool packed_projection = false;
  if (!projected.empty()) {
    packed_projection = true;
  }  else {
    if (layer.linear.in_proj_all_cpu.is_cpu_quantized()) {
      auto & packed = workspace.packed;
      if (!matvec_2d(cpu_context, layer.linear.in_proj_all_cpu, x, packed, error_message)) {
        return false;
      }
      const std::size_t qkv_count = static_cast<std::size_t>(dims.linear_conv_channels);
      const std::size_t z_count = static_cast<std::size_t>(dims.linear_v_dim);
      const std::size_t ba_count = static_cast<std::size_t>(dims.linear_num_v_heads);
      const std::size_t expected = qkv_count + z_count + 2 * ba_count;
      if (packed.size() != expected) {
        error_message = "Packed linear-attention projection output size mismatch.";
        return false;
      }
      packed_projection = true;
    } else {
      if (!matvec_2d(cpu_context, layer.linear.in_proj_qkv, x, mixed_qkv, error_message) ||
          !matvec_2d(cpu_context, layer.linear.in_proj_z, x, z_vec, error_message) ||
          !matvec_2d(cpu_context, layer.linear.in_proj_b, x, b_vec, error_message) ||
          !matvec_2d(cpu_context, layer.linear.in_proj_a, x, a_vec, error_message)) {
        return false;
      }
    }
  }

  const float * mixed_values = packed_projection ? (projected.empty() ? workspace.packed.data() : projected.data()) : mixed_qkv.data();
  const float * z_values = packed_projection ? mixed_values + dims.linear_conv_channels : z_vec.data();
  const float * b_values = packed_projection ? z_values + dims.linear_v_dim : b_vec.data();
  const float * a_values = packed_projection ? b_values + dims.linear_num_v_heads : a_vec.data();
  auto & beta = workspace.beta;
  beta.resize(static_cast<std::size_t>(dims.linear_num_v_heads));
  auto & alpha = workspace.alpha;
  alpha.resize(static_cast<std::size_t>(dims.linear_num_v_heads));
  for (int h = 0; h < dims.linear_num_v_heads; ++h) {
    beta[static_cast<std::size_t>(h)] = sigmoidf_stable(b_values[static_cast<std::size_t>(h)]);
    const float pre_gate = softplusf_stable(a_values[static_cast<std::size_t>(h)] + layer.linear.dt_bias.data[static_cast<std::size_t>(h)]);
    alpha[static_cast<std::size_t>(h)] = std::exp(pre_gate * layer.linear.ssm_a[static_cast<std::size_t>(h)]);
  }

  auto & conv_out = workspace.conv_out;
  conv_out.resize(static_cast<std::size_t>(dims.linear_conv_channels));
  const std::size_t conv_channels = static_cast<std::size_t>(dims.linear_conv_channels);
  const std::size_t conv_kernel = static_cast<std::size_t>(dims.linear_kernel);
  const std::size_t conv_history = conv_kernel - 1;
  if (layer.linear.conv1d_kernel_major.size() != conv_kernel * conv_channels) {
    error_message = "Kernel-major causal-convolution weights are missing.";
    return false;
  }
  cpu::causal_conv1d_silu_f32(
    state.conv_state.data(), state.conv_ring_index, mixed_values, conv_channels,
    layer.linear.conv1d_kernel_major.data(), conv_out.data(), 1, conv_channels,
    conv_kernel, 0, conv_channels,
    layer.linear.out_proj.q8_0_backend);
  if (conv_history != 0) {
    state.conv_ring_index = (state.conv_ring_index + 1) % conv_history;
  }

  const std::span<float> q(conv_out.data(), static_cast<std::size_t>(dims.linear_q_dim));
  const std::span<float> k(conv_out.data() + dims.linear_q_dim, static_cast<std::size_t>(dims.linear_q_dim));
  const std::span<const float> v(conv_out.data() + 2 * dims.linear_q_dim, static_cast<std::size_t>(dims.linear_v_dim));
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(dims.linear_head_k_dim));
  l2_norm_per_head(
    q, dims.linear_num_k_heads, dims.linear_head_k_dim, 1.0e-6F, q_scale);
  l2_norm_per_head(k, dims.linear_num_k_heads, dims.linear_head_k_dim);

  auto & core_out = workspace.core_out;
  core_out.resize(static_cast<std::size_t>(dims.linear_v_dim));
  const cpu::Q8_0Backend cpu_backend = layer.linear.out_proj.is_cpu_quantized()
    ? layer.linear.out_proj.q8_0_backend
    : cpu::Q8_0Backend::auto_select;
  CpuExecutionContext * const cpu_runtime = cpu_context;
  if (cpu_runtime != nullptr && cpu_runtime->executor != nullptr) {
    GatedDeltaNetCpuJob job{
      state.recurrent_state.data(),
      q.data(),
      k.data(),
      v.data(),
      alpha.data(),
      beta.data(),
      core_out.data(),
      static_cast<std::size_t>(dims.linear_num_v_heads),
      static_cast<std::size_t>(dims.linear_head_k_dim),
      static_cast<std::size_t>(dims.linear_head_v_dim),
      cpu_backend,
    };
    CpuDecodeProbe probe(cpu_runtime, "delta-net", dims.linear_num_v_heads * dims.linear_head_v_dim, dims.linear_head_k_dim);
    const cpu::CpuExecutorStatus status = cpu_runtime->executor->parallel_for_rows(
      static_cast<std::size_t>(dims.linear_num_v_heads * dims.linear_head_v_dim),
      run_gated_delta_net_cpu_rows,
      &job, probe.executor_timing());
    if (status != cpu::CpuExecutorStatus::ok) {
      error_message = std::string("Gated DeltaNet CPU executor failed: ") +
        cpu::cpu_executor_status_name(status) + ".";
      return false;
    }
  } else {
    cpu::gated_delta_net_update_rows(
      state.recurrent_state.data(),
      q.data(),
      k.data(),
      v.data(),
      alpha.data(),
      beta.data(),
      core_out.data(),
      static_cast<std::size_t>(dims.linear_num_v_heads),
      static_cast<std::size_t>(dims.linear_head_k_dim),
      static_cast<std::size_t>(dims.linear_head_v_dim),
      0,
      static_cast<std::size_t>(dims.linear_num_v_heads * dims.linear_head_v_dim),
      cpu_backend);
  }

  auto & gated_norm = workspace.gated_norm;
  gated_norm.resize(static_cast<std::size_t>(dims.linear_v_dim));
  cpu::rms_norm_f32(
    core_out.data(), layer.linear.norm.data.data(), gated_norm.data(),
    static_cast<std::size_t>(dims.linear_num_v_heads),
    static_cast<std::size_t>(dims.linear_head_v_dim), dims.rms_eps, 0.0F,
    cpu_backend);
  cpu::silu_mul_f32(
    z_values, gated_norm.data(), gated_norm.data(), gated_norm.size(),
    cpu_backend);

  if (return_core) {
    out.assign(gated_norm.begin(), gated_norm.end());
    return true;
  }
  if (!matvec_2d(cpu_context, layer.linear.out_proj, gated_norm, out, error_message)) {
    return false;
  }
  return true;
}

bool run_full_attention_step(CpuExecutionContext *cpu_context,
  const LayerWeights & layer,
  const RuntimeDims & dims,
  FullAttentionState & state,
  const std::vector<float> & x,
  const int position,
  const float * rope_cosine,
  const float * rope_sine,
  std::vector<float> & out,
  std::string & error_message,
  const std::span<const float> projected = {},
  const bool return_core = false,
  CpuDecodeWorkspace::Full * row_workspace = nullptr) {
  if ((!projected.empty() && (projected.size() != static_cast<std::size_t>((dims.n_heads + dims.n_kv_heads) * dims.head_dim * 2))) ||
      (return_core && projected.empty())) {
    error_message = "Invalid preprojected CPU attention row.";
    return false;
  }
  CpuDecodeWorkspace::Full local_workspace;
  auto & workspace = row_workspace != nullptr ? *row_workspace : cpu_context != nullptr
    ? cpu_context->decode.full : local_workspace;
  auto & q_full = workspace.q_full;
  auto & k_flat = workspace.k_flat;
  auto & v_flat = workspace.v_flat;
  bool packed_projection = false;
  const std::size_t full_q_out = static_cast<std::size_t>(dims.n_heads * dims.head_dim * 2);
  const std::size_t full_kv_out = static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim);
  if (!projected.empty()) {
    packed_projection = true;
  }  else {
    if (layer.full.qkv_proj_cpu.is_cpu_quantized()) {
      auto & packed = workspace.packed;
      if (!matvec_2d(cpu_context, layer.full.qkv_proj_cpu, x, packed, error_message)) {
        return false;
      }
      const std::size_t expected = full_q_out + 2 * full_kv_out;
      if (packed.size() != expected) {
        error_message = "Packed full-attention projection output size mismatch.";
        return false;
      }
      packed_projection = true;
    } else {
      if (!matvec_2d(cpu_context, layer.full.q_proj, x, q_full, error_message) ||
          !matvec_2d(cpu_context, layer.full.k_proj, x, k_flat, error_message) ||
          !matvec_2d(cpu_context, layer.full.v_proj, x, v_flat, error_message)) {
        return false;
      }
    }
  }

  const float * q_values = packed_projection ? (projected.empty() ? workspace.packed.data() : projected.data()) : q_full.data();
  const float * k_values = packed_projection ? q_values + full_q_out : k_flat.data();
  const float * v_values = packed_projection ? k_values + full_kv_out : v_flat.data();
  const int q_span = dims.head_dim * 2;
  auto & q = workspace.q;
  q.resize(static_cast<std::size_t>(dims.n_heads * dims.head_dim));
  auto & gate = workspace.gate;
  gate.resize(static_cast<std::size_t>(dims.n_heads * dims.head_dim));
  for (int h = 0; h < dims.n_heads; ++h) {
    const std::size_t src = static_cast<std::size_t>(h * q_span);
    const std::size_t dst = static_cast<std::size_t>(h * dims.head_dim);
    std::memcpy(q.data() + dst, q_values + src, static_cast<std::size_t>(dims.head_dim) * sizeof(float));
    std::memcpy(gate.data() + dst, q_values + src + static_cast<std::size_t>(dims.head_dim),
                static_cast<std::size_t>(dims.head_dim) * sizeof(float));
  }

  auto & q_normed = workspace.q_normed;
  auto & k_normed = workspace.k_normed;
  rms_norm_per_head_qwen3next(q, dims.n_heads, dims.head_dim, layer.full.q_norm, dims.rms_eps, q_normed);
  rms_norm_per_head_qwen3next(std::span<const float>(k_values, full_kv_out), dims.n_kv_heads, dims.head_dim, layer.full.k_norm, dims.rms_eps, k_normed);

  cpu::rope_f32(
    q_normed.data(), static_cast<std::size_t>(dims.n_heads),
    static_cast<std::size_t>(dims.head_dim), static_cast<std::size_t>(dims.rope_dim),
    rope_cosine, rope_sine, layer.full.o_proj.q8_0_backend);
  cpu::rope_f32(
    k_normed.data(), static_cast<std::size_t>(dims.n_kv_heads),
    static_cast<std::size_t>(dims.head_dim), static_cast<std::size_t>(dims.rope_dim),
    rope_cosine, rope_sine, layer.full.o_proj.q8_0_backend);

  const std::size_t token_stride = static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim);
  if (state.pages.width()) {
    if (state.pages.size()!=static_cast<std::size_t>(position)) {
      error_message="Paged KV append position mismatch.";return false;
    }
    state.pages.append(k_normed.data(),v_values,layer.full.o_proj.q8_0_backend);
  }
  if (!state.k_cache.empty()) {
    std::memcpy(
      state.k_cache.data() + static_cast<std::size_t>(position) * token_stride,
      k_normed.data(),
      token_stride * sizeof(float));
    std::memcpy(
      state.v_cache.data() + static_cast<std::size_t>(position) * token_stride,
      v_values,
      token_stride * sizeof(float));
  }
  if (!state.k_cache_f16.empty() && !state.v_cache_f16.empty()) {
    const std::size_t offset = static_cast<std::size_t>(position) * token_stride;
    cpu::attention_cache_store_f16(
      k_normed.data(), state.k_cache_f16.data() + offset, token_stride,
      layer.full.o_proj.q8_0_backend);
    cpu::attention_cache_store_f16(
      v_values, state.v_cache_f16.data() + offset, token_stride,
      layer.full.o_proj.q8_0_backend);
  }


  const float scale = 1.0f / std::sqrt(static_cast<float>(dims.head_dim));
  const int seq_len = position + 1;
  const std::size_t query_width =
    static_cast<std::size_t>(dims.n_heads * dims.head_dim);
  const std::size_t attention_rows = static_cast<std::size_t>(dims.n_heads);
  const std::size_t attention_pairs = attention_rows / 2U;
  const std::size_t context_stride = static_cast<std::size_t>(seq_len);
  auto & attn_cat = workspace.attn_cat;
  attn_cat.resize(query_width);
  auto & scores = workspace.scores;
  scores.resize(attention_rows * context_stride);
  const cpu::AttentionKvRows page_rows{state.pages.keys(),state.pages.values()};
  FullAttentionBatchCpuJob job{
    q_normed.data(),
    gate.data(),
    state.k_cache.data(),
    state.v_cache.data(),
    state.k_cache_f16.empty() ? nullptr : state.k_cache_f16.data(),
    state.v_cache_f16.empty() ? nullptr : state.v_cache_f16.data(),
    scores.data(),
    attn_cat.data(),
    context_stride,
    query_width,
    token_stride,
    position,
    dims.n_heads,
    dims.n_kv_heads,
    dims.head_dim,
    scale,
    layer.full.o_proj.q8_0_backend,
  };
  job.pages=state.pages.width()?&page_rows:nullptr;
  CpuExecutionContext * const runtime = cpu_context;
  if (runtime != nullptr && runtime->executor != nullptr) {
    CpuDecodeProbe probe(runtime, "full-attention", attention_pairs, seq_len);
    const cpu::CpuExecutorStatus status = runtime->executor->parallel_for_rows(
      attention_pairs, run_full_attention_decode_cpu_pairs, &job, probe.executor_timing());
    if (status != cpu::CpuExecutorStatus::ok) {
      error_message = std::string("Full-attention CPU executor failed: ") +
        cpu::cpu_executor_status_name(status) + ".";
      return false;
    }
  } else {
    run_full_attention_decode_cpu_pairs(&job, 0, attention_pairs);
  }

  if (return_core) {
    out.assign(attn_cat.begin(), attn_cat.end());
    return true;
  }
  if (!matvec_2d(cpu_context, layer.full.o_proj, attn_cat, out, error_message)) {
    return false;
  }
  return true;
}

bool compute_next_logits_from_embedding(CpuExecutionContext *cpu_context,
  const TensorData & embed,
  const std::vector<float> & hidden,
  std::vector<float> & out_logits,
  std::string & error_message) {
  const int vocab = static_cast<int>(embed.shape[0]);
  const int dim = static_cast<int>(embed.shape[1]);


    if (!matvec_2d(cpu_context, embed, hidden, out_logits, error_message)) {
      return false;
    }


  return true;
}

int argmax_index(const std::vector<float> & values) {
  int best_idx = 0;
  float best_value = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    if (values[static_cast<std::size_t>(i)] > best_value) {
      best_value = values[static_cast<std::size_t>(i)];
      best_idx = i;
    }
  }
  return best_idx;
}

void apply_repetition_penalty_inplace(
  std::vector<float> & logits,
  const std::vector<int> & token_counts,
  const float repetition_penalty) {
  if (repetition_penalty <= 1.0f) {
    return;
  }
  const std::size_t n = std::min(logits.size(), token_counts.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (token_counts[i] <= 0) {
      continue;
    }
    if (logits[i] > 0.0f) {
      logits[i] /= repetition_penalty;
    } else {
      logits[i] *= repetition_penalty;
    }
  }
}

bool sample_token_from_logits(
  const std::vector<float> & raw_logits,
  const SamplingOptions & sampling,
  const std::vector<int> & token_counts,
  std::mt19937 & rng,
  int & out_token,
  std::string & error_message) {
  if (raw_logits.empty()) {
    error_message = "Sampling logits are empty.";
    return false;
  }
  if (sampling.temperature < 0.0f) {
    error_message = "temperature must be >= 0.";
    return false;
  }
  if (sampling.top_p <= 0.0f || sampling.top_p > 1.0f) {
    error_message = "top_p must be in (0, 1].";
    return false;
  }
  if (sampling.top_k < 0) {
    error_message = "top_k must be >= 0.";
    return false;
  }
  if (sampling.repetition_penalty < 1.0f) {
    error_message = "repeat_penalty must be >= 1.0.";
    return false;
  }

  if (sampling.temperature <= 1.0e-6f) {
    if (sampling.repetition_penalty <= 1.0f) {
      out_token = argmax_index(raw_logits);
      return true;
    }
    int best_idx = 0;
    float best_value = -std::numeric_limits<float>::infinity();
    const std::size_t count = std::min(raw_logits.size(), token_counts.size());
    for (std::size_t index = 0; index < raw_logits.size(); ++index) {
      float value = raw_logits[index];
      if (index < count && token_counts[index] > 0) {
        value = value > 0.0f
          ? value / sampling.repetition_penalty
          : value * sampling.repetition_penalty;
      }
      if (value > best_value) {
        best_value = value;
        best_idx = static_cast<int>(index);
      }
    }
    out_token = best_idx;
    return true;
  }

  std::vector<float> logits = raw_logits;
  apply_repetition_penalty_inplace(logits, token_counts, sampling.repetition_penalty);

  const float inv_temp = 1.0f / sampling.temperature;
  for (float & v : logits) {
    v *= inv_temp;
  }

  std::vector<int> candidate_ids(logits.size());
  std::iota(candidate_ids.begin(), candidate_ids.end(), 0);
  if (sampling.top_k > 0 && sampling.top_k < static_cast<int>(candidate_ids.size())) {
    const std::size_t keep = static_cast<std::size_t>(sampling.top_k);
    std::nth_element(
      candidate_ids.begin(),
      candidate_ids.begin() + static_cast<std::ptrdiff_t>(keep),
      candidate_ids.end(),
      [&](const int a, const int b) {
        return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)];
      });
    candidate_ids.resize(keep);
  }

  std::sort(candidate_ids.begin(), candidate_ids.end(), [&](const int a, const int b) {
    return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)];
  });

  if (candidate_ids.empty()) {
    error_message = "Sampling candidates are empty after top_k filtering.";
    return false;
  }

  float max_logit = -std::numeric_limits<float>::infinity();
  for (const int id : candidate_ids) {
    max_logit = std::max(max_logit, logits[static_cast<std::size_t>(id)]);
  }

  std::vector<float> probs(candidate_ids.size(), 0.0f);
  float denom = 0.0f;
  for (std::size_t i = 0; i < candidate_ids.size(); ++i) {
    const float p = std::exp(logits[static_cast<std::size_t>(candidate_ids[i])] - max_logit);
    probs[i] = p;
    denom += p;
  }
  if (!(denom > 0.0f)) {
    out_token = candidate_ids.front();
    return true;
  }
  for (float & p : probs) {
    p /= denom;
  }

  std::size_t nucleus_keep = probs.size();
  if (sampling.top_p < 1.0f) {
    float cumulative = 0.0f;
    nucleus_keep = 0;
    for (; nucleus_keep < probs.size(); ++nucleus_keep) {
      cumulative += probs[nucleus_keep];
      if (cumulative >= sampling.top_p) {
        ++nucleus_keep;
        break;
      }
    }
    nucleus_keep = std::max<std::size_t>(1, std::min(nucleus_keep, probs.size()));
  }

  candidate_ids.resize(nucleus_keep);
  probs.resize(nucleus_keep);
  float renorm = 0.0f;
  for (const float p : probs) {
    renorm += p;
  }
  if (!(renorm > 0.0f)) {
    out_token = candidate_ids.front();
    return true;
  }
  for (float & p : probs) {
    p /= renorm;
  }

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  const float r = dist(rng);
  float cumulative = 0.0f;
  for (std::size_t i = 0; i < probs.size(); ++i) {
    cumulative += probs[i];
    if (r <= cumulative || i + 1 == probs.size()) {
      out_token = candidate_ids[i];
      return true;
    }
  }

  out_token = candidate_ids.back();
  return true;
}

bool generated_ends_with_sequence(
  const std::vector<std::int32_t> & generated_tokens,
  const std::vector<std::int32_t> & stop_sequence) {
  if (stop_sequence.empty() || stop_sequence.size() > generated_tokens.size()) {
    return false;
  }
  const std::size_t start = generated_tokens.size() - stop_sequence.size();
  for (std::size_t i = 0; i < stop_sequence.size(); ++i) {
    if (generated_tokens[start + i] != stop_sequence[i]) {
      return false;
    }
  }
  return true;
}
