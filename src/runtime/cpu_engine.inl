// Included after the reference implementation's internal CPU helpers.
struct CpuModel::Impl {
  ModelProfile profile;
  RuntimeDims dims;
  ModelWeights weights;
  cpu::Q8_0Backend backend;
};
CpuModel::CpuModel(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
CpuModel::~CpuModel() = default;
std::shared_ptr<const CpuModel>
CpuModel::load(const ModelProfile &profile,
               const CpuLoadOptions &options, std::string &error) {
  error.clear();
  if (profile.family != "qwen3.5" ||
      options.cpu_q4_h128_path.empty()) {
    error = "CpuModel currently requires a CPU Q4 H128 artifact.";
    return nullptr;
  }
  auto model = std::make_shared<Impl>();
  model->profile = profile;
  model->backend = cpu::q8_0_resolve_backend(options.cpu_q8_backend);
  if (!build_runtime_dims(profile, model->dims, error)) return nullptr;
  const auto &d=model->dims;
  if (d.n_layers!=24 || d.hidden!=1024 || d.intermediate!=3584 ||
      d.vocab_size!=248320 || d.n_heads!=8 || d.n_kv_heads!=2 || d.head_dim!=256 ||
      d.linear_num_v_heads!=16 || d.linear_head_k_dim!=128 || d.linear_head_v_dim!=128) {
    error="Only the Qwen3.5-0.8B text architecture is supported.";return nullptr;
  }
  if (
      !load_model_weights_from_q4_h128(options.cpu_q4_h128_path,
                          model->dims, profile, model->backend, 0,
                          model->weights, error))
    return nullptr;
  return std::shared_ptr<const CpuModel>(new CpuModel(std::move(model)));
}

struct CpuPrefix::Impl {
  std::shared_ptr<const CpuModel> model;
  CpuPrefixCacheSnapshot snapshot;
  std::vector<float> final_hidden;
  std::string trust_namespace;
  std::size_t prefill_chunk_size = 0, bytes = 0;
  bool paged_kv = false;
};

struct CpuEngine::Impl {
  struct Sequence {
    CpuRequestSpec spec;
    CpuRequestResult result;
    ModelState state;
    std::vector<int> counts;
    std::mt19937 rng;
    std::vector<float> logits;
    int pending_token = -1;
    bool resident = false;
    std::size_t consumed = 0;
    std::chrono::steady_clock::time_point arrival = std::chrono::steady_clock::now();
  };
  std::shared_ptr<const CpuModel> model;
  CpuEngineConfig config;
  std::unique_ptr<CpuExecutionContext> context;
  std::unordered_map<CpuRequestId, std::unique_ptr<Sequence>> requests;
  CpuRequestId next_id = 1;
  std::string attention_kernel;
  CpuEngineStats stats;
  bool scheduling = false, prefer_decode = true;
  CpuRequestId last_prefill = 0, last_decode = 0;
  std::vector<std::shared_ptr<const CpuPrefix>> prefixes;
  std::size_t prefix_bytes = 0;
  bool prefix_room(std::size_t bytes, std::string &error) {
    if (bytes > config.max_prefix_bytes) {
      error = "Prefix exceeds the hybrid snapshot budget.";
      return false;
    }
    for (auto it = prefixes.begin();
         prefix_bytes > config.max_prefix_bytes - bytes &&
         it != prefixes.end();) {
      if (it->use_count() == 1) {
        prefix_bytes -= (*it)->size_bytes();
        it = prefixes.erase(it);
      } else
        ++it;
    }
    if (prefix_bytes > config.max_prefix_bytes - bytes) {
      error = "Prefix store is full; active handles pin its snapshots.";
      return false;
    }
    return true;
  }
  bool adopt_prefix(const std::shared_ptr<const CpuPrefix> &prefix,
                    std::string &error) {
    if (std::find(prefixes.begin(), prefixes.end(), prefix) != prefixes.end())
      return true;
    if (!prefix_room(prefix->size_bytes(), error))
      return false;
    prefixes.push_back(prefix);
    prefix_bytes += prefix->size_bytes();
    return true;
  }
  struct BatchScratch {
    std::vector<float> x, normed, projected, core, row, attention, residual;
    std::vector<float> post_norm, gate_up, mlp_hidden, mlp_output, final_hidden;
    std::vector<float> logits;
    std::vector<cpu::Q4_0ArgmaxResult> greedy_results;
    std::vector<int> greedy_tokens;
  } batch;

#include "cpu_engine_greedy.inl"
#include "cpu_engine_state_batch.inl"

  bool forward_batch(std::span<Sequence *> rows, std::string &error) {
    const auto &d = model->impl_->dims;
    const auto &w = model->impl_->weights;
    const std::size_t count = rows.size(), hidden = d.hidden;
    const std::size_t intermediate = d.intermediate;
    auto &b = batch;
    b.x.resize(count * hidden);
    b.residual.resize(count * hidden);
    b.mlp_hidden.resize(count * intermediate);
    for (std::size_t r = 0; r < count; ++r)
      (w.embed_tokens.q4_dot4 ? cpu::q4_dot4_dequantize_row
                              : cpu::q4_0_packed_dequantize_row)(
          w.embed_tokens.packed_q4_0_blocks.data(), rows[r]->pending_token,
          b.x.data() + r * hidden, hidden / 32);
    std::size_t linear_index = 0, full_index = 0;
    auto project = [&](const TensorData &weight,
                       const std::vector<float> &input,
                       std::vector<float> &output) {
      if (!matmul_2d_quantized_batch(context.get(), weight, input, count,
                                     output, error))
        return false;
      ++stats.projection_batches;
      return true;
    };
    for (const auto &layer : w.layers) {
      rms_norm_qwen3next_batch(b.x, count, hidden, layer.input_layernorm,
                               d.rms_eps, b.normed);
      const auto &input_weight = layer.is_linear ? layer.linear.in_proj_all_cpu
                                                 : layer.full.qkv_proj_cpu;
      if (!project(input_weight, b.normed, b.projected))
        return false;
      const std::size_t width = b.projected.size() / count;
      const std::size_t core_width =
          layer.is_linear ? d.linear_v_dim : d.n_heads * d.head_dim;
      b.core.resize(count * core_width);
      // Each stateful kernel sees one request's state and absolute position;
      // no padded row can mutate state or access another request's scratch.
      if (config.parallel_state_batches &&
          count >= context->executor->thread_count()) {
        if (!state_batch(layer, rows, linear_index, full_index, width,
                         core_width, error))
          return false;
      } else {
        for (std::size_t r = 0; r < count; ++r) {
          auto &s = *rows[r];
          const std::span<const float> projected(b.projected.data() + r * width,
                                                 width);
          bool ok;
          if (layer.is_linear)
            ok = run_linear_attention_step(
                context.get(), layer, d, s.state.linear_states[linear_index],
                b.normed, b.row, error, projected, true);
          else {
            const auto position = s.result.committed_tokens;
            const auto rope_offset = position * (d.rope_dim / 2);
            ok = run_full_attention_step(
                context.get(), layer, d, s.state.full_states[full_index],
                b.normed, static_cast<int>(position),
                s.state.rope_cosine.data() + rope_offset,
                s.state.rope_sine.data() + rope_offset, b.row, error, projected, true);
          }
          if (!ok)
            return false;
          std::copy(b.row.begin(), b.row.end(),
                    b.core.begin() + r * core_width);
        }
      }
      if (layer.is_linear)
        ++linear_index;
      else
        ++full_index;
      if (!project(layer.is_linear ? layer.linear.out_proj : layer.full.o_proj,
                   b.core, b.attention))
        return false;
      cpu::add_f32(b.x.data(), b.attention.data(), b.residual.data(),
                   b.x.size());
      rms_norm_qwen3next_batch(b.residual, count, hidden,
                               layer.post_attention_layernorm, d.rms_eps,
                               b.post_norm);
      if (!project(layer.mlp_gate_up_cpu, b.post_norm, b.gate_up))
        return false;
      for (std::size_t r = 0; r < count; ++r) {
        const auto *gate = b.gate_up.data() + r * 2 * intermediate;
        cpu::silu_mul_f32(gate, gate + intermediate,
                          b.mlp_hidden.data() + r * intermediate, intermediate,
                          layer.mlp_down.q8_0_backend);
      }
      if (!project(layer.mlp_down, b.mlp_hidden, b.mlp_output))
        return false;
      cpu::add_f32(b.residual.data(), b.mlp_output.data(), b.x.data(),
                   b.x.size());
    }
    rms_norm_qwen3next_batch(b.x, count, hidden, w.final_norm, d.rms_eps,
                             b.final_hidden);
    const bool greedy =
        config.batch_greedy &&
        std::all_of(rows.begin(), rows.end(), [](const auto *s) {
          return s->spec.sampling.temperature <= 1e-6F &&
                 s->spec.forced_output_tokens.empty() &&
                 !s->spec.logits_callback;
        });
    b.greedy_tokens.clear();
    if (greedy) {
      if (!greedy_batch(rows, error))
        return false;
      ++stats.projection_batches;
      ++stats.greedy_lm_head_batches;
    } else if (!project(w.embed_tokens, b.final_hidden, b.logits))
      return false;
    ++stats.lm_head_batches;
    return true;
  }

  bool initialize(Sequence &s, std::string &error) {
    const auto &d = model->impl_->dims;
    const auto capacity =
        s.spec.prompt_tokens.size() + s.spec.max_new_tokens - 1;
    const auto half = static_cast<std::size_t>(d.rope_dim / 2);
    s.state.rope_inverse_frequency.resize(half);
    for (std::size_t i = 0; i < half; ++i)
      s.state.rope_inverse_frequency[i] =
          std::pow(d.rope_theta, -static_cast<float>(2 * i) / d.rope_dim);
    s.state.rope_cosine.resize(capacity * half);
    s.state.rope_sine.resize(capacity * half);
    for (std::size_t p = 0; p < capacity; ++p)
      for (std::size_t i = 0; i < half; ++i) {
        const float angle =
            static_cast<float>(p) * s.state.rope_inverse_frequency[i];
        s.state.rope_cosine[p * half + i] = std::cos(angle);
        s.state.rope_sine[p * half + i] = std::sin(angle);
      }
    for (const auto &layer : model->impl_->weights.layers) {
      if (layer.is_linear) {
        auto &state = s.state.linear_states.emplace_back();
        state.conv_state.resize(static_cast<std::size_t>(d.linear_kernel - 1) *
                                d.linear_conv_channels);
        state.recurrent_state.resize(
            static_cast<std::size_t>(d.linear_num_v_heads) *
            d.linear_head_k_dim * d.linear_head_v_dim);
      } else {
        auto &state = s.state.full_states.emplace_back();
        const auto values =
            capacity * static_cast<std::size_t>(d.n_kv_heads) * d.head_dim;
        if (config.shared_kv_pages) {
          state.pages=cpu::KvPagesF16(static_cast<std::size_t>(d.n_kv_heads)*d.head_dim,capacity);
        } else if (cpu::q8_0_backend_uses_avx2(model->impl_->backend)) {
          state.k_cache_f16.resize(values);
          state.v_cache_f16.resize(values);
        } else {
          state.k_cache.resize(values);
          state.v_cache.resize(values);
        }
      }
    }
    s.counts.resize(d.vocab_size);
    for (auto token : s.spec.prompt_tokens)
      ++s.counts[token];
    s.rng.seed(s.spec.sampling.seed < 0
                   ? std::random_device{}()
                   : static_cast<std::uint32_t>(s.spec.sampling.seed));
    return true;
  }
  bool activate(Sequence &s, std::string &error) {
    if (!initialize(s, error)) return false;
    if (s.spec.prefix) {
      const auto start = std::chrono::steady_clock::now();
      const auto count = s.spec.prefix->token_count();
      const auto &d = model->impl_->dims;
      if (!restore_cpu_prefix_state(s.spec.prefix->impl_->snapshot, s.state,
                                    count, d.n_kv_heads * d.head_dim)) {
        error = "Invalid complete hybrid prefix snapshot.";
        return false;
      }
      s.result.prefix_restore_ms = elapsed_ms(start);
      s.result.cached_prefix_tokens = count;
      s.result.committed_tokens = count;
      if (!adopt_prefix(s.spec.prefix, error)) return false;
    }
    s.resident = true;
    s.result.status = CpuRequestStatus::prefill;
    return true;
  }
  #include "cpu_engine_singleflight.inl"
  static bool terminal(const Sequence &s) {
    return s.result.status == CpuRequestStatus::complete ||
           s.result.status == CpuRequestStatus::cancelled ||
           s.result.status == CpuRequestStatus::failed;
  }
  static std::size_t confirmed(const Sequence &s) {
    const auto &tokens = s.result.output_tokens;
    if (terminal(s)) return tokens.size();
    std::size_t held = 0;
    for (const auto &stop : s.spec.stop_token_sequences)
      for (std::size_t n=1; n<stop.size() && n<=tokens.size(); ++n)
        if (std::equal(stop.begin(), stop.begin()+n, tokens.end()-n))
          held = std::max(held, n);
    return tokens.size()-held;
  }
  bool output_ready(const Sequence &s) const {
    return s.result.output_tokens.size()-s.consumed < config.max_buffered_tokens;
  }
  void retire(Sequence &s) {
    if (!config.max_queued_requests || !s.resident || !terminal(s)) return;
    s.state = ModelState{};
    std::vector<int>().swap(s.counts);
    std::vector<float>().swap(s.logits);
    s.spec.prefix.reset();
    s.resident = false;
  }
  bool select(Sequence &s, int greedy_token, std::string &error) {
    const auto index = s.result.output_tokens.size();
    if (s.spec.logits_callback &&
        !s.spec.logits_callback(s.spec.logits_callback_context, index,
                                s.spec.forced_output_tokens.empty()
                                    ? -1
                                    : s.spec.forced_output_tokens[index],
                                s.logits.data(), s.logits.size(), error))
      return false;
    int token = greedy_token;
    if (!s.spec.forced_output_tokens.empty())
      token = s.spec.forced_output_tokens[index];
    else if (token < 0 &&
             !sample_token_from_logits(s.logits, s.spec.sampling, s.counts,
                                       s.rng, token, error))
      return false;
    if (token < 0 || token >= model->impl_->dims.vocab_size) {
      error = "Invalid sampled token.";
      return false;
    }
    s.result.output_tokens.push_back(token);
    ++s.counts[token];
    s.pending_token = token;
    std::size_t trim =
        std::find(s.spec.stop_token_ids.begin(), s.spec.stop_token_ids.end(),
                  token) != s.spec.stop_token_ids.end()
            ? 1
            : 0;
    for (const auto &stop : s.spec.stop_token_sequences)
      if (generated_ends_with_sequence(s.result.output_tokens, stop))
        trim = std::max(trim, stop.size());
    if (trim) {
      s.result.output_tokens.resize(s.result.output_tokens.size() - trim);
      s.result.status = CpuRequestStatus::complete;
    } else
      s.result.status = s.result.output_tokens.size() == s.spec.max_new_tokens
                            ? CpuRequestStatus::complete
                            : CpuRequestStatus::decode;
    return true;
  }
  bool advance(Sequence &s, std::string &error) {
    if (s.result.status != CpuRequestStatus::prefill &&
        s.result.status != CpuRequestStatus::decode)
      return true;
    const auto &d = model->impl_->dims;
    const auto &w = model->impl_->weights;
    const bool prefill = s.result.status == CpuRequestStatus::prefill;
    const bool greedy = s.spec.sampling.temperature <= 1e-6F &&
                        s.spec.forced_output_tokens.empty() &&
                        !s.spec.logits_callback;
    CpuGreedySamplingState sampling{
        &s.counts, s.spec.sampling.repetition_penalty, -1, greedy};
    bool predict = true, ok;
    const auto start = std::chrono::steady_clock::now();
    if (prefill && s.result.committed_tokens == s.spec.prompt_tokens.size()) {
      // A complete hit still performs private head selection; a cached sample
      // must never substitute for this request's penalties or RNG state.
      const auto &hidden = s.spec.prefix->impl_->final_hidden;
      if (greedy)
        ok = greedy_q4_token(context.get(), w.embed_tokens, hidden, s.counts,
                             s.spec.sampling.repetition_penalty,
                             sampling.next_token, error);
      else
        ok = compute_next_logits_from_embedding(context.get(), w.embed_tokens,
                                                hidden, s.logits, error);
      s.result.prefill_ms += elapsed_ms(start);
    } else if (prefill) {
      const auto chunk =
          config.prefill_chunk_size
              ? config.prefill_chunk_size
              : (s.spec.prompt_tokens.size() <= 128 ? 64U : 128U);
      // Restores can stop inside a canonical chunk. Finish that chunk first,
      // so row/tiled attention dispatch does not shift at the prefix boundary.
      auto count =
          std::min(chunk - s.result.committed_tokens % chunk,
                   s.spec.prompt_tokens.size() - s.result.committed_tokens);
      const auto boundary = s.spec.register_prefix_tokens;
      const bool building = boundary && !s.spec.prefix;
      if (building) count = std::min(count, boundary - s.result.committed_tokens);
      const bool publish = building && s.result.committed_tokens + count == boundary;
      std::vector<float> prefix_hidden;
      predict =
          s.result.committed_tokens + count == s.spec.prompt_tokens.size();
      ok = run_forward_cpu_q8_batch(
          context.get(), w, d, s.state,
          s.spec.prompt_tokens.data() + s.result.committed_tokens, count,
          static_cast<int>(s.result.committed_tokens), predict, s.logits,
          greedy ? &sampling : nullptr, nullptr, error,
          publish ? &prefix_hidden : nullptr);
      if (ok) {
        s.result.committed_tokens += count;
        s.result.prefill_tokens += count;
        if (publish) publish_prefix(s, std::move(prefix_hidden));
      }
      s.result.prefill_ms += elapsed_ms(start);
    } else {
      ok = run_forward_single_token(
          context.get(), w, d, s.state, s.pending_token,
          static_cast<int>(s.result.committed_tokens), s.logits, greedy ? &sampling : nullptr, error);
      if (ok) {
        ++s.result.committed_tokens;
        ++s.result.decode_forwards;
      }
      s.result.decode_ms += elapsed_ms(start);
    }
    if (ok && predict)
      ok = select(s, sampling.next_token, error);
    if (!ok) {
      s.result.status = CpuRequestStatus::failed;
      s.result.error = error;
    }
    return ok;
  }
};

CpuEngine::CpuEngine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CpuEngine::~CpuEngine() = default;
std::unique_ptr<CpuEngine>
CpuEngine::create(std::shared_ptr<const CpuModel> model,
                  const CpuEngineConfig &config, std::string &error) {
  error.clear();
  if (!model || config.threads < 0 || !config.max_resident_requests ||
      !config.max_context ||
      config.max_context >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      config.prefill_chunk_size > 2048 ||
      (model && config.shared_kv_pages && !cpu::q8_0_backend_uses_avx2(model->impl_->backend)) ||
      config.max_queued_requests > std::numeric_limits<std::size_t>::max() - config.max_resident_requests ||
      (config.max_queued_requests && (!config.max_decode_batch_size || !config.max_buffered_tokens))) {
    error = "Invalid CPU engine configuration.";
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->model = std::move(model);
  impl->config = config;
  impl->context = make_cpu_execution_context(config.threads, error);
  if (!impl->context)
    return nullptr;
  auto &rt = *impl->context;
  rt.automatic_attention = true;
  rt.attention_gqa = true;
  rt.attention_backend = impl->model->impl_->backend;
  const auto &d = impl->model->impl_->dims;
  rt.tiled_attention = d.head_dim == 256 && d.n_heads / d.n_kv_heads <= 4 &&
                       cpu::q8_0_backend_uses_avx2(rt.attention_backend);
  rt.query_tile = cpu::q8_0_backend_uses_avx512(rt.attention_backend) ? 16 : 8;
  rt.kv_tile = 32;
  rt.attention_kernel_result = &impl->attention_kernel;
  return std::unique_ptr<CpuEngine>(new CpuEngine(std::move(impl)));
}
CpuRequestId CpuEngine::submit(CpuRequestSpec spec, std::string &error) {
  error.clear();
  const auto &d = impl_->model->impl_->dims;
  auto valid_token = [&](std::int32_t t) { return t >= 0 && t < d.vocab_size; };
  if (impl_->requests.size() >= impl_->config.max_resident_requests + impl_->config.max_queued_requests ||
      !impl_->next_id || spec.prompt_tokens.empty() || !spec.max_new_tokens ||
      spec.prompt_tokens.size() > impl_->config.max_context ||
      spec.max_new_tokens - 1 >
          impl_->config.max_context - spec.prompt_tokens.size() ||
      (!spec.forced_output_tokens.empty() &&
       spec.forced_output_tokens.size() != spec.max_new_tokens) ||
      !std::all_of(spec.prompt_tokens.begin(), spec.prompt_tokens.end(),
                   valid_token) ||
      !std::all_of(spec.forced_output_tokens.begin(),
                   spec.forced_output_tokens.end(), valid_token) ||
      !std::isfinite(spec.sampling.temperature) ||
      spec.sampling.temperature < 0 || !std::isfinite(spec.sampling.top_p) ||
      spec.sampling.top_p <= 0 || spec.sampling.top_p > 1 ||
      spec.sampling.top_k < 0 ||
      !std::isfinite(spec.sampling.repetition_penalty) ||
      spec.sampling.repetition_penalty < 1) {
    error = "Invalid request or resident/context limit reached.";
    return 0;
  }
  if (impl_->config.max_queued_requests) {
    const auto queued = std::count_if(impl_->requests.begin(), impl_->requests.end(),
      [](const auto &entry) { return entry.second->result.status == CpuRequestStatus::queued; });
    if (static_cast<std::size_t>(queued) >= impl_->config.max_queued_requests ||
        std::any_of(spec.stop_token_sequences.begin(), spec.stop_token_sequences.end(),
          [&](const auto &stop) { return stop.size() > impl_->config.max_buffered_tokens; })) {
      error = "Queue full or stop sequence exceeds output buffer capacity.";
      return 0;
    }
  }
  if (spec.register_prefix_tokens &&
      (!impl_->config.max_queued_requests || spec.prefix || spec.trust_namespace.empty() ||
       spec.register_prefix_tokens > spec.prompt_tokens.size())) {
    error = "Automatic prefix builds require serving mode, a namespace and a valid prefix length.";
    return 0;
  }
  try {
    if (spec.prefix) {
      const auto &prefix = *spec.prefix->impl_;
      if (prefix.model != impl_->model ||
          prefix.trust_namespace != spec.trust_namespace ||
          prefix.prefill_chunk_size != impl_->config.prefill_chunk_size ||
          prefix.paged_kv != impl_->config.shared_kv_pages ||
          prefix.snapshot.prefix_tokens.size() > spec.prompt_tokens.size() ||
          !std::equal(prefix.snapshot.prefix_tokens.begin(),
                      prefix.snapshot.prefix_tokens.end(),
                      spec.prompt_tokens.begin())) {
        error = "Prefix model, configuration, namespace or token identity "
                "mismatch.";
        return 0;
      }
    }
    auto sequence = std::make_unique<Impl::Sequence>();
    sequence->spec = std::move(spec);
    if (impl_->config.max_queued_requests)
      sequence->result.status = CpuRequestStatus::queued;
    else if (!impl_->activate(*sequence, error)) return 0;
    const auto id = impl_->next_id;
    impl_->requests.emplace(id, std::move(sequence));
    ++impl_->next_id;
    return id;
  } catch (const std::bad_alloc &) {
    error = "Could not allocate request state.";
    return 0;
  }
}
bool CpuEngine::advance(CpuRequestId id, std::string &error) {
  error.clear();
  if (impl_->config.max_queued_requests && !impl_->scheduling) {
    error = "Serving mode must be driven through step.";
    return false;
  }
  auto it = impl_->requests.find(id);
  if (it == impl_->requests.end()) {
    error = "Unknown request ID.";
    return false;
  }
  return impl_->advance(*it->second, error);
}
bool CpuEngine::advance_batch(std::span<const CpuRequestId> ids,
                              std::string &error) {
  error.clear();
  if (impl_->config.max_queued_requests && !impl_->scheduling) {
    error = "Serving mode must be driven through step.";
    return false;
  }
  std::vector<Impl::Sequence *> rows;
  rows.reserve(ids.size());
  std::unordered_set<CpuRequestId> seen;
  for (auto id : ids) {
    auto it = impl_->requests.find(id);
    if (it == impl_->requests.end() || !seen.insert(id).second ||
        it->second->result.status != CpuRequestStatus::decode) {
      error = "Decode batch requires distinct active decode requests.";
      return false;
    }
    rows.push_back(it->second.get());
  }
  if (rows.empty())
    return true;
  if (rows.size() == 1)
    return impl_->advance(*rows[0], error);
  const auto start = std::chrono::steady_clock::now();
  bool ok = false;
  try {
    ok = impl_->forward_batch(rows, error);
  } catch (const std::bad_alloc &) {
    error = "Could not allocate decode batch workspace.";
  }
  const double ms = elapsed_ms(start);
  if (!ok) {
    // A failed forward may have updated state. Never silently retry it.
    for (auto *s : rows) {
      s->result.status = CpuRequestStatus::failed;
      s->result.error = error;
    }
    return false;
  }
  ++impl_->stats.batch_ticks;
  impl_->stats.batch_rows += rows.size();
  const auto vocab =
      static_cast<std::size_t>(impl_->model->impl_->dims.vocab_size);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    auto &s = *rows[r];
    ++s.result.committed_tokens;
    ++s.result.decode_forwards;
    // Attribution only, not a substitute for common wall-clock throughput.
    s.result.decode_ms += ms / rows.size();
    const bool greedy = !impl_->batch.greedy_tokens.empty();
    if (!greedy)
      s.logits.assign(impl_->batch.logits.data() + r * vocab,
                      impl_->batch.logits.data() + (r + 1) * vocab);
    std::string local_error;
    if (!impl_->select(s, greedy ? impl_->batch.greedy_tokens[r] : -1,
                       local_error)) {
      s.result.status = CpuRequestStatus::failed;
      s.result.error = local_error;
      if (error.empty())
        error = local_error;
      ok = false;
    }
  }
  return ok;
}
CpuEngineStats CpuEngine::stats() const {
  auto stats = impl_->stats;
  std::unordered_set<const void*> pages;
  auto count_state=[&](const auto &state) {
    stats.physical_kv_bytes+=(state.k_cache.size()+state.v_cache.size())*sizeof(float)+
      (state.k_cache_f16.size()+state.v_cache_f16.size())*sizeof(std::uint16_t);
    for(const auto &page:state.pages.pages())
      if(pages.insert(page.get()).second)
        stats.physical_kv_bytes+=(page->keys.size()+page->values.size())*sizeof(std::uint16_t);
  };
  for (const auto &[id,s] : impl_->requests) {
    stats.queued_requests += s->result.status == CpuRequestStatus::queued;
    stats.resident_requests += s->resident;
    for(const auto &state:s->state.full_states)count_state(state);
  }
  for(const auto &prefix:impl_->prefixes)
    for(const auto &state:prefix->impl_->snapshot.full_states)count_state(state);
  return stats;
}

bool CpuEngine::cancel(CpuRequestId id) {
  auto it = impl_->requests.find(id);
  if (it == impl_->requests.end())
    return false;
  if (it->second->result.status == CpuRequestStatus::queued ||
      it->second->result.status == CpuRequestStatus::prefill ||
      it->second->result.status == CpuRequestStatus::decode)
    it->second->result.status = CpuRequestStatus::cancelled;
  impl_->retire(*it->second);
  return true;
}
bool CpuEngine::release(CpuRequestId id) {
  return impl_->requests.erase(id) != 0;
}
bool CpuEngine::result(CpuRequestId id, CpuRequestResult &out) const {
  auto it = impl_->requests.find(id);
  if (it == impl_->requests.end())
    return false;
  out = it->second->result;
  return true;
}

#include "cpu_engine_prefix.inl"
#include "cpu_engine_scheduler.inl"
