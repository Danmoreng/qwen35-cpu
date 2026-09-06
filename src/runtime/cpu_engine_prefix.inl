// Included after CpuEngine::Impl and the public engine methods.
CpuPrefix::CpuPrefix(std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {}
CpuPrefix::~CpuPrefix() = default;
std::size_t CpuPrefix::token_count() const {
  return impl_->snapshot.prefix_tokens.size();
}
std::size_t CpuPrefix::size_bytes() const { return impl_->bytes; }

std::shared_ptr<const CpuPrefix>
CpuEngine::register_prefix(std::span<const std::int32_t> tokens,
                           std::string trust_namespace, std::string &error) {
  error.clear();
  const auto &d = impl_->model->impl_->dims;
  if (tokens.empty() || tokens.size() > impl_->config.max_context ||
      trust_namespace.empty() ||
      !std::all_of(tokens.begin(), tokens.end(),
                   [&](auto t) { return t >= 0 && t < d.vocab_size; })) {
    error = "Invalid registered prefix tokens or trust namespace.";
    return nullptr;
  }
  for (const auto &prefix : impl_->prefixes)
    if (prefix->impl_->trust_namespace == trust_namespace &&
        prefix->token_count() == tokens.size() &&
        std::equal(tokens.begin(), tokens.end(),
                   prefix->impl_->snapshot.prefix_tokens.begin()))
      return prefix;
  const bool fp16 = cpu::q8_0_backend_uses_avx2(impl_->model->impl_->backend);
  std::size_t bytes =
      tokens.size() * sizeof(std::int32_t) + d.hidden * sizeof(float);
  const auto kv_tokens=impl_->config.shared_kv_pages?
    ((tokens.size()+cpu::KvPagesF16::page_tokens-1)/cpu::KvPagesF16::page_tokens)*cpu::KvPagesF16::page_tokens:tokens.size();
  for (const auto &layer : impl_->model->impl_->weights.layers)
    if (layer.is_linear)
      bytes += (static_cast<std::size_t>(d.linear_kernel - 1) *
                    d.linear_conv_channels +
                static_cast<std::size_t>(d.linear_num_v_heads) *
                    d.linear_head_k_dim * d.linear_head_v_dim) *
               sizeof(float);
    else
      bytes += kv_tokens * d.n_kv_heads * d.head_dim * 2 *
               (fp16 ? sizeof(std::uint16_t) : sizeof(float))+
               (impl_->config.shared_kv_pages?tokens.size()*2*sizeof(const std::uint16_t*):0);
  if (!impl_->prefix_room(bytes, error))
    return nullptr;
  try {
    Impl::Sequence sequence;
    sequence.spec.prompt_tokens.assign(tokens.begin(), tokens.end());
    sequence.spec.max_new_tokens = 1;
    if (!impl_->initialize(sequence, error))
      return nullptr;
    auto prefix = std::make_shared<CpuPrefix::Impl>();
    prefix->model = impl_->model;
    prefix->trust_namespace = std::move(trust_namespace);
    prefix->prefill_chunk_size = impl_->config.prefill_chunk_size;
    prefix->paged_kv=impl_->config.shared_kv_pages;
    prefix->snapshot.prefix_tokens = sequence.spec.prompt_tokens;
    const std::size_t chunk = impl_->config.prefill_chunk_size
                                  ? impl_->config.prefill_chunk_size
                              : tokens.size() <= 128 ? 64
                                                     : 128;
    for (std::size_t position = 0; position < tokens.size();) {
      const auto count = std::min(chunk, tokens.size() - position);
      const bool last = position + count == tokens.size();
      if (!run_forward_cpu_q8_batch(
              impl_->context.get(), impl_->model->impl_->weights, d,
              sequence.state, tokens.data() + position, count,
              static_cast<int>(position), false, sequence.logits, nullptr,
              nullptr, error, last ? &prefix->final_hidden : nullptr))
        return nullptr;
      position += count;
    }
    capture_cpu_prefix_state(sequence.state, tokens.size(),
                             d.n_kv_heads * d.head_dim, fp16, prefix->snapshot);
    prefix->bytes = cpu_prefix_cache_size_bytes(prefix->snapshot) +
                    prefix->final_hidden.size() * sizeof(float);
    auto handle =
        std::shared_ptr<const CpuPrefix>(new CpuPrefix(std::move(prefix)));
    impl_->prefixes.push_back(handle);
    impl_->prefix_bytes += handle->size_bytes();
    return handle;
  } catch (const std::bad_alloc &) {
    error = "Could not allocate registered prefix.";
    return nullptr;
  }
}
