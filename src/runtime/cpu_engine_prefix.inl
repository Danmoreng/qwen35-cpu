// Included after CpuEngine::Impl and the public engine methods.
CpuPrefix::CpuPrefix(std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {}
CpuPrefix::~CpuPrefix() = default;
std::size_t CpuPrefix::token_count() const {
  return impl_->snapshot.prefix_tokens.size();
}
std::size_t CpuPrefix::size_bytes() const { return impl_->bytes; }

std::shared_ptr<const CpuPrefix>
CpuEngine::find_prefix(std::span<const std::int32_t> tokens, const std::string &trust_namespace,
                       CpuDraftMode draft_mode) {
  auto &prefixes = impl_->prefixes;
  auto best = prefixes.end();
  const bool mtp = draft_mode == CpuDraftMode::mtp;
  for (auto it = prefixes.begin(); it != prefixes.end(); ++it) {
    const auto &prefix = **it;
    if (prefix.impl_->trust_namespace == trust_namespace && prefix.impl_->has_mtp() == mtp &&
        prefix.token_count() <= tokens.size() &&
        (best == prefixes.end() || prefix.token_count() > (*best)->token_count()) &&
        std::equal(prefix.impl_->snapshot.prefix_tokens.begin(), prefix.impl_->snapshot.prefix_tokens.end(), tokens.begin()))
      best = it;
  }
  if (best == prefixes.end()) return {};
  auto result = *best;
  std::rotate(best, best + 1, prefixes.end());
  return result;
}

std::shared_ptr<const CpuPrefix>
CpuEngine::register_prefix(std::span<const std::int32_t> tokens,
                           std::string trust_namespace, std::string &error,
                           CpuDraftMode draft_mode) {
  error.clear();
  const auto &d = impl_->model->impl_->dims;
  const bool mtp = draft_mode == CpuDraftMode::mtp;
  if (mtp && !impl_->supports_mtp()) {
    error = "An MTP prefix requires a 4B sidecar and AVX2-capable Q4 DOT4 generation.";
    return nullptr;
  }
  if (tokens.empty() || tokens.size() > impl_->config.max_context ||
      trust_namespace.empty() ||
      !std::all_of(tokens.begin(), tokens.end(),
                   [&](auto t) { return t >= 0 && t < d.vocab_size; })) {
    error = "Invalid registered prefix tokens or trust namespace.";
    return nullptr;
  }
  for (const auto &prefix : impl_->prefixes)
    if (prefix->impl_->trust_namespace == trust_namespace &&
        prefix->impl_->has_mtp() == mtp &&
        prefix->token_count() == tokens.size() &&
        std::equal(tokens.begin(), tokens.end(),
                   prefix->impl_->snapshot.prefix_tokens.begin()))
      return prefix;
  if (!impl_->prefix_room(impl_->prefix_size_estimate(tokens.size(), mtp), error))
    return nullptr;
  try {
    Impl::Sequence sequence;
    sequence.spec.prompt_tokens.assign(tokens.begin(), tokens.end());
    sequence.spec.max_new_tokens = 1;
    sequence.spec.draft_mode = draft_mode;
    sequence.spec.register_prefix_tokens = tokens.size();
    sequence.spec.trust_namespace = std::move(trust_namespace);
    if (!impl_->initialize(sequence, error))
      return nullptr;
    std::vector<float> hidden;
    const std::size_t chunk = impl_->config.prefill_chunk_size
                                  ? impl_->config.prefill_chunk_size
                              : tokens.size() <= 128 ? 64
                                                     : 128;
    for (std::size_t position = 0; position < tokens.size();) {
      const auto count = detail::prefill_chunk_token_count(
          position, tokens.size() - position, chunk);
      const bool last = position + count == tokens.size();
      if (!run_forward_cpu_q8_batch(
              impl_->context.get(), impl_->model->impl_->weights, d,
              sequence.state, tokens.data() + position, count,
              static_cast<int>(position), false, sequence.logits, nullptr,
              nullptr, error, last ? &hidden : nullptr, {},
              mtp ? &impl_->mtp_prefill_hidden : nullptr))
        return nullptr;
      if (mtp && !impl_->advance_mtp_cache(sequence, position, tokens.subspan(position, count),
                                          impl_->mtp_prefill_hidden, impl_->stats.mtp_prefill_ms, error))
        return nullptr;
      position += count;
    }
    return impl_->capture_prefix(sequence, tokens.size(), std::move(hidden));
  } catch (const std::bad_alloc &) {
    error = "Could not allocate registered prefix.";
    return nullptr;
  }
}
