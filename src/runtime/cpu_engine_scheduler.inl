// Included after CpuEngine::Impl and the direct execution methods.
bool CpuEngine::step(bool &progressed, std::string &error) {
  progressed = false;
  error.clear();
  auto &e = *impl_;
  if (!e.config.max_queued_requests || e.scheduling) {
    error = "step requires serving mode and a non-reentrant owner.";
    return false;
  }
  struct Guard {
    bool &value;
    explicit Guard(bool &v) : value(v) { value = true; }
    ~Guard() { value = false; }
  } guard(e.scheduling);
  std::size_t resident = 0;
  CpuRequestId queued = 0;
  for (auto &[id, s] : e.requests) {
    e.retire(*s);
    resident += s->resident;
    if (s->result.status == CpuRequestStatus::queued && e.prefix_admissible(*s) &&
        (!queued || id < queued))
      queued = id;
  }
  if (queued && resident < e.config.max_resident_requests) {
    auto &s = *e.requests.at(queued);
    s.result.queue_ms = elapsed_ms(s.arrival);
    s.resident = true; // Also makes partial allocations eligible for cleanup.
    bool ok = false;
    try {
      ok = e.activate(s, error);
    } catch (const std::bad_alloc &) {
      error = "Could not allocate admitted request state.";
    }
    progressed = true;
    if (!ok) {
      s.result.status = CpuRequestStatus::failed;
      s.result.error = error;
      e.retire(s);
      return false;
    }
  }
  std::vector<CpuRequestId> prefill, decode;
  for (const auto &[id, s] : e.requests) {
    if (!e.output_ready(*s))
      continue;
    if (s->result.status == CpuRequestStatus::prefill)
      prefill.push_back(id);
    if (s->result.status == CpuRequestStatus::decode)
      decode.push_back(id);
  }
  auto rotate = [](auto &ids, CpuRequestId last) {
    std::sort(ids.begin(), ids.end());
    std::rotate(ids.begin(), std::upper_bound(ids.begin(), ids.end(), last),
                ids.end());
  };
  rotate(prefill, e.last_prefill);
  rotate(decode, e.last_decode);
  if (prefill.empty() && decode.empty())
    return true;
  const bool do_decode =
      !decode.empty() && (prefill.empty() || e.prefer_decode);
  std::vector<CpuRequestId> work;
  if (do_decode) {
    decode.resize(std::min(decode.size(), e.config.max_decode_batch_size));
    work = std::move(decode);
    e.last_decode = work.back();
  } else {
    work.push_back(prefill.front());
    e.last_prefill = work.front();
  }
  e.prefer_decode = !do_decode;
  progressed = true;
  bool ok = false;
  try {
    ok = do_decode ? advance_batch(work, error) : advance(work.front(), error);
  } catch (const std::exception &exception) {
    error = exception.what();
    for (auto id : work) {
      auto &s = *e.requests.at(id);
      s.result.status = CpuRequestStatus::failed;
      s.result.error = error;
    }
  }
  for (auto id : work)
    e.retire(*e.requests.at(id));
  return ok;
}

bool CpuEngine::read_output(CpuRequestId id, std::size_t max_tokens,
                            std::vector<std::int32_t> &output,
                            std::string &error) {
  error.clear();
  if (!impl_->config.max_queued_requests) {
    error = "read_output requires serving mode.";
    return false;
  }
  const auto it = impl_->requests.find(id);
  if (it == impl_->requests.end()) {
    error = "Unknown request ID.";
    return false;
  }
  auto &s = *it->second;
  const auto count = std::min(max_tokens, Impl::confirmed(s) - s.consumed);
  output.insert(output.end(), s.result.output_tokens.begin() + s.consumed,
                s.result.output_tokens.begin() + s.consumed + count);
  s.consumed += count;
  return true;
}
