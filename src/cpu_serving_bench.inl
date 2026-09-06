// Included after benchmark helpers; uses the same sequential runner as static
// tests.
int serving_benchmark(CpuEngine &engine, const CpuRequestSpec &spec,
                      const std::shared_ptr<const CpuPrefix> &prefix,
                      std::size_t count, int vocab, bool identical,
                      double arrival_gap_ms, double load_ms, double build_ms,
                      const std::string &profile_path) {
  std::string error;
  auto check = [&](bool ok) {
    if (!ok)
      throw std::runtime_error(error);
  };
  std::vector<CpuRequestId> ids(count);
  std::vector<CpuRequestResult> results(count);
  std::vector<std::vector<std::int32_t>> outputs(count);
  std::vector<double> ttft(count, -1), last(count, -1), submitted(count), itl,
      ticks;
  std::size_t next = 0, finished = 0;
  std::size_t peak_kv_bytes = 0;
  std::vector<bool> done(count);
  const auto start = Clock::now();
  while (finished < count) {
    double now = ms(start);
    while (next < count && now >= next * arrival_gap_ms) {
      auto request = spec;
      request.prefix = prefix;
      request.trust_namespace = "benchmark";
      if (!identical)
        request.prompt_tokens.back() =
            (request.prompt_tokens.back() + static_cast<int>(next)) % vocab;
      submitted[next] = ms(start);
      ids[next] = engine.submit(std::move(request), error);
      check(ids[next] != 0);
      ++next;
    }
    bool progressed = false;
    const auto tick = Clock::now();
    check(engine.step(progressed, error));
    peak_kv_bytes=std::max(peak_kv_bytes,engine.stats().physical_kv_bytes);
    if (progressed)
      ticks.push_back(ms(tick));
    now = ms(start);
    for (std::size_t r = 0; r < next; ++r)
      if (!done[r]) {
        const auto before = outputs[r].size();
        check(
            engine.read_output(ids[r], spec.max_new_tokens, outputs[r], error));
        if (outputs[r].size() != before) {
          if (last[r] < 0)
            ttft[r] = now - r * arrival_gap_ms;
          else
            itl.push_back(now - last[r]);
          last[r] = now;
        }
        check(engine.result(ids[r], results[r]));
        if (results[r].status == CpuRequestStatus::failed)
          throw std::runtime_error(results[r].error);
        if (results[r].status == CpuRequestStatus::complete) {
          check(outputs[r] == results[r].output_tokens);
          done[r] = true;
          ++finished;
          engine.release(ids[r]);
        }
      }
    if (!progressed && next == count && finished < count)
      throw std::runtime_error("Serving benchmark stalled");
    if (!progressed && next < count)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const double elapsed = ms(start);
  std::size_t delivered = 0, forwards = 0, newly = 0, cached = 0;
  double restore = 0;
  std::vector<double> queue;
  for (std::size_t r = 0; r < count; ++r) {
    delivered += outputs[r].size();
    forwards += results[r].decode_forwards;
    newly += results[r].prefill_tokens;
    cached += results[r].cached_prefix_tokens;
    restore += results[r].prefix_restore_ms;
    queue.push_back(results[r].queue_ms + submitted[r] - r * arrival_gap_ms);
  }
  auto percentile = [](std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[static_cast<std::size_t>(p * (v.size() - 1))];
  };
  const auto stats = engine.stats();
  std::ofstream out(profile_path);
  if (!out)
    throw std::runtime_error("Cannot write serving profile");
  out << std::setprecision(12)
      << "{\"cpu_serving\":true,\"cpu_batch\":" << count
      << ",\"arrival_gap_ms\":" << arrival_gap_ms
      << ",\"prompt_tokens\":" << spec.prompt_tokens.size() * count
      << ",\"generated_tokens\":" << delivered
      << ",\"decode_forwards\":" << forwards << ",\"load_time_ms\":" << load_ms
      << ",\"prefix_build_time_ms\":" << build_ms
      << ",\"new_prefill_tokens\":" << newly
      << ",\"cached_prefix_tokens\":" << cached
      << ",\"prefix_cache_restore_time_ms\":" << restore
      << ",\"prefix_cache_bytes\":" << (prefix ? prefix->size_bytes() : 0)
      << ",\"serving_time_ms\":" << elapsed
      << ",\"tokens_per_second\":" << delivered * 1000.0 / elapsed
      << ",\"ttft_p50_ms\":" << percentile(ttft, .5)
      << ",\"ttft_p95_ms\":" << percentile(ttft, .95)
      << ",\"itl_p50_ms\":" << percentile(itl, .5)
      << ",\"itl_p95_ms\":" << percentile(itl, .95)
      << ",\"itl_p99_ms\":" << percentile(itl, .99)
      << ",\"queue_p50_ms\":" << percentile(queue, .5)
      << ",\"queue_p95_ms\":" << percentile(queue, .95)
      << ",\"tick_p50_ms\":" << percentile(ticks, .5)
      << ",\"peak_rss_bytes\":" << peak_rss()
      << ",\"physical_kv_bytes\":" << peak_kv_bytes
      << ",\"batch_ticks\":" << stats.batch_ticks
      << ",\"batch_rows\":" << stats.batch_rows
      << ",\"state_batches\":" << stats.state_batches
      << ",\"prefix_builds\":" << stats.prefix_builds
      << ",\"prefix_build_fallbacks\":" << stats.prefix_build_fallbacks
      << ",\"requests\":[";
  for (std::size_t r = 0; r < count; ++r) {
    if (r)
      out << ',';
    out << '[';
    for (std::size_t t = 0; t < outputs[r].size(); ++t) {
      if (t)
        out << ',';
      out << outputs[r][t];
    }
    out << ']';
  }
  out << "]}\n";
  if (!out)
    throw std::runtime_error("Failed writing serving profile");
  std::cout << "Serving delivered=" << delivered * 1000.0 / elapsed
            << " token/s TTFT p50=" << percentile(ttft, .5) << " ms\n";
  return 0;
}
