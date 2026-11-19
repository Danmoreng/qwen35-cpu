#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
using namespace qwen35x;

static void check_mtp_prefixes(const std::shared_ptr<const CpuModel> &model,
                               CpuEngineConfig config) {
  std::string error;
  auto check = [&](bool ok, const char *what) {
    if (!ok) throw std::runtime_error(std::string(what) + " " + error);
  };
  config.prefill_chunk_size = 64;
  config.max_queued_requests = 3;
  config.max_resident_requests = 3;
  config.max_prefill_requests = 2;
  auto engine = CpuEngine::create(model, config, error);
  check(bool(engine), "MTP prefix engine");
  auto run = [&](CpuEngine &engine, const std::vector<CpuRequestSpec> &specs) {
    std::vector<CpuRequestId> ids;
    for (const auto &spec : specs) {
      ids.push_back(engine.submit(spec, error));
      check(ids.back() != 0, "MTP prefix submit");
    }
    std::vector<CpuRequestResult> results(ids.size());
    for (unsigned tick = 0; tick < 300; ++tick) {
      bool progressed = false;
      check(engine.step(progressed, error) && progressed, "MTP prefix step");
      for (std::size_t i = 0; i < ids.size(); ++i) {
        if (!ids[i]) continue;
        check(engine.result(ids[i], results[i]), "MTP prefix result");
        std::vector<std::int32_t> ignored;
        check(engine.read_output(ids[i], config.max_buffered_tokens, ignored, error), "MTP prefix drain");
        if (results[i].status == CpuRequestStatus::complete) {
          check(engine.release(ids[i]), "MTP prefix release");
          ids[i] = 0;
        }
      }
      if (std::all_of(ids.begin(), ids.end(), [](auto id) { return id == 0; }))
        return results;
    }
    throw std::runtime_error("MTP prefix requests did not finish");
  };

  CpuRequestSpec req;
  req.prompt_tokens.resize(128, 2233);
  req.prompt_tokens.back() = 13;
  req.trust_namespace = "explicit-mtp-prefix";
  req.max_new_tokens = 12;
  req.draft_mode = CpuDraftMode::mtp;
  const auto target = engine->register_prefix(req.prompt_tokens, req.trust_namespace, error);
  check(target && !engine->find_prefix(req.prompt_tokens, req.trust_namespace, req.draft_mode),
        "Target-only prefix must miss MTP lookup");
  req.prefix = target;
  check(!engine->submit(req, error), "MTP must reject a target-only snapshot");
  req.prefix.reset();
  const auto mtp = engine->register_prefix(req.prompt_tokens, req.trust_namespace, error, req.draft_mode);
  check(mtp && mtp->size_bytes() > target->size_bytes(), "MTP KV must count against the prefix budget");
  check(engine->find_prefix(req.prompt_tokens, req.trust_namespace) == target &&
        engine->find_prefix(req.prompt_tokens, req.trust_namespace, req.draft_mode) == mtp &&
        !engine->find_prefix(req.prompt_tokens, "other", req.draft_mode), "MTP prefix lookup identity");
  for (float temperature : {0.0f, 0.7f}) {
    req.sampling.temperature = temperature;
    req.sampling.seed = 91;
    req.sampling.top_k = 40;
    req.sampling.top_p = 0.9f;
    for (bool full : {true, false}) {
      auto cold = req;
      if (!full) cold.prompt_tokens.insert(cold.prompt_tokens.end(), {785, 2747, 315, 279});
      const auto expected = run(*engine, {cold}).front();
      const auto before = engine->stats();
      cold.prefix = mtp;
      const auto warm = run(*engine, {cold}).front();
      check(warm.cached_prefix_tokens == 128 && warm.prefill_tokens == (full ? 0 : 4) &&
            warm.output_tokens == expected.output_tokens, "Cold and warm MTP parity");
      check(engine->stats().proposed_tokens > before.proposed_tokens, "MTP did not draft after restore");
    }
  }

  req.sampling.temperature = 0;
  req.trust_namespace = "automatic-mtp-prefix";
  req.register_prefix_tokens = 64;
  auto ordinary = req;
  ordinary.draft_mode = CpuDraftMode::off;
  const auto mixed = run(*engine, {req, ordinary, req});
  check(mixed[0].prefix_written_tokens == 64 && mixed[1].prefix_written_tokens == 64 &&
        mixed[2].cached_prefix_tokens == 64, "MTP and target-only single-flight builds");
  check(mixed[2].output_tokens == mixed[0].output_tokens, "MTP single-flight output parity");
  req.prefix = engine->find_prefix(req.prompt_tokens, req.trust_namespace, req.draft_mode);
  req.register_prefix_tokens = 128;
  const auto extended = run(*engine, {req}).front();
  check(extended.cached_prefix_tokens == 64 && extended.prefix_written_tokens == 64 &&
        extended.output_tokens == mixed[0].output_tokens, "Extend a restored MTP snapshot");
  req.prefix = engine->find_prefix(req.prompt_tokens, req.trust_namespace, req.draft_mode);
  req.register_prefix_tokens = 0;
  req.max_new_tokens = 1;
  const auto short_hit = run(*engine, {req}).front();
  check(short_hit.cached_prefix_tokens == 128 && short_hit.output_tokens.front() == mixed[0].output_tokens.front(),
        "Short request can reuse an MTP snapshot");

  config.max_prefix_bytes = target->size_bytes();
  auto bounded = CpuEngine::create(model, config, error);
  check(bool(bounded), "Budgeted MTP prefix engine");
  req.prefix.reset();
  req.max_new_tokens = 12;
  req.register_prefix_tokens = 128;
  const auto fallback = run(*bounded, {req}).front();
  check(fallback.prefix_written_tokens == 0 && bounded->stats().prefix_build_fallbacks == 1 &&
        fallback.output_tokens == mixed[0].output_tokens, "MTP budget fallback");
}

int main(int argc, char **argv) try {
  if (argc != 3 && argc != 4)
    throw std::runtime_error("Usage: cpu_draft_model_test MODEL_DIR WEIGHTS [MTP_SIDECAR]");
  CpuLoadOptions load;
  load.model_dir = argv[1]; load.cpu_q4_h128_path = argv[2];
  if (argc == 4) load.mtp_weights_path = argv[3];
  std::string error;
  auto require = [&](bool ok, const char *what) {
    if (!ok) throw std::runtime_error(std::string(what) + " " + error);
  };
  auto profile = ProfileLoader::load_from_hf_directory(load.model_dir, error);
  require(bool(profile), "profile");
  auto model = CpuModel::load(*profile, load, error);
  require(bool(model), "weights");
  for (bool paged : {false,true}) {
    CpuEngineConfig cfg; cfg.threads = 3; cfg.max_context = 512; cfg.shared_kv_pages = paged;
    cfg.mtp_max_tokens = 4;
    auto engine = CpuEngine::create(model, cfg, error);
    require(bool(engine), "engine");
    auto run = [&](CpuRequestSpec spec) {
      auto id = engine->submit(std::move(spec), error); require(id != 0, "submit");
      CpuRequestResult r;
      do {
        require(engine->advance(id,error) && engine->result(id,r),"advance");
      } while (r.status != CpuRequestStatus::complete);
      require(engine->release(id),"release");
      return r.output_tokens;
    };
    for (float temperature : {0.0f,0.7f}) {
      CpuRequestSpec req;
      req.prompt_tokens = {785,2747,315,279,3638}; req.max_new_tokens = 12;
      req.sampling.temperature = temperature; req.sampling.top_p = 0.9f;
      req.sampling.top_k = 40; req.sampling.seed = 91; req.sampling.repetition_penalty = 1.07f;
      auto expected = run(req);
      req.draft_mode = CpuDraftMode::input;
      for (int wrong : {0,1,2,3}) {
        req.draft_source_tokens = expected;
        if (wrong) req.draft_source_tokens[wrong] = (expected[wrong]+1)%248320;
        const auto before = engine->stats();
        require(run(req) == expected,"mismatch after verification or rollback");
        const auto after = engine->stats();
        require(after.draft_rounds > before.draft_rounds,"verifier did not run");
        if (wrong >= 2) require(after.replayed_tokens > before.replayed_tokens,"replay missing");
      }
      req.stop_token_sequences = {{expected[2],expected[3]}};
      req.draft_source_tokens = expected;
      const auto stopped = run(req);
      req.draft_mode = CpuDraftMode::off; req.draft_source_tokens.clear();
      require(stopped==run(req),"multi-token stop");

      // A scheduler turn may have space for only one additional verified token.
      auto serving = cfg;
      serving.max_queued_requests = 1;
      serving.max_buffered_tokens = 2;
      auto queued = CpuEngine::create(model, serving, error);
      require(bool(queued),"serving engine");
      req.stop_token_sequences.clear();
      req.draft_mode = CpuDraftMode::input;
      req.draft_source_tokens = expected;
      const auto id = queued->submit(req,error); require(id!=0,"serve submit");
      std::vector<std::int32_t> streamed;
      CpuRequestResult r;
      for (unsigned tick = 0; tick < 100; ++tick) {
        bool progressed = false;
        require(queued->step(progressed,error) && queued->result(id,r),"serve step");
        require(r.output_tokens.size()-streamed.size() <= 2,"backpressure");
        require(queued->read_output(id,1,streamed,error),"read output");
        if (r.status == CpuRequestStatus::complete && streamed.size()==r.output_tokens.size()) break;
      }
      require(streamed==expected && queued->stats().draft_rounds>0,"serving parity");

      req.draft_mode = CpuDraftMode::off;
      req.draft_source_tokens.clear();
      req.max_new_tokens = 48;
      const auto continuation = run(req);
      std::int32_t decoy = 0;
      while (std::find(continuation.begin(), continuation.end(), decoy) != continuation.end())
        ++decoy;
      req.draft_mode = CpuDraftMode::input;
      for (std::size_t i = 0; i < continuation.size(); i += 2) {
        req.draft_source_tokens.push_back(continuation[i]);
        if (i + 1 < continuation.size())
          req.draft_source_tokens.push_back(continuation[i + 1]);
        req.draft_source_tokens.push_back(decoy);
      }
      const auto before_cooldown = engine->stats();
      require(run(req) == continuation, "cooldown output parity");
      require(engine->stats().draft_cooldowns > before_cooldown.draft_cooldowns,
              "cooldown did not run");
      if (!load.mtp_weights_path.empty()) {
        req.draft_source_tokens.clear();
        req.draft_mode = CpuDraftMode::mtp;
        const auto before_mtp = engine->stats();
        require(run(req) == continuation, "MTP output parity");
        const auto after_mtp = engine->stats();
        const auto proposed = after_mtp.proposed_tokens - before_mtp.proposed_tokens;
        const auto accepted = after_mtp.accepted_draft_tokens - before_mtp.accepted_draft_tokens;
        require(proposed > 0 && accepted > 0 && accepted < proposed,
                "MTP did not exercise acceptance and rejection");
        require(after_mtp.mtp_batches_by_proposals.back() > before_mtp.mtp_batches_by_proposals.back(),
                "MTP did not verify four proposals in one forward");
        req.stop_token_sequences = {{continuation[2], continuation[3]}};
        const auto mtp_stopped = run(req);
        req.draft_mode = CpuDraftMode::off;
        require(run(req) == mtp_stopped, "MTP stop sequence parity");

        req.stop_token_sequences.clear();
        req.max_new_tokens = 16;
        req.prompt_tokens.resize(70, 2233);
        const auto batched_expected = run(req);
        req.draft_mode = CpuDraftMode::mtp;
        auto multi_config = cfg;
        multi_config.max_resident_requests = 2;
        multi_config.prefill_chunk_size = 16;
        auto multi = CpuEngine::create(model, multi_config, error);
        require(bool(multi), "MTP multi-request engine");
        auto first = multi->submit(req, error);
        auto second = multi->submit(req, error);
        require(first && second, "MTP multi-request submit");
        const std::array batch{first, second};
        for (auto id : batch) {
          CpuRequestResult result;
          do {
            require(multi->advance(id, error) && multi->result(id, result), "MTP prefill");
          } while (result.status == CpuRequestStatus::prefill);
        }
        require(multi->advance_batch(batch, error), "MTP ordinary batched decode");
        for (auto id : batch) {
          CpuRequestResult result;
          do {
            require(multi->advance(id, error) && multi->result(id, result), "MTP decode after batch");
          } while (result.status != CpuRequestStatus::complete);
          require(result.output_tokens == batched_expected, "MTP switched batch output parity");
        }
        require(multi->stats().proposed_tokens > 0, "MTP did not resume after batch");

        auto mtp_serving = CpuEngine::create(model, serving, error);
        require(bool(mtp_serving), "MTP serving engine");
        const auto stream_id = mtp_serving->submit(req, error);
        require(stream_id != 0, "MTP serving submit");
        std::vector<std::int32_t> output;
        CpuRequestResult result;
        for (unsigned tick = 0; tick < 200; ++tick) {
          bool progressed = false;
          require(mtp_serving->step(progressed, error) && mtp_serving->result(stream_id, result),
                  "MTP serving step");
          require(result.output_tokens.size() - output.size() <= 2, "MTP backpressure");
          require(mtp_serving->read_output(stream_id, 1, output, error), "MTP serving output");
          if (result.status == CpuRequestStatus::complete && output.size() == result.output_tokens.size())
            break;
        }
        const auto serving_stats = mtp_serving->stats();
        require(output == batched_expected && serving_stats.proposed_tokens > 0,
                "MTP streaming parity");
        require(serving_stats.physical_kv_bytes == 0, "MTP serving KV was retained after retirement");
      }
    }
    if (!load.mtp_weights_path.empty()) check_mtp_prefixes(model, cfg);
  }
  std::cout << "Verified acceptance, rollback, cooldown, stop sequences, seeded sampling and both KV layouts\n";
  if (!load.mtp_weights_path.empty())
    std::cout << "Verified MTP sampling, multi-request fallback, streaming, KV retirement and prefix reuse\n";
} catch(const std::exception &e) {std::cerr << e.what() << '\n'; return 1;}
