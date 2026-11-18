#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include <iostream>
#include <stdexcept>
using namespace qwen35x;

int main(int argc, char **argv) try {
  if (argc != 3) throw std::runtime_error("Expected model directory and H128 weights");
  CpuLoadOptions load;
  load.model_dir = argv[1]; load.cpu_q4_h128_path = argv[2];
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
    }
  }
  std::cout << "Verified full acceptance, each rejection position, stop sequences, seeded sampling and both KV layouts\n";
} catch(const std::exception &e) {std::cerr << e.what() << '\n'; return 1;}
