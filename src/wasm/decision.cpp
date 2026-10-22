// SPDX-License-Identifier: MIT
#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include "qwen35x/runtime/browser_decision.h"
#include "systemone.h"
#include <chrono>
#include <bit>
#ifdef __EMSCRIPTEN__
#include <emscripten/heap.h>
#endif

namespace {
using namespace qwen35x;
using Json = systemone::Json;
using Clock = std::chrono::steady_clock;
struct Session {
  QwenTokenizer tokenizer;
  systemone::Config protocol;
  std::shared_ptr<const CpuModel> model;
  CpuEngineConfig config;
  std::unique_ptr<CpuEngine> engine;
  std::optional<systemone::Request> request;
  CpuRequestId active = 0;
  std::size_t context = 2048, question = 0, steps = 0, evaluated = 0, decode = 0;
  double prefill_ms = 0;
  std::vector<float> logits;
  std::vector<CpuDecodeStage> operations;
  std::vector<CpuPrefillStage> stages;
  Json debug = Json::object();
  Clock::time_point started;
  static bool capture(void *opaque, std::size_t, std::int32_t,
                      const float *values, std::size_t size, std::string &error) {
    auto &s = *static_cast<Session *>(opaque);
    s.logits.clear();
    for (std::size_t i = 0; i < s.request->questions[s.question].names.size(); ++i) {
      const auto id = s.protocol.ids[i];
      if (id < 0 || std::size_t(id) >= size || !std::isfinite(values[id])) {
        error = "Invalid decision logit"; return false;
      }
      s.logits.push_back(values[id]);
    }
    return true;
  }
  static bool capture_selected(void *opaque,std::size_t,std::int32_t,
      const float *values,std::size_t size,std::string &error) {
    auto &s=*static_cast<Session *>(opaque);
    if(size!=s.request->questions[s.question].names.size() ||
       !std::all_of(values,values+size,[](float x){return std::isfinite(x);})) {
      error="Invalid selected decision logits";return false;
    }
    s.logits.assign(values,values+size);return true;
  }
  void cancel() {
    if (active) { engine->cancel(active); engine->release(active); active = 0; }
    request.reset();
  }
  void submit(bool selected_head) {
    CpuRequestSpec spec;
    spec.prompt_tokens = request->rows[question]; spec.max_new_tokens = 1;
    spec.forced_output_tokens = {protocol.ids.front()};
    if(selected_head) {
      spec.selected_logit_tokens.assign(protocol.ids.begin(),
        protocol.ids.begin()+request->questions[question].names.size());
      spec.selected_logits_callback=capture_selected;spec.selected_logits_callback_context=this;
    } else { spec.logits_callback = capture; spec.logits_callback_context = this; }
    std::string error; active = engine->submit(std::move(spec), error);
    systemone::require(active != 0, error);
  }
};
std::unique_ptr<Session> session;
int chunk_size = 32;
bool profiling = false;
Json tuning = Json::object();
std::string output;
const char *reply(const Json &j) { output = j.dump(); return output.c_str(); }
const char *failure(const std::exception &e) {
  if (session) session->cancel();
  return reply({{"error", e.what()}});
}
void apply_tuning(CpuEngineConfig &config) {
  config.linear_prepare_min_tokens = tuning.value("linear_prepare_min_tokens", std::size_t{128});
  config.mlp_prepare_min_tokens = tuning.value("mlp_prepare_min_tokens", std::size_t{8});
  config.balance_attention_rows = tuning.value("balance_attention_rows", std::size_t{0}) != 0;
  config.merge_tail_jobs = tuning.value("merge_tail_jobs", std::size_t{0}) != 0;
  config.gqa4_min_tokens = tuning.value("gqa4_min_tokens",std::size_t{8});
  config.kv_cache_bits = tuning.value("kv_cache_bits", std::size_t{0});
  config.executor_spin_count = tuning.value("executor_spin_count", std::size_t{1024U * 128U * 50U});
}
}
extern "C" {
const char *qev_tune(const char *body) try {
  systemone::require(!session || !session->request, "Finish the request before configuring it");
  auto options = Json::parse(body);
  systemone::require(options.is_object(), "Tuning must be an object");
  for (const auto &[key, value] : options.items()) {
    systemone::require(key == "linear_prepare_min_tokens" || key == "executor_spin_count" || key == "mlp_prepare_min_tokens" || key == "kv_cache_bits" || key == "balance_attention_rows" || key == "merge_tail_jobs" || key == "gqa4_min_tokens" || key == "selected_option_head", "Unknown tuning option: " + key);
    systemone::require(value.is_number_unsigned() && value.get<std::size_t>() <= 67108864, "Invalid tuning value");
  }
  if (options.contains("kv_cache_bits")) {
    const auto bits = options["kv_cache_bits"].get<std::size_t>();
    systemone::require(bits == 0 || bits == 16 || bits == 32, "kv_cache_bits must be 0, 16 or 32");
  }
  for (const auto *key : {"balance_attention_rows", "merge_tail_jobs", "selected_option_head"})
    systemone::require(!options.contains(key) || options[key].get<std::size_t>() <= 1,
                       std::string(key) + " must be 0 or 1");
  tuning = std::move(options);
  if (session) {
    // Recreate only the idle executor/workspaces, retaining immutable weights.
    // Destroy the old pool first so the fixed WASM pthread pool can be reused.
    session->engine.reset(); apply_tuning(session->config);
    std::string error;
    session->engine = CpuEngine::create(session->model, session->config, error);
    systemone::require(bool(session->engine), error);
  }
  return reply(tuning);
} catch (const std::exception &e) { return failure(e); }
const char *qev_options(int chunk, int profile) try {
  systemone::require(!session, "Unload the engine before configuring it");
  systemone::require(chunk >= 4 && chunk <= 1024 && chunk % 4 == 0, "chunk must be a multiple of 4 in 4..1024");
  chunk_size = chunk; profiling = profile != 0;
  return reply({{"chunk", chunk_size}, {"profiling", profiling}});
} catch (const std::exception &e) { return failure(e); }
const char *qev_unload() {
  session.reset(); return reply({{"status", "unloaded"}});
}
const char *qev_load(const char *directory, int threads, int context) try {
  systemone::require(threads >= 1 && threads <= 8, "threads must be 1..8");
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
  systemone::require(threads == 1, "This build supports one thread");
#endif
  systemone::require(context >= 128 && context <= 8192, "context must be 128..8192");
  // Release the old model before loading a new one to bound peak memory.
  session.reset(); auto s = std::make_unique<Session>(); s->context = context;
  std::string error;
  systemone::require(QwenTokenizer::load_from_hf_directory(directory, s->tokenizer, error), error);
  auto cfg = systemone::Config::load(directory, s->tokenizer);
  systemone::require(cfg.has_value(), "qev_config.json is required"); s->protocol = *cfg;
  auto profile = ProfileLoader::load_from_hf_directory(directory, error);
  systemone::require(profile.has_value(), error);
  CpuLoadOptions load; load.model_dir = directory; load.cpu_q4_h128_path = std::string(directory) + "/model.q35h";
  auto model = CpuModel::load(*profile, load, error); systemone::require(bool(model), error);
  systemone::require(std::string(model->weight_format()) == "h128-q4-dot4", "Expected H128/Q4-G32-DOT4 weights");
  CpuEngineConfig config; config.threads = threads; config.max_context = context;
  config.max_resident_requests = 1; config.prefill_chunk_size = chunk_size;
  apply_tuning(config);
  if (profiling) { config.operation_profile = &s->operations; config.prefill_profile = &s->stages; }
  config.max_prefix_bytes = 0;
  s->model = model; s->config = config;
  s->engine = CpuEngine::create(model, config, error); systemone::require(bool(s->engine), error);
  session = std::move(s);
  Json info = {{"status", "ready"}, {"threads", threads}, {"max_context", context},
                {"format", "H128/Q4-G32-DOT4"}, {"chunk", chunk_size}, {"profiling", profiling},
                {"tuning", tuning}, {"kv_type", (config.kv_cache_bits == 16 || (!config.kv_cache_bits && cpu::q8_0_backend_uses_avx2(cpu::Q8_0Backend::auto_select))) ? "f16" : "f32"},
                {"backend", cpu::q8_0_backend_name(cpu::q8_0_resolve_backend(cpu::Q8_0Backend::auto_select))}};
#ifdef __EMSCRIPTEN__
  info["wasm_heap_bytes"] = emscripten_get_heap_size();
#endif
  return reply(info);
} catch (const std::exception &e) { return failure(e); }
const char *qev_begin(const char *body) try {
  systemone::require(bool(session), "Load a model first"); auto &s = *session;
  systemone::require(!s.request, "A decision is already running");
  const auto started = Clock::now();
  auto request = systemone::parse(body, "per_question");
  systemone::prepare(request, s.tokenizer, s.protocol, s.context);
  s.request = std::move(request); s.question = s.steps = s.evaluated = s.decode = 0;
  s.prefill_ms = 0; s.debug = {{"prompt_tokens", s.request->rows}, {"option_logits", Json::array()}, {"option_logit_bits", Json::array()}};
  s.operations.clear(); s.stages.clear();
  s.started = started; s.submit(tuning.value("selected_option_head",std::size_t{1})!=0);
  return reply({{"status", "running"}, {"questions", s.request->questions.size()}, {"input_tokens", s.request->input_tokens}});
} catch (const std::exception &e) { return failure(e); }
const char *qev_step() try {
  systemone::require(session && session->request, "No decision is running"); auto &s = *session;
  std::string error; systemone::require(s.engine->advance(s.active, error), error); ++s.steps;
  CpuRequestResult result; systemone::require(s.engine->result(s.active, result), "Missing engine result");
  systemone::require(result.status != CpuRequestStatus::failed, result.error);
  if (result.status != CpuRequestStatus::complete)
    return reply({{"status", "running"}, {"question", s.question}, {"prefill_tokens", result.prefill_tokens}});
  const auto &q = s.request->questions[s.question];
  s.request->answers[q.id] = systemone::answer(q, s.logits, s.protocol);
  s.debug["option_logits"].push_back(s.logits);
  std::vector<std::uint32_t> bits;
  for (float value : s.logits) bits.push_back(std::bit_cast<std::uint32_t>(value));
  s.debug["option_logit_bits"].push_back(bits);
  s.debug["kv_payload_bytes"].push_back(s.engine->stats().physical_kv_bytes);
  s.evaluated += result.prefill_tokens; s.prefill_ms += result.prefill_ms; s.decode += result.decode_forwards;
  s.engine->release(s.active); s.active = 0;
  if (++s.question < s.request->questions.size()) {
    s.submit(tuning.value("selected_option_head",std::size_t{1})!=0); return reply({{"status", "running"}, {"question", s.question}});
  }
  const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - s.started).count();
  Json answer = {{"status", "complete"}, {"model", "qev-0.8b-H128-Q4-G32-DOT4"},
    {"answers", s.request->answers}, {"usage", {{"input_tokens", s.request->input_tokens}, {"output_tokens", 0}}},
    {"latency_ms", ms}, {"metrics", {{"prefill_ms", s.prefill_ms}, {"evaluated_tokens", s.evaluated},
    {"decode_forwards", s.decode}, {"steps", s.steps}}}};
#ifdef __EMSCRIPTEN__
  answer["metrics"]["wasm_heap_bytes"] = emscripten_get_heap_size();
#endif
  s.request.reset(); return reply(answer);
} catch (const std::exception &e) { return failure(e); }
const char *qev_cancel() {
  if (session) session->cancel(); return reply({{"status", "cancelled"}});
}
const char *qev_debug() {
  if (!session) return reply(Json::object());
  auto diagnostic = session->debug;
  diagnostic["head_readout"]=tuning.value("selected_option_head",std::size_t{1})?"selected":"full";
  const auto bits = session->config.kv_cache_bits;
  diagnostic["kv_cache_bits"] = bits ? bits :
    (cpu::q8_0_backend_uses_avx2(cpu::Q8_0Backend::auto_select) ? 16 : 32);
  if (profiling) {
    diagnostic["operations"] = Json::array(); diagnostic["stages"] = Json::array();
    for (const auto &x : session->operations)
      diagnostic["operations"].push_back({{"kind", x.kind}, {"rows", x.rows}, {"columns", x.columns},
        {"vectors", x.vectors}, {"participants", x.participants}, {"prepare_ms", x.prepare_ms}, {"wall_ms", x.wall_ms},
        {"dispatch_ms", x.dispatch_ms}, {"caller_ms", x.caller_ms}, {"wait_ms", x.wait_ms}});
    for (const auto &x : session->stages)
      diagnostic["stages"].push_back({{"kind", x.kind}, {"tokens", x.tokens}, {"kernel", x.kernel},
        {"participants", x.participants}, {"position", x.position}, {"query_key_pairs", x.query_key_pairs},
        {"query_tile", x.query_tile}, {"kv_tile", x.kv_tile}, {"shared_gqa", x.shared_gqa},
        {"projection_ms", x.projection_ms}, {"prepare_ms", x.prepare_ms},
        {"attention_wall_ms", x.attention_wall_ms}, {"output_ms", x.output_ms},
        {"pack_worker_ms",x.pack_worker_ms},{"qk_worker_ms",x.qk_worker_ms},
        {"softmax_worker_ms",x.softmax_worker_ms},{"pv_worker_ms",x.pv_worker_ms}});
    for (std::size_t i=0; i<session->stages.size(); ++i) {
      const auto &x=session->stages[i]; auto &j=diagnostic["stages"][i];
      j["conv_ms"]=x.conv_ms; j["head_prepare_ms"]=x.head_prepare_ms;
      j["activation_ms"]=x.activation_ms; j["post_ms"]=x.post_ms;
    }
  }
  return reply(diagnostic);
}
}
