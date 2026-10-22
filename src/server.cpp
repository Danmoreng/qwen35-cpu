#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include "qwen35x/tokenizer/tokenizer.h"
#include "httplib.h"
#include "systemone.h"
#include "server_protocol.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>

using namespace qwen35x;
using Json = systemone::Json;
namespace {
const char *model_name = "Qwen3.5-0.8B-H128-Q4-G32-DOT4";
struct Reply { int status; Json body; };
Reply failure(int status, const std::string &message) {
  return {status, {{"error", {{"message", message}, {"type", status < 500 ? "invalid_request_error" : "server_error"}, {"param", nullptr}, {"code", nullptr}}}}};
}
struct Job : server::Request {
  std::promise<Reply> promise;
  std::atomic<bool> cancelled{false};
  std::size_t prompt_count = 0, limit = 0;
  std::optional<systemone::Request> decision;
  std::size_t question = 0;
  std::vector<std::int32_t> readout_ids;
  std::vector<float> option_logits;
  std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
  static bool capture(void *opaque, std::size_t, std::int32_t, const float *logits, std::size_t n, std::string &error) {
    auto &job = *static_cast<Job *>(opaque);
    job.option_logits.clear();
    for (auto id : job.readout_ids) {
      if (id < 0 || std::size_t(id) >= n || !std::isfinite(logits[id])) { error = "Invalid decision logit"; return false; }
      job.option_logits.push_back(logits[id]);
    }
    return true;
  }
  static bool capture_selected(void *opaque, std::size_t, std::int32_t,
      const float *values, std::size_t n, std::string &error) {
    auto &job = *static_cast<Job *>(opaque);
    if (n != job.readout_ids.size() ||
        !std::all_of(values, values + n, [](float value) { return std::isfinite(value); })) {
      error = "Invalid selected decision logits"; return false;
    }
    job.option_logits.assign(values, values + n);
    return true;
  }
};

// Only this worker touches the tokenizer and CpuEngine. HTTP workers wait on
// futures; they never nest inference thread pools or call engine methods.
class Service {
  struct Active { CpuRequestId id; std::shared_ptr<Job> job; std::vector<int32_t> output; };
  QwenTokenizer tokenizer_;
  std::unique_ptr<CpuEngine> engine_;
  std::optional<systemone::Config> decision_;
  std::size_t max_context_;
  bool selected_decision_head_ = false;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::shared_ptr<Job>> pending_;
  bool stopping_ = false;
  std::size_t outstanding_ = 0;
  std::thread worker_;
  CpuRequestId submit_decision(const std::shared_ptr<Job> &job, std::string &error) {
    const auto count = job->decision->questions[job->question].names.size();
    job->readout_ids.assign(decision_->ids.begin(), decision_->ids.begin() + count);
    job->option_logits.clear();
    CpuRequestSpec spec;
    spec.prompt_tokens = job->decision->rows[job->question]; spec.max_new_tokens = 1;
    spec.forced_output_tokens = {decision_->ids.front()};
    if (selected_decision_head_) {
      spec.selected_logit_tokens = job->readout_ids;
      spec.selected_logits_callback = Job::capture_selected;
      spec.selected_logits_callback_context = job.get();
    } else {
      spec.logits_callback = Job::capture; spec.logits_callback_context = job.get();
    }
    return engine_->submit(std::move(spec), error);
  }
  void finish(const std::shared_ptr<Job> &job, Reply reply) {
    job->promise.set_value(std::move(reply));
    std::lock_guard lock(mutex_); --outstanding_;
  }
  void run() {
    std::vector<Active> active;
    for (;;) {
      std::deque<std::shared_ptr<Job>> pending;
      bool stopping;
      {
        std::unique_lock lock(mutex_);
        if (active.empty() && pending_.empty() && !stopping_)
          wake_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
        stopping = stopping_;
        pending.swap(pending_);
      }
      for (auto &job : pending) {
        if (stopping || job->cancelled) { finish(job, failure(499, "Request cancelled")); continue; }
        try {
          std::string error;
          if (job->decision) {
            try { systemone::prepare(*job->decision, tokenizer_, *decision_, max_context_); }
            catch (const std::exception &e) { finish(job, failure(400, e.what())); continue; }
            auto id = submit_decision(job, error);
            if (!id) { finish(job, failure(400, error)); continue; }
            active.push_back({id, job, {}}); continue;
          }
          if (!tokenizer_.encode(job->prompt, job->spec.prompt_tokens, error)) {
            finish(job, failure(400, error)); continue;
          }
          job->prompt_count = job->spec.prompt_tokens.size();
          job->limit = job->spec.max_new_tokens;
          for (const auto *token : {"<|im_end|>", "<|endoftext|>"})
            if (auto id = tokenizer_.token_to_id(token)) job->spec.stop_token_ids.push_back(*id);
          auto id = engine_->submit(std::move(job->spec), error);
          if (!id) { finish(job, failure(400, error)); continue; }
          active.push_back({id, job, {}});
        } catch (const std::exception &e) { finish(job, failure(500, e.what())); }
      }
      for (auto &item : active)
        if (stopping || item.job->cancelled) engine_->cancel(item.id);
      bool progressed = false;
      std::string error;
      if (!active.empty() && !engine_->step(progressed, error)) {
        for (auto &item : active) {
          engine_->cancel(item.id); engine_->release(item.id);
          finish(item.job, failure(500, error));
        }
        active.clear();
      }
      for (auto it = active.begin(); it != active.end();) {
        CpuRequestResult result;
        if (!engine_->read_output(it->id, 256, it->output, error) ||
            !engine_->result(it->id, result)) {
          engine_->cancel(it->id); engine_->release(it->id);
          finish(it->job, failure(500, error)); it = active.erase(it); continue;
        }
        if (result.status == CpuRequestStatus::complete || result.status == CpuRequestStatus::failed ||
            result.status == CpuRequestStatus::cancelled) {
          Reply reply = failure(500, result.error);
          if (result.status == CpuRequestStatus::cancelled) reply = failure(499, "Request cancelled");
          if (result.status == CpuRequestStatus::complete) {
            if (it->job->decision) {
              auto &job = it->job; auto &decision = *job->decision;
              try {
                const auto &q = decision.questions[job->question];
                decision.answers[q.id] = systemone::answer(q, job->option_logits, *decision_);
                if (++job->question < decision.questions.size() && !job->cancelled && !stopping) {
                  engine_->release(it->id);
                  it->id = submit_decision(job, error); it->output.clear();
                  if (!it->id) { finish(job, failure(500, error)); it = active.erase(it); }
                  else ++it;
                  continue;
                }
                const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - job->started).count();
                reply = job->cancelled || stopping ? failure(499, "Request cancelled") : Reply{200, {
                  {"model", model_name}, {"answers", decision.answers},
                  {"usage", {{"input_tokens", decision.input_tokens}, {"output_tokens", 0}}}, {"latency_ms", elapsed}}};
              } catch (const std::exception &e) { reply = failure(500, e.what()); }
            } else {
            std::string text;
            if (!tokenizer_.decode(it->output, text, error)) reply = failure(500, error);
            else reply = {200, server::completion(it->job->api, model_name, it->id,
              std::time(nullptr), text, result.output_tokens.size() >= it->job->limit ? "length" : "stop",
              it->job->prompt_count, result.output_tokens.size())};
            }
          }
          engine_->release(it->id);
          finish(it->job, std::move(reply)); it = active.erase(it);
        } else ++it;
      }
      if (stopping && active.empty()) return;
    }
  }
public:
  Service(const CpuLoadOptions &load, const CpuEngineConfig &config, bool auto_decision_head) : max_context_(config.max_context) {
    std::string error;
    if (!QwenTokenizer::load_from_hf_directory(load.model_dir, tokenizer_, error)) throw std::runtime_error(error);
    decision_ = systemone::Config::load(load.model_dir, tokenizer_);
    auto profile = ProfileLoader::load_from_hf_directory(load.model_dir, error);
    if (!profile) throw std::runtime_error(error);
    auto model = CpuModel::load(*profile, load, error);
    if (!model) throw std::runtime_error(error);
    if (std::string(model->weight_format()).starts_with("gguf-"))
      model_name = "Qwen3.5-0.8B-GGUF";
    if (decision_) model_name = "qev-0.8b-H128-Q4-G32-DOT4";
    selected_decision_head_ = auto_decision_head && model->supports_selected_logits();
    if (decision_) std::cout << "System One readout: " << (selected_decision_head_ ? "selected" : "full") << '\n';
    engine_ = CpuEngine::create(model, config, error);
    if (!engine_) throw std::runtime_error(error);
    worker_ = std::thread([this] { run(); });
  }
  ~Service() {
    { std::lock_guard lock(mutex_); stopping_ = true; }
    wake_.notify_one(); worker_.join();
  }
  bool enqueue(std::shared_ptr<Job> job) {
    std::lock_guard lock(mutex_);
    if (stopping_ || outstanding_ >= 64) return false;
    ++outstanding_; pending_.push_back(std::move(job)); wake_.notify_one(); return true;
  }
  bool decisions_enabled() const { return decision_.has_value(); }
};

std::shared_ptr<Job> parse(const std::string &body, std::size_t context, server::Api api) {
  auto job = std::make_shared<Job>();
  static_cast<server::Request &>(*job) = server::parse(body, context, model_name, api);
  return job;
}
}

int main(int argc, char **argv) try {
  CpuLoadOptions load;
  load.model_dir = "models/qwen3.5-0.8b";
  CpuEngineConfig config;
  config.max_resident_requests = 16; config.max_queued_requests = 64;
  config.max_decode_batch_size = 16; config.shared_kv_pages = true;
  std::string host = "127.0.0.1";
  std::string decision_encoding = "joint";
  bool auto_decision_head = true;
  int port = 8080;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      std::cout << "Qwen3.5 CPU HTTP server\n--model-dir DIR [--weights FILE] [--threads N]\n"
        "[--host 127.0.0.1] [--port 8080] [--max-context 8192] [--residents 16]\n"
        "[--decision-encoding joint|per_question] (qev_config.json required)\n"
        "[--decision-head auto|full] [--prefill-chunk N] (1..2048; default adaptive)\n"
        "GET /health, GET /v1/models, POST /v1/completions, POST /v1/chat/completions, POST /v1/systemone\n";
      return 0;
    }
    if (++i == argc) throw std::runtime_error("Missing value: " + key);
    const std::string value = argv[i];
    auto number = [&]() {
      std::size_t used; const auto n = std::stoll(value, &used);
      if (used != value.size() || n < 1 || n > 65535) throw std::runtime_error("Invalid value: " + key);
      return static_cast<int>(n);
    };
    if (key == "--model-dir") load.model_dir = value;
    else if (key == "--weights") load.cpu_q4_h128_path = value;
    else if (key == "--threads") config.threads = number();
    else if (key == "--host") host = value;
    else if (key == "--port") port = number();
    else if (key == "--decision-encoding") {
      if (value != "joint" && value != "per_question") throw std::runtime_error("Invalid decision encoding");
      decision_encoding = value;
    }
    else if (key == "--decision-head") {
      if (value != "auto" && value != "full") throw std::runtime_error("Invalid decision head");
      auto_decision_head = value == "auto";
    }
    else if (key == "--prefill-chunk") {
      config.prefill_chunk_size = number();
      if (config.prefill_chunk_size > 2048) throw std::runtime_error("Prefill chunk must be <=2048");
    }
    else if (key == "--max-context") config.max_context = number();
    else if (key == "--residents") {
      config.max_resident_requests = number();
      if (config.max_resident_requests > 64) throw std::runtime_error("Residents must be <=64");
      config.max_decode_batch_size = config.max_resident_requests;
    } else throw std::runtime_error("Unknown option: " + key);
  }
  if (load.cpu_q4_h128_path.empty()) load.cpu_q4_h128_path = load.model_dir + "/model.q35h";
  const char *key_env = std::getenv("QWEN35_API_KEY");
  const std::string api_key = key_env ? key_env : "";
  if (host != "127.0.0.1" && host != "::1" && host != "localhost" && api_key.empty())
    throw std::runtime_error("Set QWEN35_API_KEY before binding a non-loopback interface");
  Service service(load, config, auto_decision_head);
  httplib::Server http;
  http.new_task_queue = [] { return new httplib::ThreadPool(32, 32, 64); };
  http.set_payload_max_length(262144);
  http.set_read_timeout(10, 0); http.set_write_timeout(10, 0); http.set_keep_alive_max_count(1);
  auto send = [](httplib::Response &response, const Reply &reply) {
    response.status = reply.status; response.set_content(reply.body.dump(), "application/json");
  };
  auto authorized = [&](const httplib::Request &request, httplib::Response &response) {
    if (api_key.empty() || request.get_header_value("Authorization") == "Bearer " + api_key) return true;
    send(response, failure(401, "Unauthorized")); return false;
  };
  http.Get("/health", [&](const auto &request, auto &response) {
    if (authorized(request, response)) send(response, {200, {{"status", "ok"}}});
  });
  http.Get("/v1/models", [&](const auto &request, auto &response) {
    if (authorized(request, response)) send(response, {200, {{"object", "list"},
      {"data", Json::array({{{"id", model_name}, {"object", "model"}, {"owned_by", "local"}}})}}});
  });
  auto dispatch = [&](const std::shared_ptr<Job> &job, const httplib::Request &request, httplib::Response &response) {
    auto result = job->promise.get_future();
    if (!service.enqueue(job)) { send(response, failure(429, "Request capacity exhausted")); return; }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
    while (result.wait_for(std::chrono::milliseconds(50)) != std::future_status::ready) {
      if (request.is_connection_closed()) { job->cancelled = true; return; }
      if (std::chrono::steady_clock::now() >= deadline) {
        job->cancelled = true; send(response, failure(504, "Request deadline exceeded")); return;
      }
    }
    send(response, result.get());
  };
  http.Post("/v1/completions", [&](const httplib::Request &request, httplib::Response &response) {
    if (!authorized(request, response)) return;
    std::shared_ptr<Job> job;
    try { job = parse(request.body, config.max_context, server::Api::completion); }
    catch (const std::exception &e) { send(response, failure(400, e.what())); return; }
    dispatch(job, request, response);
  });
  http.Post("/v1/chat/completions", [&](const httplib::Request &request, httplib::Response &response) {
    if (!authorized(request, response)) return;
    std::shared_ptr<Job> job;
    try { job = parse(request.body, config.max_context, server::Api::chat); }
    catch (const std::exception &e) { send(response, failure(400, e.what())); return; }
    dispatch(job, request, response);
  });
  http.Post("/v1/systemone", [&](const httplib::Request &request, httplib::Response &response) {
    if (!authorized(request, response)) return;
    if (!service.decisions_enabled()) { send(response, failure(400, "This model has no supported qev_config.json")); return; }
    auto job = std::make_shared<Job>();
    try { job->decision = systemone::parse(request.body, decision_encoding); }
    catch (const std::exception &e) { send(response, failure(400, e.what())); return; }
    dispatch(job, request, response);
  });
  std::cout << "Listening on http://" << host << ':' << port << " (" << config.threads << " inference threads)" << std::endl;
  if (!http.listen(host, port)) throw std::runtime_error("Cannot bind HTTP listener");
  return 0;
} catch (const std::exception &e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
