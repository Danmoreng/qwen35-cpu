#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include "qwen35x/tokenizer/tokenizer.h"
#include "httplib.h"
#include <nlohmann/json.hpp>
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
using Json = nlohmann::json;
namespace {
constexpr const char *model_name = "Qwen3.5-0.8B-H128-Q4-G32-DOT4";
struct Reply { int status; Json body; };
Reply failure(int status, const std::string &message) {
  return {status, {{"error", {{"message", message}, {"type", "request_error"}}}}};
}
struct Job {
  std::string prompt;
  CpuRequestSpec spec;
  std::promise<Reply> promise;
  std::atomic<bool> cancelled{false};
  std::size_t prompt_count = 0, limit = 0;
};

// Only this worker touches the tokenizer and CpuEngine. HTTP workers wait on
// futures; they never nest inference thread pools or call engine methods.
class Service {
  struct Active { CpuRequestId id; std::shared_ptr<Job> job; std::vector<int32_t> output; };
  QwenTokenizer tokenizer_;
  std::unique_ptr<CpuEngine> engine_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::shared_ptr<Job>> pending_;
  bool stopping_ = false;
  std::size_t outstanding_ = 0;
  std::thread worker_;
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
            std::string text;
            if (!tokenizer_.decode(it->output, text, error)) reply = failure(500, error);
            else reply = {200, {
              {"id", "cmpl-" + std::to_string(it->id)}, {"object", "text_completion"},
              {"created", std::time(nullptr)}, {"model", model_name},
              {"choices", Json::array({{{"text", text}, {"index", 0}, {"logprobs", nullptr},
                {"finish_reason", result.output_tokens.size() >= it->job->limit ? "length" : "stop"}}})},
              {"usage", {{"prompt_tokens", it->job->prompt_count},
                {"completion_tokens", result.output_tokens.size()},
                {"total_tokens", it->job->prompt_count + result.output_tokens.size()}}}}};
          }
          engine_->release(it->id);
          finish(it->job, std::move(reply)); it = active.erase(it);
        } else ++it;
      }
      if (stopping && active.empty()) return;
    }
  }
public:
  Service(const CpuLoadOptions &load, const CpuEngineConfig &config) {
    std::string error;
    if (!QwenTokenizer::load_from_hf_directory(load.model_dir, tokenizer_, error)) throw std::runtime_error(error);
    auto profile = ProfileLoader::load_from_hf_directory(load.model_dir, error);
    if (!profile) throw std::runtime_error(error);
    auto model = CpuModel::load(*profile, load, error);
    if (!model) throw std::runtime_error(error);
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
};

std::shared_ptr<Job> parse(const std::string &body, std::size_t context) {
  const auto j = Json::parse(body);
  if (!j.is_object()) throw std::runtime_error("Expected JSON object");
  for (const auto &[key, value] : j.items())
    if (key != "model" && key != "prompt" && key != "max_tokens" && key != "temperature" &&
        key != "stream" && key != "n" && key != "prefix_tokens" && key != "top_p" &&
        key != "top_k" && key != "seed" && key != "repetition_penalty")
      throw std::runtime_error("Unsupported field: " + key);
  if (j.contains("model") && j.at("model").get<std::string>() != model_name)
    throw std::runtime_error("Unknown model");
  if (j.value("stream", false)) throw std::runtime_error("Streaming is not supported yet");
  if (j.contains("n") && j.at("n") != 1) throw std::runtime_error("Only n=1 is supported");
  auto job = std::make_shared<Job>();
  job->prompt = j.at("prompt").get<std::string>();
  if (job->prompt.empty() || job->prompt.size() > 65536) throw std::runtime_error("Prompt must contain 1..65536 bytes");
  auto integer = [&](const char *name, std::int64_t fallback) {
    if (!j.contains(name)) return fallback;
    if (!j.at(name).is_number_integer()) throw std::runtime_error(std::string(name) + " must be an integer");
    if (j.at(name).is_number_unsigned() && j.at(name).get<std::uint64_t>() >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
      throw std::runtime_error(std::string(name) + " is out of range");
    return j.at(name).get<std::int64_t>();
  };
  const auto max_tokens = integer("max_tokens", 128), prefix = integer("prefix_tokens", 0);
  if (max_tokens < 1 || max_tokens > static_cast<std::int64_t>(context) || prefix < 0 || prefix > static_cast<std::int64_t>(context))
    throw std::runtime_error("Token limit out of range");
  job->spec.max_new_tokens = static_cast<std::size_t>(max_tokens);
  job->spec.register_prefix_tokens = static_cast<std::size_t>(prefix);
  job->spec.trust_namespace = "local-server";
  auto real = [&](const char *name, float fallback) {
    if (!j.contains(name)) return fallback;
    if (!j.at(name).is_number()) throw std::runtime_error(std::string(name) + " must be a number");
    const double value = j.at(name).get<double>();
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
      throw std::runtime_error(std::string(name) + " must be finite and representable");
    return static_cast<float>(value);
  };
  auto &sampling = job->spec.sampling;
  sampling.temperature = real("temperature", 0);
  sampling.top_p = real("top_p", 0.8f);
  sampling.repetition_penalty = real("repetition_penalty", 1);
  const auto top_k = integer("top_k", 20), seed = integer("seed", -1);
  if (sampling.temperature < 0 || sampling.top_p <= 0 || sampling.top_p > 1 ||
      sampling.repetition_penalty < 1 || top_k < 0 || top_k > 248320 ||
      seed < -1 || seed > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("Sampling parameter out of range");
  sampling.top_k = static_cast<int>(top_k);
  sampling.seed = seed;
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
  int port = 8080;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      std::cout << "Qwen3.5 CPU HTTP server\n--model-dir DIR [--weights FILE] [--threads N]\n"
        "[--host 127.0.0.1] [--port 8080] [--max-context 8192] [--residents 16]\n"
        "GET /health, GET /v1/models, POST /v1/completions (sampling, non-streaming)\n";
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
  Service service(load, config);
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
  http.Post("/v1/completions", [&](const httplib::Request &request, httplib::Response &response) {
    if (!authorized(request, response)) return;
    std::shared_ptr<Job> job;
    try { job = parse(request.body, config.max_context); }
    catch (const std::exception &e) { send(response, failure(400, e.what())); return; }
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
  });
  std::cout << "Listening on http://" << host << ':' << port << " (" << config.threads << " inference threads)" << std::endl;
  if (!http.listen(host, port)) throw std::runtime_error("Cannot bind HTTP listener");
  return 0;
} catch (const std::exception &e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
