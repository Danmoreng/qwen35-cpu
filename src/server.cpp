#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include "qwen35x/tokenizer/tokenizer.h"
#include "httplib.h"
#include "systemone.h"
#include "server_protocol.h"
#include "server_classifier.h"
#include "server_anthropic.h"
#include "server_constraint.h"
#include "server_stream.h"
#include "server_responses.h"
#include "server_batch_db.h"
#include "server_batch_tuning.h"
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
using Reply = server::Reply;
Reply failure(int status, const std::string &message) {
  return {status, {{"error", {{"message", message}, {"type", status < 500 ? "invalid_request_error" : "server_error"}, {"param", nullptr}, {"code", nullptr}}}}};
}
struct Job : server::Request {
  std::promise<Reply> promise, admitted;
  bool admission_sent = false, count_only = false;
  server::StreamQueue events;
  server::Utf8Decoder utf8;
  std::shared_ptr<server::JsonConstraint> constraint;
  std::unique_ptr<server::StopMatcher> stop_matcher;
  std::string text;
  std::shared_ptr<server::ResponseRecord> response_record;
  Json truncation_messages;
  CpuRequestId id = 0;
  std::time_t created = std::time(nullptr);
  void accept(std::string_view value) {
    text.append(value);
    if (response_record && !tool_mode && !value.empty()) response_record->delta(std::string(value));
    if (stream && !tool_mode) events.push(std::string(value));
  }
  void ready(Reply reply) {
    if (!admission_sent) { admitted.set_value(std::move(reply)); admission_sent = true; }
  }
  std::atomic<bool> cancelled{false};
  std::size_t prompt_count = 0, limit = 0;
  std::optional<systemone::Request> decision;
  bool classify = false;
  std::size_t question = 0;
  std::size_t decision_prefix = 0, cached_tokens = 0;
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
  std::shared_ptr<const server::TokenVocabulary> vocabulary_;
  std::unique_ptr<CpuEngine> engine_;
  std::optional<systemone::Config> decision_;
  server::Classifier classifier_;
  std::size_t max_context_;
  server::BatchTuning tuning_;
  bool selected_decision_head_ = false;
  bool automatic_prefix_ = true;
  std::size_t prefix_budget_ = 0;
  bool mtp_enabled_ = false;
  Json cache_snapshot_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::shared_ptr<Job>> pending_;
  bool stopping_ = false;
  std::size_t outstanding_ = 0;
  std::thread worker_;
  void configure_cache(CpuRequestSpec &spec, std::size_t candidate) {
    if (!automatic_prefix_ || !prefix_budget_) return;
    spec.prefix = engine_->find_prefix(spec.prompt_tokens, spec.trust_namespace, spec.draft_mode);
    if (candidate >= 32 && (!spec.prefix || spec.prefix->token_count() < candidate))
      spec.register_prefix_tokens = candidate;
  }
  Json cache_stats() const {
    const auto s = engine_->stats();
    Json result = {{"enabled", automatic_prefix_}, {"budget_bytes", prefix_budget_},
      {"shared_prefill_ticks", s.shared_prefill_ticks}, {"shared_prefill_requests", s.shared_prefill_requests},
      {"entries", s.prefix_entries}, {"bytes", s.prefix_bytes}, {"hits", s.prefix_hits},
      {"cached_tokens", s.prefix_cached_tokens}, {"builds", s.prefix_builds},
      {"evictions", s.prefix_evictions}, {"budget_fallbacks", s.prefix_build_fallbacks}};
    if (mtp_enabled_) {
      result["draft"] = {{"proposed_tokens", s.proposed_tokens},
        {"accepted_tokens", s.accepted_draft_tokens},
        {"mtp_batches_by_proposals", s.mtp_batches_by_proposals},
        {"mtp_prefill_ms", s.mtp_prefill_ms}, {"mtp_draft_ms", s.mtp_draft_ms},
        {"mtp_verify_ms", s.mtp_verify_ms}, {"mtp_repair_ms", s.mtp_repair_ms}};
    }
    return result;
  }
  void configure_readout(CpuRequestSpec &spec, const std::shared_ptr<Job> &job) {
    spec.max_new_tokens = 1;
    spec.forced_output_tokens = {job->readout_ids.front()};
    if (selected_decision_head_) {
      spec.selected_logit_tokens = job->readout_ids;
      spec.selected_logits_callback = Job::capture_selected;
      spec.selected_logits_callback_context = job.get();
    } else {
      spec.logits_callback = Job::capture; spec.logits_callback_context = job.get();
    }
  }
  CpuRequestId submit_decision(const std::shared_ptr<Job> &job, std::string &error) {
    const auto count = job->decision->questions[job->question].names.size();
    job->readout_ids.assign(decision_->ids.begin(), decision_->ids.begin() + count);
    job->option_logits.clear();
    CpuRequestSpec spec;
    spec.prompt_tokens = job->decision->rows[job->question];
    spec.trust_namespace = "local-server-qev";
    configure_cache(spec, job->decision_prefix);
    configure_readout(spec, job);
    return engine_->submit(std::move(spec), error);
  }
  void finish(const std::shared_ptr<Job> &job, Reply reply) {
    {
      const auto snapshot = cache_stats();
      std::lock_guard lock(mutex_); cache_snapshot_ = snapshot; --outstanding_;
    }
    if (job->response_record) job->response_record->finish(reply);
    job->ready(reply);
    if (job->stream) job->events.finish(reply);
    job->promise.set_value(std::move(reply));
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
            const auto &rows = job->decision->rows;
            job->decision_prefix = rows.front().size();
            for (const auto &row : rows) {
              const auto n = std::min(job->decision_prefix, row.size());
              std::size_t common = 0;
              while (common < n && rows.front()[common] == row[common]) ++common;
              job->decision_prefix = common;
            }
            auto id = submit_decision(job, error);
            if (!id) { finish(job, failure(400, error)); continue; }
            active.push_back({id, job, {}}); continue;
          }
          if (!tokenizer_.encode(job->prompt, job->spec.prompt_tokens, error)) {
            finish(job, failure(400, error)); continue;
          }
          if (job->classify) {
            job->readout_ids = classifier_.tokens;
            configure_readout(job->spec, job);
            job->spec.trust_namespace = "local-server-classifier";
          }
          if (job->truncation_messages.is_array()) {
            const auto reserve = job->count_only ? 0 : job->spec.max_new_tokens - 1;
            const auto budget = max_context_ - reserve;
            if (job->spec.prompt_tokens.size() > budget) {
              auto &messages = job->truncation_messages;
              // Drop whole old turns, including their tool calls and results. Keep
              // leading instructions and at least the most recent user question.
              std::vector<std::size_t> turns;
              for (std::size_t i=0; i<messages.size(); ++i)
                if (messages[i]["role"]=="user")
                  turns.push_back(i);
              bool fits=false;
              std::size_t low=1, high=turns.size();
              while (low<high) {
                const auto mid=low+(high-low)/2;
                auto shortened=messages;
                shortened.erase(shortened.begin()+turns.front(),shortened.begin()+turns[mid]);
                std::string prefix;
                std::string prompt;
                try {
                  prompt=server::chat_prompt(shortened, job->automatic_prefix ? &prefix : nullptr,
                                             job->template_tools, job->template_kwargs);
                } catch (const std::exception &e) { error=e.what(); break; }
                std::vector<std::int32_t> ids;
                if (!tokenizer_.encode(prompt,ids,error)) break;
                if (ids.size()>budget) {low=mid+1; continue;}
                fits=true; high=mid;
                job->spec.prompt_tokens=std::move(ids);
                job->prefix_prompt=std::move(prefix);
                job->prompt=std::move(prompt);
              }
              if (!fits) {finish(job,failure(400,error.empty() ? "Latest turn and instructions exceed the context window" : error)); continue;}
            }
          }
          if (!job->draft_source.empty() &&
              !tokenizer_.encode(job->draft_source, job->spec.draft_source_tokens, error)) {
            finish(job, failure(400, error)); continue;
          }
          job->prompt_count = job->spec.prompt_tokens.size();
          if (job->count_only) {
            finish(job, {200, {{"input_tokens", job->prompt_count}}}); continue;
          }
          job->stop_matcher = std::make_unique<server::StopMatcher>(job->stops, 8192);
          job->limit = job->spec.max_new_tokens;
          if (job->automatic_prefix) {
            auto count = job->spec.prompt_tokens.size();
            if (!job->prefix_prompt.empty()) {
              std::vector<std::int32_t> prefix;
              if (!tokenizer_.encode(job->prefix_prompt, prefix, error)) { finish(job, failure(400, error)); continue; }
              count = 0;
              while (count < prefix.size() && count < job->spec.prompt_tokens.size() &&
                     prefix[count] == job->spec.prompt_tokens[count]) ++count;
            }
            configure_cache(job->spec, count);
          }
          for (const auto *token : {"<|im_end|>", "<|endoftext|>"})
            if (auto id = tokenizer_.token_to_id(token)) job->spec.stop_token_ids.push_back(*id);
          if (job->grammar) {
            if (!vocabulary_) vocabulary_ = std::make_shared<server::TokenVocabulary>(tokenizer_);
            job->constraint = std::make_shared<server::JsonConstraint>(job->grammar, vocabulary_,
              job->spec.sampling.temperature <= 1e-6F && job->spec.sampling.repetition_penalty == 1);
            job->spec.sampling_constraint = job->constraint;
          }
          auto id = engine_->submit(std::move(job->spec), error);
          if (!id) { finish(job, failure(400, error)); continue; }
          job->id = id;
          if (job->response_record) job->response_record->start();
          active.push_back({id, job, {}});
          job->ready({200, Json::object()});
        } catch (const std::exception &e) { finish(job, failure(500, e.what())); }
      }
      for (auto &item : active)
        if (stopping || item.job->cancelled || (item.job->response_record && item.job->response_record->cancel_requested))
          engine_->cancel(item.id);
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
        auto &job = it->job;
        if (std::chrono::steady_clock::now() - job->started >= std::chrono::minutes(5)) {
          engine_->cancel(it->id); engine_->release(it->id);
          finish(job, failure(504, "Request deadline exceeded")); it = active.erase(it); continue;
        }
        if (job->cancelled || stopping) {
          engine_->cancel(it->id); engine_->release(it->id);
          finish(job, failure(499, "Request cancelled")); it = active.erase(it); continue;
        }
        // A single scheduler step can confirm the pending token and a full
        // draft block. Drain that many tokens so a fast reader can sustain it.
        auto read_limit = max_mtp_draft_tokens + 1;
        if (job->stream && !job->tool_mode)
          read_limit = std::min(read_limit, job->events.room_for_output());
        if (!read_limit) { ++it; continue; }
        const auto previous = it->output.size();
        CpuRequestResult result;
        if (!engine_->read_output(it->id, read_limit, it->output, error) ||
            !engine_->result(it->id, result)) {
          engine_->cancel(it->id); engine_->release(it->id);
          finish(it->job, failure(500, error)); it = active.erase(it); continue;
        }
        if (!job->decision && !job->classify) {
          try {
            for (auto i = previous; i < it->output.size(); ++i) {
              std::string bytes;
              if (!tokenizer_.decode({it->output[i]}, bytes, error)) throw std::runtime_error(error);
              job->accept(job->stop_matcher->Push(job->utf8.push(bytes)));
              if (job->stop_matcher->stopped()) {
                it->output.resize(i + 1);
                break;
              }
            }
            if (job->stop_matcher->stopped()) {
              engine_->cancel(it->id);
              // Count through the token containing the byte stop; a verified
              // block can already have buffered additional output.
              result.generated_tokens = it->output.size();
              result.status = CpuRequestStatus::complete;
            } else if (result.status == CpuRequestStatus::complete && it->output.size() < result.output_tokens.size()) {
              ++it; continue; // Drain remaining buffered tokens before finalizing.
            }
          } catch (const std::exception &e) {
            engine_->cancel(it->id); engine_->release(it->id);
            finish(job, failure(500, e.what())); it = active.erase(it); continue;
          }
        }
        if (result.status == CpuRequestStatus::complete || result.status == CpuRequestStatus::failed ||
            result.status == CpuRequestStatus::cancelled) {
          Reply reply = failure(500, result.error);
          if (result.status == CpuRequestStatus::cancelled) reply = failure(499, "Request cancelled");
          if (result.status == CpuRequestStatus::complete) {
            if (job->classify) {
              reply = {200, classifier_.score(job->option_logits)};
              reply.body["usage"] = {{"input_tokens", job->prompt_count}, {"output_tokens", 0}};
              reply.body["cache"] = {{"read_tokens", result.cached_prefix_tokens}};
            } else if (it->job->decision) {
              auto &job = it->job; auto &decision = *job->decision;
              job->cached_tokens += result.cached_prefix_tokens;
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
                  {"usage", {{"input_tokens", decision.input_tokens}, {"output_tokens", 0}}},
                  {"cache", {{"read_tokens", job->cached_tokens}}}, {"latency_ms", elapsed}}};
              } catch (const std::exception &e) { reply = failure(500, e.what()); }
            } else {
            try {
              job->accept(job->stop_matcher->Push(job->utf8.push({}, true)));
              job->accept(job->stop_matcher->Finish());
              reply = {200, server::completion(job->api, model_name, it->id, job->created,
                job->text, job->stop_matcher->stopped() || result.output_tokens.size() < job->limit ? "stop" : "length",
                job->prompt_count, result.generated_tokens)};
              if (job->api == server::Api::messages) {
                const auto &stop = job->stop_matcher->matched();
                const auto reason = !stop.empty() ? "stop_sequence" :
                  result.output_tokens.size() < job->limit ? "end_turn" : "max_tokens";
                reply.body = server::message_response(model_name, it->id, job->text, reason,
                  stop.empty() ? Json(nullptr) : Json(stop), server::messages_usage(job->prompt_count,
                    result.generated_tokens, result.cached_prefix_tokens, result.prefix_written_tokens));
              } else reply.body["usage"]["prompt_tokens_details"] = {{"cached_tokens", result.cached_prefix_tokens}};
              if (!job->tool_mode && job->constraint && job->constraint->complete())
                server::require(Json::accept(job->text), "Constrained output is not valid JSON");
              if (job->tool_mode) server::apply_tool_response(reply.body, job->api, job->text, it->id, job->constraint->complete(), job->template_tools);
            } catch (const std::exception &e) { reply = failure(500, e.what()); }
            }
          }
          engine_->release(it->id);
          finish(it->job, std::move(reply)); it = active.erase(it);
        } else ++it;
      }
      if (stopping && active.empty()) return;
      if (!progressed && !active.empty()) {
        std::unique_lock lock(mutex_); wake_.wait_for(lock, std::chrono::milliseconds(1));
      }
    }
  }
public:
  Service(const CpuLoadOptions &load, CpuEngineConfig &config, bool auto_decision_head, bool automatic_prefix, std::size_t fixed_width, bool tune_chunk)
      : max_context_(config.max_context), automatic_prefix_(automatic_prefix),
        prefix_budget_(config.max_prefix_bytes), mtp_enabled_(!load.mtp_weights_path.empty()) {
    std::string error;
    if (!QwenTokenizer::load_from_hf_directory(load.model_dir, tokenizer_, error)) throw std::runtime_error(error);
    classifier_ = server::Classifier::load(load.model_dir, tokenizer_);
    decision_ = systemone::Config::load(load.model_dir, tokenizer_);
    auto profile = ProfileLoader::load_from_hf_directory(load.model_dir, error);
    if (!profile) throw std::runtime_error(error);
    auto model = CpuModel::load(*profile, load, error);
    if (!model) throw std::runtime_error(error);
    const auto architecture = text_architecture(*profile);
    if (std::string(model->weight_format()).starts_with("gguf-"))
      model_name = "Qwen3.5-0.8B-GGUF";
    else if (architecture == TextArchitecture::qwen35_2b)
      model_name = "Qwen3.5-2B-H128-Q4-G32-DOT4";
    else if (architecture == TextArchitecture::qwen35_4b)
      model_name = "Qwen3.5-4B-H128-Q4-G32-DOT4";
    if (decision_) model_name = "qev-0.8b-H128-Q4-G32-DOT4";
    selected_decision_head_ = auto_decision_head && model->supports_selected_logits();
    if (decision_) std::cout << "System One readout: " << (selected_decision_head_ ? "selected" : "full") << '\n';
    tuning_ = server::tune_batches(model, config, fixed_width, decision_ ? std::span<const std::int32_t>(decision_->ids) : std::span<const std::int32_t>{}, selected_decision_head_, tune_chunk);
    engine_ = CpuEngine::create(model, config, error);
    if (!engine_) throw std::runtime_error(error);
    cache_snapshot_ = cache_stats();
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
  const server::BatchTuning &batch_tuning() const { return tuning_; }
  bool classifier_enabled() const { return !classifier_.tokens.empty(); }
  bool decisions_enabled() const { return decision_.has_value(); }
  Json cache_status() { std::lock_guard lock(mutex_); return cache_snapshot_; }
};

std::shared_ptr<Job> parse(const std::string &body, std::size_t context, server::Api api) {
  auto job = std::make_shared<Job>();
  static_cast<server::Request &>(*job) = server::parse_openai(body, context, model_name, api);
  return job;
}
#include "server_batches.inl"
}

int main(int argc, char **argv) try {
  CpuLoadOptions load;
  load.model_dir = "models/qwen3.5-0.8b";
  CpuEngineConfig config;
  config.threads = 0;
  std::size_t batch_width = 0;
  std::string batch_db = ":memory:";
  config.max_resident_requests = 16; config.max_queued_requests = 64;
  config.max_decode_batch_size = 16; config.shared_kv_pages = true;
  std::string host = "127.0.0.1";
  std::string decision_encoding = "joint";
  bool auto_decision_head = true, automatic_prefix = true, tune_chunk = false;
  int port = 8080;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      std::cout << "Qwen3.5 CPU HTTP server\n--model-dir DIR [--weights FILE] [--threads auto|N] [--batch-width auto|N] [--batch-db PATH]\n"
        "[--mtp-weights FILE] [--mtp-depth 1..4] (4B MTP only)\n"
        "[--host 127.0.0.1] [--port 8080] [--max-context 8192] [--residents 16]\n"
        "[--decision-encoding joint|per_question] (qev_config.json required)\n"
        "[--prefill-requests 1..8] [--decision-head auto|full] [--prefill-chunk auto|N] (auto probes short qev decisions; default adaptive)\n"
        "[--prefix-cache auto|off] [--prefix-cache-mib 256]\n"
        "GET /health, GET /v1/models, GET /v1/cache, POST /v1/responses (stored response subresources available)\n"
        "POST /v1/completions, POST /v1/chat/completions, POST /v1/classify, POST /v1/systemone\nPOST /v1/messages, POST /v1/messages/count_tokens\n";
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
    else if (key == "--mtp-weights") load.mtp_weights_path = value;
    else if (key == "--mtp-depth") config.mtp_max_tokens = number();
    else if (key == "--threads") config.threads = value == "auto" ? 0 : number();
    else if (key == "--batch-width") batch_width = value == "auto" ? 0 : number();
    else if (key == "--batch-db") batch_db = value;
    else if (key == "--prefix-cache") {
      if (value != "auto" && value != "off") throw std::runtime_error("Prefix cache must be auto or off");
      automatic_prefix = value == "auto";
    }
    else if (key == "--prefix-cache-mib") {
      const auto mib = number();
      if (mib > 4096) throw std::runtime_error("Prefix cache budget must be <=4096 MiB");
      config.max_prefix_bytes = std::size_t(mib) * 1024 * 1024;
    }
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
    else if (key == "--prefill-requests") config.max_prefill_requests = number();
    else if (key == "--prefill-chunk") {
      tune_chunk = value == "auto";
      config.prefill_chunk_size = tune_chunk ? 0 : number();
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
  if (batch_width > config.max_resident_requests) throw std::runtime_error("--batch-width exceeds --residents");
  server::load_chat_template(load.model_dir);
  Service service(load, config, auto_decision_head, automatic_prefix, batch_width, tune_chunk);
  BatchService batches(service, batch_db, config.max_context, service.batch_tuning().width, decision_encoding);
  server::ResponseStore responses;
  httplib::Server http;
  http.new_task_queue = [] { return new httplib::ThreadPool(32, 32, 64); };
  http.set_payload_max_length(4 * 1024 * 1024 + 16384);
  http.set_read_timeout(10, 0); http.set_write_timeout(10, 0); http.set_keep_alive_max_count(1);
  auto send = [](httplib::Response &response, const Reply &reply, bool messages = false) {
    response.status = reply.status;
    response.set_content((messages && reply.status != 200 ? server::messages_error(reply.status, reply.body) : reply.body).dump(), "application/json");
  };
  auto authorized = [&](const httplib::Request &request, httplib::Response &response, bool messages = false) {
    if (api_key.empty() || request.get_header_value("Authorization") == "Bearer " + api_key ||
        (messages && request.get_header_value("x-api-key") == api_key)) return true;
    send(response, failure(401, "Unauthorized"), messages); return false;
  };
  http.set_pre_routing_handler([&](const httplib::Request &req, httplib::Response &res) {
    if (req.method == "POST" && req.path != "/v1/files" &&
        req.get_header_value_u64("Content-Length") > 262144) {
      send(res,failure(413,"Request body exceeds 256 KiB")); return httplib::Server::HandlerResponse::Handled;
    }
    return httplib::Server::HandlerResponse::Unhandled;
  });
  http.set_pre_request_handler([&](const httplib::Request &req, httplib::Response &res) {
    if (req.path != "/v1/files" && (req.body.size()>262144 || req.is_multipart_form_data())) {
      send(res,failure(413,"Invalid or oversized request body")); return httplib::Server::HandlerResponse::Handled;
    }
    return httplib::Server::HandlerResponse::Unhandled;
  });
#include "server_batch_routes.inl"
  http.Get("/health", [&](const auto &request, auto &response) {
    if (authorized(request, response)) send(response, {200, {{"status", "ok"}}});
  });
  http.Get("/v1/models", [&](const auto &request, auto &response) {
    if (authorized(request, response)) send(response, {200, {{"object", "list"},
      {"data", Json::array({{{"id", model_name}, {"object", "model"}, {"owned_by", "local"}}})}}});
  });
  http.Get("/v1/cache", [&](const auto &request, auto &response) {
    if (authorized(request, response)) send(response, {200, service.cache_status()});
  });
  auto dispatch = [&](const std::shared_ptr<Job> &job, const httplib::Request &request, httplib::Response &response) {
    const bool messages = job->api == server::Api::messages;
    auto result = job->stream ? job->admitted.get_future() : job->promise.get_future();
    if (!service.enqueue(job)) { send(response, failure(429, "Request capacity exhausted"), messages); return; }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
    while (result.wait_for(std::chrono::milliseconds(50)) != std::future_status::ready) {
      if (request.is_connection_closed()) { job->cancelled = true; return; }
      if (std::chrono::steady_clock::now() >= deadline) {
        job->cancelled = true; send(response, failure(504, "Request deadline exceeded"), messages); return;
      }
    }
    const auto reply = result.get();
    if (!job->stream || reply.status != 200) { send(response, reply, messages); return; }
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
    response.set_chunked_content_provider("text/event-stream",
      [job, closed = request.is_connection_closed](std::size_t, httplib::DataSink &sink) {
        // Deferred providers execute after the route handler: contain every exception.
        try {
          const bool messages = job->api == server::Api::messages;
          auto wire_write = [&](const std::string &wire) { return sink.write(wire.data(), wire.size()); };
          auto write = [&](const Json &data) {
            const auto wire = "data: " + data.dump() + "\n\n";
            return sink.write(wire.data(), wire.size());
          };
          auto chunk = [&](const std::string &text, bool begin, const Json &reason = nullptr) {
            auto data = server::chunk(job->api, model_name, job->id, job->created, text, begin, reason);
            if (job->include_usage) data["usage"] = nullptr;
            return data;
          };
          if (!(messages ? wire_write(server::message_start(model_name, job->id, !job->tool_mode)) : write(chunk("", true)))) { job->cancelled = true; return false; }
          auto last_write = std::chrono::steady_clock::now();
          for (;;) {
            if (closed()) { job->cancelled = true; return false; }
            std::string text; std::optional<Reply> terminal;
            if (job->events.pop(text, terminal)) {
              if (!(messages ? wire_write(server::message_delta(text)) : write(chunk(text, false)))) { job->cancelled = true; return false; }
              last_write = std::chrono::steady_clock::now();
            } else if (terminal) {
              if (terminal->status != 200) {
                const bool sent = messages ? wire_write(server::message_event(server::messages_error(terminal->status, terminal->body))) : write(terminal->body);
                sink.done(); return sent;
              }
              if (messages) {
                if (job->tool_mode && !wire_write(server::message_tool_events(terminal->body))) return false;
                if (!wire_write(server::message_end(terminal->body, !job->tool_mode))) return false;
                sink.done(); return true;
              }
              if (job->tool_mode) {
                const auto &message = terminal->body["choices"][0]["message"];
                auto data = chunk("", false);
                if (message["content"].is_string() && !write(chunk(message["content"].get<std::string>(), false))) return false;
                if (message.contains("tool_calls")) {
                  auto calls = message["tool_calls"]; calls[0]["index"] = 0;
                  data["choices"][0]["delta"] = {{"tool_calls", calls}};
                  if (!write(data)) return false;
                }
              }
              if (!write(chunk("", false, terminal->body["choices"][0]["finish_reason"]))) return false;
              if (job->include_usage) {
                auto usage = chunk("", false); usage["choices"] = Json::array();
                usage["usage"] = terminal->body["usage"];
                if (!write(usage)) return false;
              }
              if (!sink.write("data: [DONE]\n\n", 14)) return false;
              sink.done(); return true;
            }
            if (std::chrono::steady_clock::now() - last_write >= std::chrono::seconds(10)) {
              if (!sink.write(": heartbeat\n\n", 13)) { job->cancelled = true; return false; }
              last_write = std::chrono::steady_clock::now();
            }
          }
        } catch (...) { job->cancelled = true; return false; }
      }, [job](bool success) { if (!success) job->cancelled = true; });
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
#include "server_responses_routes.inl"
  auto messages_route = [&](const httplib::Request &request, httplib::Response &response, bool count_only) {
    if (!authorized(request, response, true)) return;
    const auto version = request.get_header_value("anthropic-version");
    if ((version != "2023-06-01" && version != "2026-10-21") || request.has_header("anthropic-beta")) {
      send(response, failure(400, "Require anthropic-version: 2023-06-01 or 2026-10-21; beta features are unsupported"), true); return;
    }
    auto job = std::make_shared<Job>();
    try {
      if (version == "2026-10-21") {
        const auto body = Json::parse(request.body);
        for (const auto *key : {"temperature", "top_p", "top_k"})
          server::require(!body.contains(key), std::string(key) + " was removed in API version 2026-10-21");
      }
      static_cast<server::Request &>(*job) = server::parse_messages(request.body, config.max_context, model_name, count_only);
      job->count_only = count_only;
    } catch (const std::exception &e) { send(response, failure(400, e.what()), true); return; }
    dispatch(job, request, response);
  };
  http.Post("/v1/messages", [&](const httplib::Request &request, httplib::Response &response) {
    messages_route(request, response, false);
  });
  http.Post("/v1/messages/count_tokens", [&](const httplib::Request &request, httplib::Response &response) {
    messages_route(request, response, true);
  });
  http.Post("/v1/classify", [&](const httplib::Request &request, httplib::Response &response) {
    if (!authorized(request, response)) return;
    if (!service.classifier_enabled()) { send(response, failure(400, "This model has no classifier.json")); return; }
    auto job = std::make_shared<Job>();
    try {
      const auto body = Json::parse(request.body);
      job->prompt = body.at("prompt").get<std::string>();
      server::require(body.size() == 1 && !job->prompt.empty(), "Expected a non-empty prompt");
      job->classify = true;
    } catch (const std::exception &e) { send(response, failure(400, e.what())); return; }
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
