#pragma once
#include "qwen35x/runtime/cpu_engine.h"
#include "server_text.h"
#include "server_template.h"
#include "server_unicode_trim.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <string_view>

// HTTP shape and model prompt formatting are independent of inference ownership.
namespace qwen35x::server {
using Json = nlohmann::ordered_json;
enum class Api { completion, chat, messages };
class ByteGrammar;
struct Request {
  Api api = Api::completion;
  std::string prompt, prefix_prompt, draft_source;
  CpuRequestSpec spec;
  bool stream = false, include_usage = false;
  bool automatic_prefix = true;
  std::vector<std::string> stops;
  std::shared_ptr<const ByteGrammar> grammar;
  bool tool_mode = false;
  Json template_tools = Json::array(), template_kwargs = Json::object();
};
inline void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}
inline void fields(const Json &j, std::initializer_list<std::string_view> allowed) {
  require(j.is_object(), "Expected JSON object");
  for (const auto &[key, value] : j.items()) {
    (void)value;
    require(std::find(allowed.begin(), allowed.end(), key) != allowed.end(), "Unsupported field: " + key);
  }
}
inline std::int64_t integer(const Json &j, const char *key, std::int64_t fallback) {
  if (!j.contains(key) || j[key].is_null()) return fallback;
  const auto &value = j[key];
  require(value.is_number_integer(), std::string(key) + " must be an integer");
  require(!value.is_number_unsigned() || value.get<std::uint64_t>() <=
          std::uint64_t(std::numeric_limits<std::int64_t>::max()), std::string(key) + " is out of range");
  return value.get<std::int64_t>();
}
inline float real(const Json &j, const char *key, float fallback) {
  if (!j.contains(key) || j[key].is_null()) return fallback;
  require(j[key].is_number(), std::string(key) + " must be a number");
  const double value = j[key].get<double>();
  require(std::isfinite(value) && std::abs(value) <= std::numeric_limits<float>::max(),
          std::string(key) + " must be finite and representable");
  return static_cast<float>(value);
}
inline std::string content(const Json &value, bool nullable = false) {
  if (nullable && value.is_null()) return {};
  if (value.is_string()) return value.get<std::string>();
  require(value.is_array(), "Content must be text or an array of text parts");
  std::string out;
  for (const auto &part : value) {
    fields(part, {"type", "text"});
    require(part.at("type") == "text" && part.at("text").is_string(), "Only text content parts are supported");
    out += part.at("text").get<std::string>();
  }
  return out;
}
inline std::string chat_prompt(const Json &messages, std::string *prefix_prompt = nullptr,
                               const Json &tools = Json::array(), const Json &kwargs = Json::object()) {
  require(messages.is_array() && !messages.empty() && messages.size() <= 1024,
          "messages must contain 1..1024 messages");
  Json rows = Json::array();
  std::size_t last_user = 0;
  bool has_user = false;
  for (const auto &m : messages) {
    fields(m, {"role", "content", "reasoning_content", "tool_calls", "tool_call_id"});
    const auto role = m.at("role").get<std::string>();
    require(role == "system" || role == "developer" || role == "user" || role == "assistant" ||
            role == "tool", "Unsupported message role: " + role);
    const auto text = content(m.at("content"), role == "assistant");
    if (m.contains("reasoning_content") && !m["reasoning_content"].is_null())
      require(role == "assistant" && m["reasoning_content"].is_string(), "Invalid assistant reasoning");
    if (role == "system" || role == "developer") {
      require(rows.empty() || (rows.size() == 1 && rows.front()["role"] == "system"),
              "System/developer messages must precede the conversation");
      if (!rows.empty()) {
        rows[0]["content"] = trim(content(rows[0]["content"])) + "\n\n" + trim(text);
        continue;
      }
    }
    if (role == "user") { last_user = rows.size(); has_user = true; }
    rows.push_back(m);
    if (role == "developer") rows.back()["role"] = "system";
  }
  require(has_user, "At least one user query is required");
  auto prompt = render_chat_template(rows, tools, kwargs, true);
  if (prefix_prompt) {
    // A custom template may transform, omit, or reorder a partial history.
    // Only cache a byte prefix that the actual generation prompt also has.
    std::string candidate;
    try {
      if (rows.back()["role"] == "tool" || !last_user) {
        candidate = render_chat_template(rows, tools, kwargs, false);
      } else {
        constexpr auto marker = "Qwen35PrefixBoundary89c8";
        rows.erase(rows.begin() + last_user + 1, rows.end());
        rows.back()["content"] = marker;
        candidate = render_chat_template(rows, tools, kwargs, false);
        const auto pos = candidate.find(marker);
        if (pos == std::string::npos) candidate.clear();
        else candidate.resize(pos);
      }
    } catch (const std::exception &) {
      // Some fine-tune templates accept only complete generation requests.
      candidate.clear();
    }
    auto match = std::mismatch(prompt.begin(), prompt.end(), candidate.begin(), candidate.end());
    prefix_prompt->assign(prompt.begin(), match.first);
  }
  return prompt;
}
inline Request parse(const std::string &body, std::size_t context, const std::string &model, Api api,
                     const Json &template_tools = Json::array()) {
  const auto j = Json::parse(body);
  if (api == Api::chat)
    fields(j, {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "stream", "n",
               "prefix_tokens", "top_p", "top_k", "seed", "repetition_penalty", "stop", "stream_options",
               "cache_prompt", "prompt_cache_key", "draft", "draft_source", "chat_template_kwargs"});
  else
    fields(j, {"model", "prompt", "max_tokens", "temperature", "stream", "n", "prefix_tokens",
               "top_p", "top_k", "seed", "repetition_penalty", "stop", "stream_options", "cache_prompt", "prompt_cache_key", "draft", "draft_source"});
  if (j.contains("model")) require(j.at("model").get<std::string>() == model, "Unknown model");
  Request request; request.api = api;
  if (j.contains("draft")) request.spec.draft_mode = parse_cpu_draft_mode(j.at("draft").get<std::string>());
  if (j.contains("draft_source")) {
    require(request.spec.draft_mode == CpuDraftMode::input, "draft_source requires draft=input");
    request.draft_source = j.at("draft_source").get<std::string>();
  }
  if (j.contains("stream") && !j["stream"].is_null()) {
    require(j["stream"].is_boolean(), "stream must be boolean");
    request.stream = j["stream"].get<bool>();
  }
  if (j.contains("stream_options") && !j["stream_options"].is_null()) {
    require(request.stream, "stream_options requires stream=true");
    fields(j["stream_options"], {"include_usage"});
    if (j["stream_options"].contains("include_usage")) {
      require(j["stream_options"]["include_usage"].is_boolean(), "include_usage must be boolean");
      request.include_usage = j["stream_options"]["include_usage"].get<bool>();
    }
  }
  if (j.contains("stop") && !j["stop"].is_null()) {
    const auto &stop = j["stop"];
    require(stop.is_string() || stop.is_array(), "stop must be text or an array of strings");
    if (stop.is_string()) request.stops.push_back(stop.get<std::string>());
    else for (const auto &value : stop) request.stops.push_back(value.get<std::string>());
    (void)StopMatcher(request.stops, 8192);
  }
  require(integer(j, "n", 1) == 1, "Only n=1 is supported");
  request.template_tools = template_tools;
  if (j.contains("chat_template_kwargs")) {
    const auto &args = j.at("chat_template_kwargs");
    require(args.is_object(), "chat_template_kwargs must be an object");
    for (auto key : {"messages", "tools", "add_generation_prompt", "bos_token", "eos_token"})
      require(!args.contains(key), std::string("Reserved template variable: ") + key);
    require(!args.contains("enable_thinking") || args["enable_thinking"].is_boolean(),
            "enable_thinking must be boolean");
    request.template_kwargs = args;
  }
  if (j.contains("cache_prompt")) {
    require(j["cache_prompt"].is_boolean(), "cache_prompt must be boolean");
    request.automatic_prefix = j["cache_prompt"].get<bool>();
  }
  if (j.contains("prefix_tokens")) request.automatic_prefix = false;
  auto *prefix_prompt = request.automatic_prefix ? &request.prefix_prompt : nullptr;
  request.prompt = api == Api::chat
      ? chat_prompt(j.at("messages"), prefix_prompt, template_tools, request.template_kwargs)
      : j.at("prompt").get<std::string>();
  require(!request.prompt.empty() && request.prompt.size() <= 65536, "Prompt must contain 1..65536 bytes");
  require(!(j.contains("max_tokens") && !j["max_tokens"].is_null() && j.contains("max_completion_tokens") &&
            !j["max_completion_tokens"].is_null()), "Specify only one output token limit");
  const auto limit = integer(j, "max_completion_tokens", integer(j, "max_tokens", 128));
  const auto prefix = integer(j, "prefix_tokens", 0);
  require(limit >= 1 && limit <= static_cast<std::int64_t>(context) && prefix >= 0 &&
          prefix <= static_cast<std::int64_t>(context), "Token limit out of range");
  request.spec.max_new_tokens = static_cast<std::size_t>(limit);
  request.spec.register_prefix_tokens = static_cast<std::size_t>(prefix);
  request.spec.trust_namespace = "local-server";
  if (j.contains("prompt_cache_key") && !j["prompt_cache_key"].is_null()) {
    const auto key = j["prompt_cache_key"].get<std::string>();
    require(!key.empty() && key.size() <= 64, "prompt_cache_key must contain 1..64 UTF-8 bytes");
    request.spec.trust_namespace += "\n" + key;
  }
  auto &s = request.spec.sampling;
  s.temperature = real(j, "temperature", api == Api::chat ? 1.0f : 0.0f);
  s.top_p = real(j, "top_p", api == Api::chat ? 1.0f : 0.8f);
  s.repetition_penalty = real(j, "repetition_penalty", 1);
  const auto top_k = integer(j, "top_k", api == Api::chat ? 0 : 20), seed = integer(j, "seed", -1);
  require(s.temperature >= 0 && s.top_p > 0 && s.top_p <= 1 && s.repetition_penalty >= 1 &&
          top_k >= 0 && top_k <= 248320 && seed >= -1 && seed <= std::numeric_limits<std::uint32_t>::max(),
          "Sampling parameter out of range");
  s.top_k = static_cast<int>(top_k); s.seed = seed;
  return request;
}
inline Json completion(Api api, const std::string &model, CpuRequestId id, std::time_t created,
                       const std::string &text, const std::string &reason, std::size_t prompt, std::size_t output) {
  Json choice = {{"index", 0}, {"logprobs", nullptr}, {"finish_reason", reason}};
  if (api == Api::chat) choice["message"] = {{"role", "assistant"}, {"content", text}};
  else choice["text"] = text;
  return {{"id", (api == Api::chat ? "chatcmpl-" : "cmpl-") + std::to_string(id)},
    {"object", api == Api::chat ? "chat.completion" : "text_completion"}, {"created", created}, {"model", model},
    {"choices", Json::array({choice})}, {"usage", {{"prompt_tokens", prompt}, {"completion_tokens", output},
                                              {"total_tokens", prompt + output}}}};
}
inline Json chunk(Api api, const std::string &model, CpuRequestId id, std::time_t created,
                  const std::string &text, bool begin, const Json &reason = nullptr) {
  Json choice = {{"index", 0}, {"finish_reason", reason}, {"logprobs", nullptr}};
  if (api == Api::chat) {
    choice["delta"] = Json::object();
    if (begin) choice["delta"]["role"] = "assistant";
    if (reason.is_null()) choice["delta"]["content"] = text;
  } else choice["text"] = text;
  return {{"id", (api == Api::chat ? "chatcmpl-" : "cmpl-") + std::to_string(id)},
          {"object", api == Api::chat ? "chat.completion.chunk" : "text_completion"},
          {"created", created}, {"model", model}, {"choices", Json::array({choice})}};
}
} // namespace qwen35x::server
