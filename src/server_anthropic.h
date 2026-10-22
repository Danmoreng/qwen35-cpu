#pragma once
#include "server_protocol.h"

namespace qwen35x::server {
inline Request parse_messages(const std::string &body, std::size_t context,
                              const std::string &model, bool count_only = false) {
  const auto j = Json::parse(body);
  if (count_only) fields(j, {"model", "messages", "system"});
  else fields(j, {"model", "messages", "system", "max_tokens", "temperature", "top_p", "top_k",
                  "stop_sequences", "stream", "thinking", "cache_prompt", "prompt_cache_key"});
  require(j.at("model") == model, "Unknown model");
  if (!count_only) require(j.contains("max_tokens") && !j["max_tokens"].is_null(), "max_tokens is required");
  Json normalized = {{"model", model}, {"messages", Json::array()}};
  if (j.contains("system"))
    normalized["messages"].push_back({{"role", "system"}, {"content", content(j["system"])}});
  require(j.at("messages").is_array(), "messages must be an array");
  for (const auto &message : j["messages"]) {
    fields(message, {"role", "content"});
    require(message.at("role") == "user" || message.at("role") == "assistant", "Messages roles must be user or assistant");
    auto text = content(message.at("content"));
    // The Messages protocol combines consecutive turns with the same role.
    auto &rows = normalized["messages"];
    if (!rows.empty() && rows.back()["role"] == message["role"])
      rows.back()["content"] = rows.back()["content"].get<std::string>() + "\n\n" + text;
    else rows.push_back({{"role", message["role"]}, {"content", text}});
  }
  require(!j["messages"].empty() && j["messages"].size() <= 1024, "messages must contain 1..1024 messages");
  require(j["messages"].back()["role"] == "user", "Assistant prefilling is not supported");
  for (const auto *key : {"max_tokens", "temperature", "top_p", "top_k", "stream", "cache_prompt", "prompt_cache_key"})
    if (j.contains(key)) normalized[key] = j[key];
  if (count_only) normalized["max_tokens"] = 1;
  if (j.contains("stop_sequences")) {
    require(j["stop_sequences"].is_array(), "stop_sequences must be an array");
    normalized["stop"] = j["stop_sequences"];
  }
  if (j.contains("thinking")) {
    fields(j["thinking"], {"type"});
    require(j["thinking"].at("type") == "disabled", "Only disabled thinking is supported");
  }
  auto request = parse(normalized.dump(), context, model, Api::chat);
  require(request.spec.sampling.temperature <= 1, "temperature must be in 0..1");
  request.api = Api::messages;
  return request;
}
inline Json messages_usage(std::size_t prompt, std::size_t output,
                           std::size_t cached = 0, std::size_t written = 0) {
  require(cached + written <= prompt, "Invalid cache accounting");
  return {{"input_tokens", prompt - cached - written}, {"output_tokens", output},
          {"cache_creation_input_tokens", written}, {"cache_read_input_tokens", cached}};
}
inline Json message_response(const std::string &model, CpuRequestId id, const std::string &text,
                             const Json &reason, const Json &stop, const Json &usage) {
  return {{"id", "msg_" + std::to_string(id)}, {"type", "message"}, {"role", "assistant"},
    {"model", model}, {"content", Json::array({{{"type", "text"}, {"text", text}}})},
    {"stop_reason", reason}, {"stop_sequence", stop}, {"usage", usage}};
}
inline Json messages_error(int status, const Json &body) {
  const char *type = status == 401 ? "authentication_error" : status == 429 ? "rate_limit_error" :
                     status == 404 ? "not_found_error" : status < 500 ? "invalid_request_error" : "api_error";
  return {{"type", "error"}, {"error", {{"type", type}, {"message", body.at("error").at("message")}}}};
}
inline std::string message_event(const Json &event) {
  return "event: " + event.at("type").get<std::string>() + "\ndata: " + event.dump() + "\n\n";
}
inline std::string message_start(const std::string &model, CpuRequestId id) {
  auto message = message_response(model, id, "", nullptr, nullptr, messages_usage(0, 0));
  message["content"] = Json::array();
  return message_event({{"type", "message_start"}, {"message", message}}) +
    message_event({{"type", "content_block_start"}, {"index", 0},
                   {"content_block", {{"type", "text"}, {"text", ""}}}});
}
inline std::string message_delta(const std::string &text) {
  return message_event({{"type", "content_block_delta"}, {"index", 0},
                        {"delta", {{"type", "text_delta"}, {"text", text}}}});
}
inline std::string message_end(const Json &message) {
  return message_event({{"type", "content_block_stop"}, {"index", 0}}) +
    message_event({{"type", "message_delta"},
      {"delta", {{"stop_reason", message.at("stop_reason")}, {"stop_sequence", message.at("stop_sequence")}}},
      {"usage", message.at("usage")}}) + message_event({{"type", "message_stop"}});
}
} // namespace qwen35x::server
