#pragma once
#include "server_structured.h"

namespace qwen35x::server {
inline Request parse_messages(const std::string &body, std::size_t context,
                              const std::string &model, bool count_only = false) {
  const auto j = Json::parse(body);
  if (count_only) fields(j, {"model", "messages", "system", "tools", "tool_choice", "output_config"});
  else fields(j, {"model", "messages", "system", "max_tokens", "temperature", "top_p", "top_k",
                  "stop_sequences", "stream", "thinking", "cache_prompt", "prompt_cache_key", "tools", "tool_choice", "output_config"});
  require(j.at("model") == model, "Unknown model");
  if (!count_only) require(j.contains("max_tokens") && !j["max_tokens"].is_null(), "max_tokens is required");
  Json normalized = {{"model", model}, {"messages", Json::array()}};
  if (j.contains("system"))
    normalized["messages"].push_back({{"role", "system"}, {"content", content(j["system"])}});
  require(j.at("messages").is_array(), "messages must be an array");
  for (const auto &message : j["messages"]) {
    fields(message, {"role", "content"});
    const auto role = message.at("role").get<std::string>();
    require(role == "user" || role == "assistant", "Messages roles must be user or assistant");
    const auto blocks = message.at("content").is_string() ?
      Json::array({{{"type", "text"}, {"text", message["content"]}}}) : message["content"];
    require(blocks.is_array(), "Message content must be text or blocks");
    Json row = {{"role", role}, {"content", ""}};
    for (const auto &block : blocks) {
      const auto type = block.at("type").get<std::string>();
      if (type == "text") {
        fields(block, {"type", "text"});
        row["content"] = row["content"].get<std::string>() + block.at("text").get<std::string>();
      } else if (type == "tool_use") {
        fields(block, {"type", "id", "name", "input"});
        require(role == "assistant" && !row.contains("tool_calls") && block.at("input").is_object(),
                "Only one assistant tool_use block is supported");
        row["tool_calls"] = Json::array({{{"id", block.at("id")}, {"type", "function"},
          {"function", {{"name", block.at("name")}, {"arguments", block["input"].dump()}}}}});
      } else if (type == "tool_result") {
        fields(block, {"type", "tool_use_id", "content", "is_error"});
        require(role == "user" && row["content"] == "", "Tool results must precede user text");
        auto result = content(block.at("content"));
        if (block.contains("is_error")) {
          require(block["is_error"].is_boolean(), "is_error must be boolean");
          if (block["is_error"] == true) result = Json({{"is_error", true}, {"content", result}}).dump();
        }
        normalized["messages"].push_back({{"role", "tool"}, {"tool_call_id", block.at("tool_use_id")}, {"content", result}});
      } else require(false, "Unsupported Messages content block: " + type);
    }
    auto &rows = normalized["messages"];
    if (row["content"] == "" && !row.contains("tool_calls") && !rows.empty() && rows.back()["role"] == "tool") continue;
    if (!rows.empty() && rows.back()["role"] == role && !row.contains("tool_calls") && !rows.back().contains("tool_calls"))
      rows.back()["content"] = rows.back()["content"].get<std::string>() + "\n\n" + row["content"].get<std::string>();
    else rows.push_back(std::move(row));
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
  if (j.contains("tools")) {
    require(j["tools"].is_array(), "tools must be an array");
    normalized["tools"] = Json::array();
    for (const auto &tool : j["tools"]) {
      fields(tool, {"name", "description", "input_schema", "strict"});
      Json fn = {{"name", tool.at("name")}, {"parameters", tool.at("input_schema")}};
      for (const auto *key : {"description", "strict"}) if (tool.contains(key)) fn[key] = tool[key];
      normalized["tools"].push_back({{"type", "function"}, {"function", fn}});
    }
  }
  if (j.contains("tool_choice")) {
    const auto &choice = j["tool_choice"]; fields(choice, {"type", "name", "disable_parallel_tool_use"});
    if (choice.contains("disable_parallel_tool_use"))
      require(choice["disable_parallel_tool_use"].is_boolean(), "disable_parallel_tool_use must be boolean");
    const auto type = choice.at("type").get<std::string>();
    if (type == "tool") normalized["tool_choice"] = {{"type", "function"}, {"function", {{"name", choice.at("name")}}}};
    else {
      require(!choice.contains("name") && (type == "auto" || type == "any" || type == "none"), "Invalid tool choice");
      normalized["tool_choice"] = type == "any" ? "required" : type;
    }
  }
  if (j.contains("output_config")) {
    fields(j["output_config"], {"format"});
    const auto &format = j["output_config"].at("format"); fields(format, {"type", "schema"});
    require(format.at("type") == "json_schema", "Unsupported output format");
    normalized["response_format"] = {{"type", "json_schema"},
      {"json_schema", {{"name", "response"}, {"strict", true}, {"schema", format.at("schema")}}}};
  }
  auto request = parse_openai(normalized.dump(), context, model, Api::chat);
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
inline std::string message_start(const std::string &model, CpuRequestId id, bool text_block = true) {
  auto message = message_response(model, id, "", nullptr, nullptr, messages_usage(0, 0));
  message["content"] = Json::array();
  return message_event({{"type", "message_start"}, {"message", message}}) +
    (text_block ? message_event({{"type", "content_block_start"}, {"index", 0},
                   {"content_block", {{"type", "text"}, {"text", ""}}}}) : "");
}
inline std::string message_delta(const std::string &text) {
  return message_event({{"type", "content_block_delta"}, {"index", 0},
                        {"delta", {{"type", "text_delta"}, {"text", text}}}});
}
inline std::string message_tool_events(const Json &message) {
  std::string wire;
  for (std::size_t i = 0; i < message.at("content").size(); ++i) {
    const auto &block = message["content"][i];
    auto start = block;
    Json delta;
    if (block.at("type") == "tool_use") {
      start["input"] = Json::object();
      delta = {{"type", "input_json_delta"}, {"partial_json", block.at("input").dump()}};
    } else {
      start["text"] = ""; delta = {{"type", "text_delta"}, {"text", block.at("text")}};
    }
    wire += message_event({{"type", "content_block_start"}, {"index", i}, {"content_block", start}});
    wire += message_event({{"type", "content_block_delta"}, {"index", i}, {"delta", delta}});
    wire += message_event({{"type", "content_block_stop"}, {"index", i}});
  }
  return wire;
}
inline std::string message_end(const Json &message, bool text_block = true) {
  return (text_block ? message_event({{"type", "content_block_stop"}, {"index", 0}}) : "") +
    message_event({{"type", "message_delta"},
      {"delta", {{"stop_reason", message.at("stop_reason")}, {"stop_sequence", message.at("stop_sequence")}}},
      {"usage", message.at("usage")}}) + message_event({{"type", "message_stop"}});
}
} // namespace qwen35x::server
