#pragma once
#include "server_stream.h"
#include "server_structured.h"
#include <atomic>
#include <random>

namespace qwen35x::server {

inline std::string response_id(const char *prefix) {
  static const auto instance =
      std::to_string(std::random_device{}()) + "_" + std::to_string(std::random_device{}());
  static std::atomic<std::uint64_t> serial{};
  return prefix + instance + "_" + std::to_string(++serial);
}

inline Json response_text(const std::string &text) {
  return {{"type", "output_text"},
          {"text", text},
          {"annotations", Json::array()},
          {"logprobs", Json::array()}};
}

inline Json response_message(const std::string &id, const std::string &status, Json parts) {
  return {{"type", "message"},
          {"id", id},
          {"role", "assistant"},
          {"status", status},
          {"content", std::move(parts)}};
}

// Convert Responses history to the existing chat formatter. Keep canonical items
// separately so input_items and previous_response_id preserve the wire format.
inline Json response_chat_item(Json &item) {
  const auto type = item.value("type", std::string("message"));
  Json message;
  if (type == "message") {
    fields(item, {"type", "id", "role", "status", "content"});
    const auto role = item.at("role").get<std::string>();
    auto &parts = item.at("content");
    if (parts.is_string())
      parts = Json::array({{{"type", "input_text"}, {"text", parts.get<std::string>()}}});
    require(parts.is_array(), "Expected text content parts");
    std::string text;
    for (const auto &part : parts) {
      const auto kind = part.at("type").get<std::string>();
      if (kind == "output_text" && role == "assistant")
        fields(part, {"type", "text", "annotations", "logprobs"});
      else {
        fields(part, {"type", "text"});
        require(kind == "input_text", "Only text input is supported");
      }
      text += part.at("text").get<std::string>();
    }
    message = {{"role", role}, {"content", text}};
  } else if (type == "function_call") {
    fields(item, {"type", "id", "call_id", "name", "arguments", "status"});
    Json fn = {{"name", item.at("name")}, {"arguments", item.at("arguments")}};
    Json call = {{"type", "function"}, {"id", item.at("call_id")}, {"function", fn}};
    message = {{"role", "assistant"}, {"tool_calls", Json::array({call})}};
  } else if (type == "function_call_output") {
    fields(item, {"type", "id", "call_id", "output", "status"});
    const auto &output = item.at("output");
    std::string text;
    if (output.is_string())
      text = output.get<std::string>();
    else {
      require(output.is_array(), "Tool output must be text or input_text parts");
      for (const auto &part : output) {
        fields(part, {"type", "text"});
        require(part.at("type") == "input_text", "Only text tool output is supported");
        text += part.at("text").get<std::string>();
      }
    }
    message = {{"role", "tool"}, {"tool_call_id", item.at("call_id")}, {"content", text}};
  } else
    throw std::invalid_argument("Unsupported Responses input item: " + type);
  item["type"] = type;
  if (!item.contains("id"))
    item["id"] = response_id("item_");
  require(item["id"].is_string() && !item["id"].get_ref<const std::string &>().empty(),
          "Invalid item ID");
  if (!item.contains("status"))
    item["status"] = "completed";
  require(item["status"] == "completed" || item["status"] == "incomplete",
          "Only finalized history items are supported");
  return message;
}

struct PreparedResponse {
  Request request;
  Json body, history, formatted_messages;
  bool store, background, stream;
};

inline PreparedResponse prepare_response(Json j, Json history, std::size_t context,
                                         const std::string &model) {
  fields(j, {"model",
             "input",
             "instructions",
             "previous_response_id",
             "metadata",
             "max_output_tokens",
             "temperature",
             "top_p",
             "text",
             "tools",
             "tool_choice",
             "parallel_tool_calls",
             "max_tool_calls",
             "store",
             "stream",
             "background",
             "prompt_cache_key",
             "prompt_cache_retention",
             "truncation",
             "reasoning",
             "include",
             "top_logprobs",
             "service_tier",
             "safety_identifier",
             "user"});
  for (auto it = j.begin(); it != j.end();) {
    if (it.value().is_null())
      it = j.erase(it);
    else
      ++it;
  }
  require(j.at("model") == model, "Unknown model");
  const bool store = j.value("store", true), background = j.value("background", false),
             stream = j.value("stream", false);
  require(!background || store, "Background responses require store=true");
  const auto truncation = j.value("truncation", std::string("disabled"));
  require(truncation == "disabled" || truncation == "auto", "Invalid truncation strategy");
  require(real(j, "temperature", 1) <= 2, "temperature must be at most 2");
  for (auto key : {"safety_identifier", "user"})
    if (j.contains(key))
      require(j[key].is_string(), std::string(key) + " must be text");
  require(integer(j, "top_logprobs", 0) == 0, "Top log probabilities require a sampler readout");
  require(!j.contains("include") || (j["include"].is_array() && j["include"].empty()),
          "Unsupported include selection");
  require(!j.value("parallel_tool_calls", false), "Parallel function calls are unsupported");
  require(j.value("prompt_cache_retention", std::string("in_memory")) == "in_memory",
          "Only in_memory cache retention is supported");
  const auto tier = j.value("service_tier", std::string("default"));
  require(tier == "auto" || tier == "default", "Unsupported service tier");
  if (j.contains("reasoning")) {
    fields(j["reasoning"], {"effort"});
    require(j["reasoning"].value("effort", std::string("none")) == "none",
            "Reasoning effort is unsupported");
  }
  const Json metadata = j.value("metadata", Json::object());
  require(metadata.is_object() && metadata.size() <= 16, "Metadata must contain at most 16 entries");
  for (const auto &[key, value] : metadata.items())
    require(key.size() <= 64 && value.is_string() && value.get_ref<const std::string &>().size() <= 512,
            "Invalid metadata");
  Json messages = Json::array(), chat = {{"model", model}};
  if (j.contains("instructions"))
    messages.push_back({{"role", "system"}, {"content", j["instructions"].get<std::string>()}});
  const auto &input = j.at("input");
  if (input.is_string())
    history.push_back({{"role", "user"}, {"content", input}});
  else {
    require(input.is_array(), "input must be a string or item array");
    for (const auto &item : input)
      history.push_back(item);
  }
  require(history.size() <= 1024 && history.dump().size() <= 262144,
          "Response history exceeds input limits");
  std::set<std::string> ids;
  for (auto &item : history) {
    messages.push_back(response_chat_item(item));
    require(ids.insert(item["id"].get<std::string>()).second, "Duplicate input item ID");
  }
  chat["messages"] = std::move(messages);
  for (const auto key : {"temperature", "top_p", "prompt_cache_key"})
    if (j.contains(key))
      chat[key] = j[key];
  if (j.contains("max_output_tokens"))
    chat["max_completion_tokens"] = j["max_output_tokens"];
  if (j.contains("max_tool_calls"))
    require(integer(j, "max_tool_calls", 0) >= 1, "Invalid max_tool_calls");
  const Json tools = j.value("tools", Json::array());
  require(tools.is_array(), "tools must be an array");
  chat["tools"] = Json::array();
  for (auto tool : tools) {
    fields(tool, {"type", "name", "description", "parameters", "strict"});
    require(tool.at("type") == "function", "Only client-executed function tools are available");
    tool.erase("type");
    if (!tool.contains("parameters") || tool["parameters"].is_null())
      tool["parameters"] = object_schema(Json::object(), Json::array());
    chat["tools"].push_back({{"type", "function"}, {"function", tool}});
  }
  const auto choice = j.value("tool_choice", Json("auto"));
  if (choice.is_object()) {
    fields(choice, {"type", "name"});
    require(choice.at("type") == "function", "Unsupported tool choice");
    chat["tool_choice"] = {{"type", "function"}, {"function", {{"name", choice.at("name")}}}};
  } else
    chat["tool_choice"] = choice;
  const Json text = j.value("text", Json{{"format", {{"type", "text"}}}});
  fields(text, {"format", "verbosity"});
  if (text.contains("verbosity")) {
    const auto level = text["verbosity"].get<std::string>();
    require(level == "low" || level == "medium" || level == "high", "Invalid text verbosity");
    if (level != "medium")
      add_instruction(chat["messages"],
                      level == "low" ? "Keep the answer concise." : "Give a detailed answer.");
  }
  if (text.contains("format")) {
    auto format = text.at("format");
    if (format.at("type") == "json_schema") {
      format.erase("type");
      chat["response_format"] = {{"type", "json_schema"}, {"json_schema", format}};
    } else
      chat["response_format"] = format;
  }
  chat["chat_template_kwargs"] = {{"enable_thinking", false}};
  Json formatted;
  auto request =
      parse_openai(chat.dump(), context, model, Api::chat, truncation == "auto" ? &formatted : nullptr);
  Json body = {{"id", response_id("resp_")},
               {"object", "response"},
               {"created_at", std::time(nullptr)},
               {"completed_at", nullptr},
               {"status", "queued"},
               {"error", nullptr},
               {"incomplete_details", nullptr},
               {"output", Json::array()},
               {"model", model},
               {"usage", nullptr},
               {"tools", tools},
               {"tool_choice", choice},
               {"parallel_tool_calls", false},
               {"instructions", nullptr},
               {"previous_response_id", nullptr},
               {"metadata", metadata},
               {"text", text},
               {"temperature", request.spec.sampling.temperature},
               {"top_p", request.spec.sampling.top_p},
               {"max_output_tokens", request.spec.max_new_tokens},
               {"store", store},
               {"background", background},
               {"truncation", truncation},
               {"service_tier", "default"}};
  for (auto key : {"max_tool_calls", "instructions", "previous_response_id", "reasoning",
                   "prompt_cache_key", "prompt_cache_retention", "safety_identifier", "user"}) {
    if (j.contains(key))
      body[key] = j[key];
  }
  return {std::move(request), std::move(body), std::move(history), std::move(formatted), store,
          background,         stream};
}

class ResponseRecord {
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  Json body_;
  const std::string item_id_ = response_id("msg_");
  bool part_open_ = false, terminal_ = false;
  std::vector<std::string> events_;

  void emit(const std::string &type, Json event) {
    if (!stream)
      return;
    event["type"] = type;
    event["sequence_number"] = events_.size();
    events_.push_back("event: " + type + "\ndata: " + event.dump() + "\n\n");
    changed_.notify_all();
  }
  Json location() const { return {{"item_id", item_id_}, {"output_index", 0}, {"content_index", 0}}; }
  void open_text() {
    if (part_open_)
      return;
    emit("response.output_item.added",
         {{"output_index", 0}, {"item", response_message(item_id_, "in_progress", Json::array())}});
    auto part = location();
    part["part"] = response_text("");
    emit("response.content_part.added", part);
    part_open_ = true;
  }
  void text_delta(const std::string &delta) {
    open_text();
    auto event = location();
    event["delta"] = delta;
    event["logprobs"] = Json::array();
    emit("response.output_text.delta", event);
  }

public:
  const Json history;
  const bool background, stream;
  std::atomic<bool> cancel_requested{false};
  explicit ResponseRecord(PreparedResponse &p)
      : body_(std::move(p.body)), history(std::move(p.history)), background(p.background),
        stream(p.stream) {}
  Json snapshot() const {
    std::lock_guard lock(mutex_);
    return body_;
  }
  bool finished() const {
    std::lock_guard lock(mutex_);
    return terminal_;
  }
  void start() {
    std::lock_guard lock(mutex_);
    body_["status"] = "in_progress";
    emit("response.created", {{"response", body_}});
    emit("response.in_progress", {{"response", body_}});
  }
  void delta(const std::string &text) {
    if (!stream || text.empty())
      return;
    std::lock_guard lock(mutex_);
    text_delta(text);
  }
  void finish(const Reply &reply) {
    std::lock_guard lock(mutex_);
    std::string status;
    if (reply.status == 200) {
      const auto &choice = reply.body.at("choices")[0], &msg = choice.at("message"),
                 &usage = reply.body.at("usage");
      const bool limited = choice.at("finish_reason") == "length";
      status = limited ? "incomplete" : "completed";
      if (limited)
        body_["incomplete_details"] = {{"reason", "max_output_tokens"}};
      body_["usage"] = {{"input_tokens", usage.at("prompt_tokens")},
                        {"input_tokens_details", usage.at("prompt_tokens_details")},
                        {"output_tokens", usage.at("completion_tokens")},
                        {"output_tokens_details", {{"reasoning_tokens", 0}}},
                        {"total_tokens", usage.at("total_tokens")}};
      if (msg.at("content").is_string()) {
        const auto text = msg["content"].get<std::string>();
        if (stream && !part_open_) {
          open_text();
          if (!text.empty())
            text_delta(text);
        }
        const auto part = response_text(text);
        auto event = location();
        event["text"] = text;
        event["logprobs"] = Json::array();
        emit("response.output_text.done", event);
        event = location();
        event["part"] = part;
        emit("response.content_part.done", event);
        const auto item = response_message(item_id_, status, Json::array({part}));
        body_["output"].push_back(item);
        emit("response.output_item.done", {{"output_index", 0}, {"item", item}});
      }
      if (msg.contains("tool_calls")) {
        const auto index = body_["output"].size();
        const auto &call = msg["tool_calls"][0], &fn = call.at("function");
        Json item = {{"type", "function_call"}, {"id", response_id("fc_")}, {"call_id", call.at("id")},
                {"name", fn.at("name")},   {"arguments", ""},          {"status", "in_progress"}};
        emit("response.output_item.added", {{"output_index", index}, {"item", item}});
        item["arguments"] = fn.at("arguments");
        item["status"] = "completed";
        Json event = {{"item_id", item["id"]}, {"output_index", index}, {"delta", fn.at("arguments")}};
        emit("response.function_call_arguments.delta", event);
        event.erase("delta");
        event["arguments"] = fn.at("arguments");
        event["name"] = fn.at("name");
        emit("response.function_call_arguments.done", event);
        body_["output"].push_back(item);
        emit("response.output_item.done", {{"output_index", index}, {"item", item}});
      }
    } else {
      status = reply.status == 499 ? "cancelled" : "failed";
      if (status == "failed")
        body_["error"] = {{"code", "server_error"}, {"message", reply.body.at("error").at("message")}};
    }
    body_["status"] = status;
    if (status == "completed")
      body_["completed_at"] = std::time(nullptr);
    if (status == "cancelled")
      emit("error",
           {{"code", "response_cancelled"}, {"message", "Response was cancelled"}, {"param", nullptr}});
    else
      emit("response." + status, {{"response", body_}});
    terminal_ = true;
    changed_.notify_all();
  }
  bool next(std::size_t &cursor, std::string &wire) {
    std::unique_lock lock(mutex_);
    changed_.wait_for(lock, std::chrono::seconds(1),
                      [&] { return cursor < events_.size() || terminal_; });
    if (cursor < events_.size()) {
      wire = events_[cursor++];
      return true;
    }
    wire.clear();
    return !terminal_;
  }
  Json wait_terminal() {
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return terminal_; });
    return body_;
  }
};

class ResponseStore {
  std::mutex mutex_;
  std::deque<std::pair<std::string, std::shared_ptr<ResponseRecord>>> records_;

public:
  bool insert(const std::string &id, const std::shared_ptr<ResponseRecord> &record) {
    std::lock_guard lock(mutex_);
    if (records_.size() >= 32) {
      auto it = std::find_if(records_.begin(), records_.end(),
                             [](const auto &p) { return p.second->finished(); });
      if (it == records_.end())
        return false;
      records_.erase(it);
    }
    records_.emplace_back(id, record);
    return true;
  }
  std::shared_ptr<ResponseRecord> find(const std::string &id) {
    std::lock_guard lock(mutex_);
    for (const auto &p : records_)
      if (p.first == id)
        return p.second;
    return {};
  }
  std::optional<Json> find_item(const std::string &id) {
    std::lock_guard lock(mutex_);
    for (auto it = records_.rbegin(); it != records_.rend(); ++it) {
      const auto &record = it->second;
      for (const auto &item : record->history)
        if (item.at("id") == id)
          return item;
      const auto body = record->snapshot();
      for (const auto &item : body.at("output"))
        if (item.at("id") == id)
          return item;
    }
    return {};
  }
  bool erase(const std::string &id) {
    std::lock_guard lock(mutex_);
    return std::erase_if(records_, [&](const auto &p) { return p.first == id; }) != 0;
  }
};
} // namespace qwen35x::server
