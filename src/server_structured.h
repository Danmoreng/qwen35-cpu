#pragma once
#include "server_grammar.h"
#include <map>

namespace qwen35x::server {
inline bool tool_name(const std::string &name) {
  return !name.empty() && name.size() <= 64 && std::all_of(name.begin(), name.end(), [](unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
  });
}
inline Json object_schema(const Json &properties, const Json &required) {
  return {{"type", "object"}, {"properties", properties}, {"required", required}, {"additionalProperties", false}};
}
inline void add_instruction(Json &messages, const std::string &instruction) {
  if (!messages.empty() && (messages[0].at("role") == "system" || messages[0].at("role") == "developer"))
    messages[0]["content"] = instruction + "\n\n" + content(messages[0].at("content"));
  else messages.insert(messages.begin(), Json({{"role", "system"}, {"content", instruction}}));
}
inline Json tool_history(const Json &messages) {
  require(messages.is_array(), "messages must be an array");
  Json normalized = Json::array();
  std::map<std::string, std::string> pending;
  std::set<std::string> used;
  for (const auto &message : messages) {
    const auto role = message.at("role").get<std::string>();
    if (role == "tool") {
      fields(message, {"role", "content", "tool_call_id"});
      const auto id = message.at("tool_call_id").get<std::string>();
      require(pending.contains(id), "Tool result must reference an unanswered tool call");
      (void)content(message.at("content"));
      normalized.push_back(message);
      pending.erase(id);
    } else {
      require(pending.empty(), "Supply a result for each tool call before the next turn");
      if (message.contains("tool_calls")) {
        fields(message, {"role", "content", "tool_calls", "reasoning_content"});
        require(role == "assistant" && message["tool_calls"].is_array() && message["tool_calls"].size() == 1,
                "Only one assistant tool call per turn is supported");
        const auto &call = message["tool_calls"][0];
        fields(call, {"id", "type", "function"}); fields(call.at("function"), {"name", "arguments"});
        const auto id = call.at("id").get<std::string>(), name = call["function"].at("name").get<std::string>();
        require(!id.empty() && id.size() <= 128 && used.insert(id).second && call.at("type") == "function" && tool_name(name),
                "Invalid/duplicate tool call identity");
        const auto arguments = Json::parse(call["function"].at("arguments").get<std::string>());
        require(arguments.is_object(), "Tool arguments must be a JSON object");
        auto row = message;
        if (!row.contains("content")) row["content"] = nullptr;
        row["tool_calls"][0]["function"]["arguments"] = arguments;
        normalized.push_back(std::move(row)); pending.emplace(id, name);
      } else normalized.push_back(message);
    }
  }
  require(pending.empty(), "Tool calls require corresponding results");
  return normalized;
}
inline Request parse_openai(const std::string &body, std::size_t context, const std::string &model, Api api,
                            Json *formatted_messages = nullptr) {
  auto j = Json::parse(body);
  require(j.is_object(), "Expected JSON object");
  std::shared_ptr<const ByteGrammar> grammar;
  bool tool_mode = false;
  std::string instruction;
  Json offered = Json::array();
  if (j.contains("response_format")) {
    const auto format = j["response_format"]; j.erase("response_format");
    fields(format, {"type", "json_schema"});
    const auto type = format.at("type").get<std::string>();
    if (type == "json_object") {
      fields(format, {"type"});
      grammar = std::make_shared<ByteGrammar>(JsonGrammarBuilder().object());
      instruction = "Reply with one valid JSON object only. Do not use Markdown fences.";
    } else if (type == "json_schema") {
      const auto &spec = format.at("json_schema"); fields(spec, {"name", "description", "schema", "strict"});
      require(tool_name(spec.at("name").get<std::string>()), "Invalid output schema name");
      if (spec.contains("strict")) require(spec["strict"].is_boolean(), "strict must be boolean");
      if (spec.contains("description")) require(spec["description"].is_string(), "description must be text");
      grammar = std::make_shared<ByteGrammar>(JsonGrammarBuilder().build(spec.at("schema")));
      instruction = "Reply with JSON only, matching this schema: " + spec["schema"].dump();
    } else {
      fields(format, {"type"}); require(type == "text", "Unsupported response_format");
    }
  }
  if (api == Api::chat) j["messages"] = tool_history(j.at("messages"));
  if (j.contains("tools")) {
    require(api == Api::chat && j["tools"].is_array() && j["tools"].size() <= 8, "Chat supports up to eight tools");
    Json definitions = Json::object();
    for (const auto &tool : j["tools"]) {
      fields(tool, {"type", "function"}); require(tool.at("type") == "function", "Only function tools are supported");
      const auto &fn = tool.at("function"); fields(fn, {"name", "description", "parameters", "strict"});
      const auto name = fn.at("name").get<std::string>();
      require(tool_name(name) && !definitions.contains(name), "Invalid/duplicate tool name");
      if (fn.contains("description")) require(fn["description"].is_string(), "Tool description must be text");
      if (fn.contains("strict")) require(fn["strict"].is_boolean(), "strict must be boolean");
      require(fn.at("parameters").at("type") == "object", "Tool parameters require an object schema");
      (void)JsonGrammarBuilder().build(fn["parameters"]);
      definitions[name] = fn["parameters"];
    }
    std::string choice = "auto", named;
    if (j.contains("tool_choice")) {
      const auto &c = j["tool_choice"];
      if (c.is_string()) choice = c.get<std::string>();
      else {
        fields(c, {"type", "function"}); fields(c.at("function"), {"name"});
        require(c.at("type") == "function", "Invalid tool choice");
        choice = "required"; named = c["function"].at("name").get<std::string>();
        require(definitions.contains(named), "Named tool was not offered");
      }
    }
    require(choice == "auto" || choice == "none" || choice == "required", "Unsupported tool_choice");
    require(choice != "required" || !definitions.empty(), "Required tool choice needs tools");
    tool_mode = choice != "none" && !definitions.empty();
    if (tool_mode) {
      require(!grammar, "Tools and response_format cannot be combined");
      offered = j.at("tools");
      if (!named.empty()) definitions = Json::object({{named, definitions.at(named)}});
      grammar = std::make_shared<ByteGrammar>(JsonGrammarBuilder().tool_calls(definitions, choice == "auto"));
    }
    j.erase("tools"); j.erase("tool_choice");
  } else require(!j.contains("tool_choice"), "tool_choice requires tools");
  if (j.contains("parallel_tool_calls")) {
    require(j["parallel_tool_calls"].is_boolean() && j["parallel_tool_calls"] == false,
            "Only parallel_tool_calls=false is supported");
    j.erase("parallel_tool_calls");
  }
  if (!instruction.empty() && api == Api::chat) add_instruction(j["messages"], instruction);
  auto request = parse(j.dump(), context, model, api, offered);
  if (formatted_messages) *formatted_messages = std::move(j.at("messages"));
  require(!grammar || request.stops.empty(), "Text stop sequences cannot be combined with constrained output");
  request.grammar = std::move(grammar); request.tool_mode = tool_mode;
  return request;
}
inline Json tool_value(const std::string &text, const Json &schema) {
  if (schema.contains("anyOf")) {
    for (const auto &option : schema.at("anyOf")) {
      try { return tool_value(text, option); } catch (const std::exception &) {}
    }
    throw std::runtime_error("Unexpected tool parameter type");
  }
  const auto &type = schema.at("type");
  Json value;
  if (type == "string") value = text;
  else if (type == "boolean" && (text == "True" || text == "False")) value = text == "True";
  else if (type == "null" && text == "None") value = nullptr;
  else value = Json::parse(text);
  const auto grammar = JsonGrammarBuilder().build(schema);
  auto state = grammar.initial();
  require(grammar.consume(state, value.dump()) && grammar.complete(state), "Invalid tool parameter value");
  return value;
}
inline void apply_tool_response(Json &reply, Api api, const std::string &text, CpuRequestId id, bool complete, const Json &tools) {
  // Never expose partial, invalid tool arguments as a callable tool. A client
  // can retry a length-limited response with a larger output budget.
  if (!complete) {
    if (api == Api::messages) reply["content"] = Json::array();
    else reply["choices"][0]["message"]["content"] = nullptr;
    return;
  }
  constexpr std::string_view marker = "<tool_call>";
  auto pos = text.find(marker);
  if (pos == std::string::npos) {
    if (api == Api::messages) reply["content"][0]["text"] = text;
    else reply["choices"][0]["message"]["content"] = text;
    return;
  }
  const auto preamble = text.substr(0, pos);
  auto consume = [&](std::string_view literal) {
    require(std::string_view(text).substr(pos).starts_with(literal), "Incomplete Qwen tool call");
    pos += literal.size();
  };
  auto through = [&](std::string_view delimiter) {
    auto end = text.find(delimiter, pos);
    require(end != std::string::npos, "Unterminated Qwen tool call");
    auto value = text.substr(pos, end - pos); pos = end + delimiter.size(); return value;
  };
  consume("<tool_call>\n<function=");
  const auto name = through(">\n");
  const auto definition = std::find_if(tools.begin(), tools.end(),
    [&](const Json &tool) { return tool.at("function").at("name") == name; });
  require(definition != tools.end(), "Unknown Qwen tool call");
  Json arguments = Json::object();
  const auto &properties = definition->at("function").at("parameters").at("properties");
  while (std::string_view(text).substr(pos).starts_with("<parameter=")) {
    consume("<parameter=");
    const auto key = through(">\n");
    arguments[key] = tool_value(through("\n</parameter>\n"), properties.at(key));
  }
  consume("</function>\n</tool_call>");
  require(pos == text.size(), "Unexpected suffix after tool call");
  const auto call_id = "call_" + std::to_string(id);
  if (api == Api::messages) {
    reply["stop_reason"] = "tool_use";
    reply["content"] = Json::array();
    if (!preamble.empty()) reply["content"].push_back({{"type", "text"}, {"text", preamble}});
    reply["content"].push_back({{"type", "tool_use"}, {"id", call_id},
      {"name", name}, {"input", arguments}});
  } else {
    auto &choice = reply["choices"][0]; choice["finish_reason"] = "tool_calls";
    choice["message"] = {{"role", "assistant"}, {"content", preamble.empty() ? Json(nullptr) : Json(preamble)},
      {"tool_calls", Json::array({{{"id", call_id}, {"type", "function"},
        {"function", {{"name", name}, {"arguments", arguments.dump()}}}}})}};
  }
}
} // namespace qwen35x::server
