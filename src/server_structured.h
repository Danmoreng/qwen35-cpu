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
inline Json scalar_constant(const std::string &value) { return {{"type", "string"}, {"const", value}}; }
inline Json call_envelope(const std::string &name, const Json &arguments) {
  return {{"type", "tool_call"}, {"name", name}, {"arguments", arguments}};
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
      const Json result = {{"tool_call_id", id}, {"name", pending.at(id)}, {"result", content(message.at("content"))}};
      normalized.push_back({{"role", "user"}, {"content", "<tool_response>\n" + result.dump() + "\n</tool_response>"}});
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
        auto row = message; row.erase("tool_calls");
        auto text = message.contains("content") ? content(message["content"], true) : std::string();
        if (!text.empty()) text += "\n";
        row["content"] = text + call_envelope(name, arguments).dump();
        normalized.push_back(std::move(row)); pending.emplace(id, name);
      } else normalized.push_back(message);
    }
  }
  require(pending.empty(), "Tool calls require corresponding results");
  return normalized;
}
inline Request parse_openai(const std::string &body, std::size_t context, const std::string &model, Api api) {
  auto j = Json::parse(body);
  require(j.is_object(), "Expected JSON object");
  std::shared_ptr<const ByteGrammar> grammar;
  bool tool_mode = false;
  std::string instruction;
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
    std::map<std::string, Json> definitions;
    Json descriptions = Json::array();
    for (const auto &tool : j["tools"]) {
      fields(tool, {"type", "function"}); require(tool.at("type") == "function", "Only function tools are supported");
      const auto &fn = tool.at("function"); fields(fn, {"name", "description", "parameters", "strict"});
      const auto name = fn.at("name").get<std::string>();
      require(tool_name(name) && !definitions.contains(name), "Invalid/duplicate tool name");
      if (fn.contains("description")) require(fn["description"].is_string(), "Tool description must be text");
      if (fn.contains("strict")) require(fn["strict"].is_boolean(), "strict must be boolean");
      require(fn.at("parameters").at("type") == "object", "Tool parameters require an object schema");
      (void)JsonGrammarBuilder().build(fn["parameters"]);
      definitions.emplace(name, fn["parameters"]); descriptions.push_back(fn);
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
      Json alternatives = Json::array();
      for (const auto &[name, parameters] : definitions)
        if (named.empty() || named == name)
          alternatives.push_back(object_schema({{"type", scalar_constant("tool_call")},
            {"name", scalar_constant(name)}, {"arguments", parameters}}, {"type", "name", "arguments"}));
      if (choice == "auto") alternatives.push_back(object_schema({{"type", scalar_constant("message")},
          {"content", {{"type", "string"}}}}, {"type", "content"}));
      grammar = std::make_shared<ByteGrammar>(JsonGrammarBuilder().build({{"anyOf", alternatives}}));
      instruction = "You can call these functions: " + descriptions.dump() +
        "\nReply using exactly one JSON object. To call a tool use {\"type\":\"tool_call\",\"name\":\"FUNCTION_NAME\",\"arguments\":{...}}. " +
        (choice == "auto" ? "If no tool is needed use {\"type\":\"message\",\"content\":\"YOUR ANSWER\"}. " : "You must call a tool. ") +
        (named.empty() ? "" : "Call " + named + ". ") + "Do not include Markdown or text outside the JSON object.";
    }
    j.erase("tools"); j.erase("tool_choice");
  } else require(!j.contains("tool_choice"), "tool_choice requires tools");
  if (j.contains("parallel_tool_calls")) {
    require(j["parallel_tool_calls"].is_boolean() && j["parallel_tool_calls"] == false,
            "Only parallel_tool_calls=false is supported");
    j.erase("parallel_tool_calls");
  }
  if (!instruction.empty() && api == Api::chat) add_instruction(j["messages"], instruction);
  auto request = parse(j.dump(), context, model, api);
  require(!grammar || request.stops.empty(), "Text stop sequences cannot be combined with constrained output");
  request.grammar = std::move(grammar); request.tool_mode = tool_mode;
  return request;
}
inline void apply_tool_response(Json &reply, Api api, const std::string &text, CpuRequestId id, bool complete) {
  // Never expose partial, invalid tool arguments as a callable tool. A client
  // can retry a length-limited response with a larger output budget.
  if (!complete) {
    if (api == Api::messages) reply["content"] = Json::array();
    else reply["choices"][0]["message"]["content"] = nullptr;
    return;
  }
  const auto envelope = Json::parse(text);
  if (envelope.at("type") == "message") {
    if (api == Api::messages) reply["content"][0]["text"] = envelope.at("content");
    else reply["choices"][0]["message"]["content"] = envelope.at("content");
    return;
  }
  const auto call_id = "call_" + std::to_string(id);
  if (api == Api::messages) {
    reply["stop_reason"] = "tool_use";
    reply["content"] = Json::array({{{"type", "tool_use"}, {"id", call_id},
      {"name", envelope.at("name")}, {"input", envelope.at("arguments")}}});
  } else {
    auto &choice = reply["choices"][0]; choice["finish_reason"] = "tool_calls";
    choice["message"] = {{"role", "assistant"}, {"content", nullptr},
      {"tool_calls", Json::array({{{"id", call_id}, {"type", "function"},
        {"function", {{"name", envelope.at("name")}, {"arguments", envelope.at("arguments").dump()}}}}})}};
  }
}
} // namespace qwen35x::server
