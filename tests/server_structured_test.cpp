#include "server_constraint.h"
#include "server_structured.h"
#include "server_anthropic.h"
#include <iostream>
using namespace qwen35x::server;
void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
  bool rejected = false; try { f(); } catch (const std::exception &) { rejected = true; }
  check(rejected, "Invalid schema/request accepted");
}
bool accepts(const ByteGrammar &grammar, std::string_view value) {
  auto state = grammar.initial();
  return grammar.consume(state, value) && grammar.complete(state);
}
int main() try {
  const Json schema = object_schema({{"name", {{"type", "string"}}},
    {"count", {{"type", "integer"}}}, {"flags", {{"type", "array"}, {"items", {{"type", "boolean"}}},
      {"minItems", 1}, {"maxItems", 2}}}}, {"name", "flags"});
  auto grammar = JsonGrammarBuilder().build(schema);
  for (const auto &text : {R"({"name":"hello","count":-2,"flags":[true,false]})",
      R"( { "name" : "\uD83D\uDC4B", "flags" : [false] } )", R"({"name":"","flags":[true]})"})
    check(accepts(grammar, text), "Valid schema output rejected");
  for (const auto &text : {R"({"name":"x","flags":[]})", R"({"name":"x","flags":[true,true,true]})",
      R"({"name":"x","count":1.5,"flags":[true]})", R"({"flags":[true]})",
      R"({"name":"x","extra":0,"flags":[true]})", R"({"name":"\uD800","flags":[true]})"})
    check(!accepts(grammar, text), "Invalid schema output allowed");
  auto generic = JsonGrammarBuilder().object();
  for (const auto &text : {R"({"a":[1,2.5e-12,null,{"escaped":"\n\"\\"}],"b":true})", "{}", "{\"x\":\"\xc3\xbc\xf0\x9f\x91\x8b\"}"})
    check(accepts(generic, text), "Valid JSON rejected");
  for (const auto &text : {"[]", "{", "{}x", "{\"x\":01}", "{\"x\":NaN}", "{\"x\":\"\xc0\xaf\"}", R"({"x":"\uDC00"})"})
    check(!accepts(generic, text), "Malformed JSON accepted");
  for (const auto &invalid : std::vector<Json>{
      {{"type", "string"}, {"pattern", ".*"}}, {{"type", "integer"}, {"minimum", 0}},
      {{"type", "object"}, {"properties", Json::object()}},
      {{"type", "array"}, {"items", {{"type", "boolean"}}}, {"maxItems", 65}},
      {{"type", "array"}, {"items", {{"type", "boolean"}}}, {"minItems", nullptr}},
      {{"type", "string"}, {"enum", Json::array({1})}}, {{"$ref", "#/x"}},
      {{"anyOf", Json::array({{{"type", "null"}}})}, {"const", 1}}})
    rejects([&] { JsonGrammarBuilder().build(invalid); });
  auto enum_grammar = std::make_shared<ByteGrammar>(JsonGrammarBuilder().build(
    object_schema({{"x", {{"type", "string"}, {"enum", {"yes", "no"}}}}}, {"x"})));
  auto vocab = std::make_shared<TokenVocabulary>(std::vector<std::string>{"{", "\"x\":", "\"yes\"", "\"no\"", "}", "bad", "", "\"y\":"}, std::vector<std::int32_t>{6});
  for (bool greedy : {false, true}) {
    JsonConstraint constraint(enum_grammar, vocab, greedy); std::string error;
    for (int chosen : {0, 1, 2, 4, 6}) {
      std::vector<float> logits(8, 1); logits[5] = 100; logits[7] = 99; logits[chosen] = 2;
      check(constraint.apply(logits.data(), logits.size(), error), error.c_str());
      check(!std::isfinite(logits[5]) && !std::isfinite(logits[7]) && std::isfinite(logits[chosen]), "Mask permits invalid token");
      check(constraint.accept(chosen, error), error.c_str());
    }
    check(constraint.complete(), "Constraint did not reach complete JSON");
  }
  JsonConstraint incomplete(enum_grammar, vocab, false); std::string error;
  check(!incomplete.accept(6, error), "EOS accepted before complete output");
  std::vector<float> empty(8, -std::numeric_limits<float>::infinity());
  check(!incomplete.apply(empty.data(), empty.size(), error), "All-masked logits accepted");
  const Json args = object_schema({{"city", {{"type", "string"}, {"enum", {"Berlin"}}}}}, {"city"});
  const Json tools = Json::array({{{"type", "function"}, {"function", {{"name", "weather"}, {"parameters", args}}}}});
  Json request = {{"model", "test"}, {"messages", Json::array({{{"role", "user"}, {"content", "Weather?"}}})},
    {"tools", tools}, {"tool_choice", "required"}, {"max_tokens", 64}};
  auto parsed = parse_openai(request.dump(), 1024, "test", Api::chat);
  check(parsed.tool_mode && accepts(*parsed.grammar, R"({"type":"tool_call","name":"weather","arguments":{"city":"Berlin"}})"), "Tool envelope not constrained");
  check(!accepts(*parsed.grammar, R"({"type":"tool_call","name":"other","arguments":{"city":"Berlin"}})"), "Unoffered tool allowed");
  check(!accepts(*parsed.grammar, R"({"type":"message","content":"No"})"), "Required tool choice allowed text");
  auto auto_request = request; auto_request["tool_choice"] = "auto";
  const auto automatic = parse_openai(auto_request.dump(), 1024, "test", Api::chat);
  check(accepts(*automatic.grammar, R"({"type":"message","content":"Hello"})"), "Automatic tool choice cannot answer with text");
  auto malformed = request; malformed["stop"] = "x";
  rejects([&] { parse_openai(malformed.dump(), 1024, "test", Api::chat); });
  Json reply = completion(Api::chat, "test", 7, 1, "", "stop", 100, 20);
  apply_tool_response(reply, Api::chat, R"({"type":"tool_call","name":"weather","arguments":{"city":"Berlin"}})", 7, true);
  const auto assistant = reply["choices"][0]["message"];
  check(reply["choices"][0]["finish_reason"] == "tool_calls" && assistant["tool_calls"][0]["function"]["arguments"] == R"({"city":"Berlin"})", "OpenAI tool response malformed");
  auto text_reply = completion(Api::chat, "test", 8, 1, "", "stop", 10, 2);
  apply_tool_response(text_reply, Api::chat, R"({"type":"message","content":"Hello"})", 8, true);
  check(text_reply["choices"][0]["message"]["content"] == "Hello", "Tool text envelope leaked");
  auto anthropic_text = message_response("test", 8, "", "end_turn", nullptr, messages_usage(10, 2));
  apply_tool_response(anthropic_text, Api::messages, R"({"type":"message","content":"Hello"})", 8, true);
  check(anthropic_text["content"][0]["text"] == "Hello" && message_tool_events(anthropic_text).find("text_delta") != std::string::npos,
        "Messages automatic text adapter differs");
  request["messages"].push_back(assistant);
  request["messages"].push_back({{"role", "tool"}, {"tool_call_id", "call_7"}, {"content", "Sunny"}});
  parsed = parse_openai(request.dump(), 1024, "test", Api::chat);
  check(parsed.prompt.find("<tool_response>") != std::string::npos, "Tool result not formatted");
  malformed = request; malformed["messages"].back()["tool_call_id"] = "unknown";
  rejects([&] { parse_openai(malformed.dump(), 1024, "test", Api::chat); });
  const auto messages = parse_messages(Json({{"model", "test"}, {"max_tokens", 64},
    {"messages", Json::array({{{"role", "user"}, {"content", "Weather?"}}})},
    {"tools", Json::array({{{"name", "weather"}, {"input_schema", args}}})},
    {"tool_choice", {{"type", "any"}}}}).dump(), 1024, "test");
  check(messages.prompt == parse_openai(Json({{"model", "test"}, {"max_tokens", 64},
    {"messages", Json::array({{{"role", "user"}, {"content", "Weather?"}}})},
    {"tools", tools}, {"tool_choice", "required"}}).dump(), 1024, "test", Api::chat).prompt,
    "Tool adapters diverge before inference");
  std::cout << "JSON grammar, Unicode, schema bounds, token masks and tool adapters passed\n";
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
