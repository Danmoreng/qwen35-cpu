#include "server_protocol.h"
#include <fstream>
#include <iostream>
using namespace qwen35x::server;
void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
  bool rejected = false;
  try { f(); } catch (const std::exception &) { rejected = true; }
  check(rejected, "Invalid request accepted");
}
int main(int argc, char **argv) try {
  check(argc == 2, "Expected template fixtures");
  std::ifstream file(argv[1]); Json fixtures; file >> fixtures;
  for (const auto &c : fixtures.at("cases"))
    check(chat_prompt(c.at("messages")) == c.at("prompt").get<std::string>(), "Qwen template differs from pinned Jinja reference");
  const Json base = {{"model", "test"}, {"messages", Json::array({{{"role", "user"}, {"content", "Hello"}}})},
                     {"temperature", 0}, {"max_completion_tokens", 16}};
  const auto request = parse(base.dump(), 256, "test", Api::chat);
  check(request.spec.max_new_tokens == 16 && request.spec.sampling.temperature == 0, "Sampling/limits lost");
  check(request.prompt == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "Generation prefix differs");
  for (const auto &patch : std::vector<Json>{
      {{"model", "wrong"}}, {{"messages", Json::array()}}, {{"max_tokens", 12}},
      {{"max_completion_tokens", true}}, {{"max_completion_tokens", -1}},
      {{"max_completion_tokens", 257}}, {{"max_completion_tokens", 18446744073709551615ULL}},
      {{"n", true}}, {{"temperature", "0"}}, {{"stream", 1}}, {{"stream", true}},
      {{"tools", Json::array()}}, {{"top_p", 0}}, {{"unknown", 1}}}) {
    auto j = base; j.update(patch);
    rejects([&] { parse(j.dump(), 256, "test", Api::chat); });
  }
  for (const auto &messages : std::vector<Json>{
      Json::array({{{"role", "assistant"}, {"content", "No user"}}}),
      Json::array({{{"role", "user"}, {"content", "Hi"}}, {{"role", "system"}, {"content", "Late"}}}),
      Json::array({{{"role", "user"}, {"content", Json::array({{{"type", "image_url"}, {"image_url", "x"}}})}}}),
      Json::array({{{"role", "tool"}, {"content", "result"}}}),
      Json::array({{{"role", "user"}, {"content", nullptr}}})})
    rejects([&] { chat_prompt(messages); });
  const auto developer = chat_prompt(Json::array({{{"role", "system"}, {"content", "First"}},
      {{"role", "developer"}, {"content", "Second"}}, {{"role", "user"}, {"content", "Hello"}}}));
  check(developer.starts_with("<|im_start|>system\nFirst\n\nSecond<|im_end|>\n"), "Instruction normalization lost");
  const auto reply = completion(Api::chat, "test", 12, 1, "Hello", "stop", 10, 2);
  check(reply["object"] == "chat.completion" && reply["choices"][0]["message"]["role"] == "assistant" &&
        reply["usage"]["total_tokens"] == 12, "Chat response contract differs");
  const auto legacy = parse(R"({"prompt":"Hello"})", 256, "test", Api::completion);
  check(legacy.spec.sampling.temperature == 0 && legacy.spec.sampling.top_k == 20 &&
        legacy.spec.sampling.top_p == .8f, "Legacy sampling defaults changed");
  std::cout << fixtures["cases"].size() << " reference template cases and server protocol checks passed\n";
  return 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
