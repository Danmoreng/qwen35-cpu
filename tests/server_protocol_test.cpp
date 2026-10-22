#include "server_protocol.h"
#include "server_stream.h"
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
      {{"n", true}}, {{"temperature", "0"}}, {{"stream", 1}},
      {{"stop", ""}}, {{"stop", Json::array({"a","b","c","d","e"})}}, {{"stop", 12}},
      {{"stream_options", {{"include_usage", true}}}},
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
  // Every possible byte split, including within UTF-8 and stop markers.
  const std::string wire = "Gr\xc3\xbc\xc3\x9f" "e \xf0\x9f\x91\x8b STOPdiscard";
  for (std::size_t split = 0; split <= wire.size(); ++split) {
    Utf8Decoder decoder; StopMatcher stop({"STOP"}, 8192); std::string text;
    text += stop.Push(decoder.push(wire.substr(0, split)));
    text += stop.Push(decoder.push(wire.substr(split)));
    text += stop.Push(decoder.push({}, true)); text += stop.Finish();
    check(text == "Gr\xc3\xbc\xc3\x9f" "e \xf0\x9f\x91\x8b " && stop.stopped(), "Split UTF-8/stop leaked");
  }
  Utf8Decoder broken;
  check(broken.push("\xf0\x9f").empty(), "Incomplete UTF-8 escaped");
  check(broken.push({}, true) == "\xef\xbf\xbd", "Incomplete UTF-8 not replaced on flush");
  StopMatcher partial({"STOP"}, 64);
  check(partial.Push("hello ST") == "hello " && partial.Finish() == "ST", "Partial stop discarded at length limit");
  StopMatcher overlap({"abc", "bc"}, 64);
  check(overlap.Push("xabctail") == "x" && overlap.stopped(), "Earliest stop not selected");
  StreamQueue queue;
  std::size_t queued = 0;
  while (queue.room()) { queue.push("payload"); ++queued; }
  check(queued > 0 && queued <= StreamQueue::capacity, "Unbounded event queue");
  queue.finish({500, {{"error", "test"}}});
  std::string delta; std::optional<Reply> terminal;
  while (queue.pop(delta, terminal)) --queued;
  check(!queued && terminal && terminal->status == 500, "Terminal state lost behind full queue");
  std::cout << fixtures["cases"].size() << " reference template cases and server protocol checks passed\n";
  return 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
