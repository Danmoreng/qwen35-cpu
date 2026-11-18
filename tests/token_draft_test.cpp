#include "../src/runtime/token_draft.h"
#include <iostream>
using namespace qwen35x;

int main() {
  CpuRequestSpec spec;
  spec.prompt_tokens = {10,11,12,20,21,22,23};
  spec.draft_mode = CpuDraftMode::ngram;
  detail::TokenDraft ngram;
  std::vector<std::int32_t> generated{10,11,12};
  auto p = ngram.propose(spec, generated, 3);
  if (p.count != 3 || p.tokens != std::array<std::int32_t,3>{20,21,22}) return 1;
  generated = {10,11,12,99,81,82,83,84,81,82,83};
  p = ngram.propose(spec, generated, 1);
  if (p.count != 1 || p.tokens[0] != 84) return 2;
  detail::TokenDraft copy;
  spec.draft_mode = CpuDraftMode::input;
  spec.draft_source_tokens = {41,42,43,44,45,46,47,48,49};
  generated = {41};
  p = copy.propose(spec, generated, 3);
  if (p.count != 3 || p.tokens[0] != 42) return 3;
  copy.accepted(p, 1);
  generated = {41,42,90,45,46};
  p = copy.propose(spec, generated, 3);
  if (p.count != 3 || p.tokens != std::array<std::int32_t,3>{47,48,49}) return 4;
  std::cout << "Prompt history, generated history and source realignment passed\n";
}
