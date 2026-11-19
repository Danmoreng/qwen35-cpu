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
  if (copy.record_verification(p, 1, 2)) return 5;
  generated = {41,42,90,45,46};
  p = copy.propose(spec, generated, 3);
  if (p.count != 3 || p.tokens != std::array<std::int32_t,3>{47,48,49}) return 4;
  if (!copy.ready(5) || copy.record_verification(p, 0, 6) ||
      copy.record_verification(p, 0, 7) || !copy.record_verification(p, 0, 8)) return 6;
  if (copy.ready(8) || copy.ready(23) || !copy.ready(24)) return 7;
  for (std::size_t generated = 25; generated <= 28; ++generated)
    if (copy.record_verification(p, 1, generated) || !copy.ready(generated)) return 8;
  for (std::size_t generated = 29; generated <= 31; ++generated)
    if (copy.record_verification(p, 0, generated) || !copy.ready(generated)) return 9;
  if (!copy.record_verification(p, 0, 32) || copy.ready(47) || !copy.ready(48)) return 10;
  std::cout << "Draft lookup, source realignment and acceptance cooldown passed\n";
}
