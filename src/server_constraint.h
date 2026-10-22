#pragma once
#include "server_grammar.h"
#include "qwen35x/tokenizer/tokenizer.h"
#include <numeric>

namespace qwen35x::server {
struct TokenVocabulary {
  struct Node {
    std::vector<std::pair<unsigned char, std::size_t>> edges;
    std::vector<std::int32_t> ids;
  };
  std::vector<Node> trie{1};
  std::vector<std::string> bytes;
  std::vector<std::int32_t> ends;
  explicit TokenVocabulary(const QwenTokenizer &tokenizer) {
    bytes.resize(tokenizer.vocab_size());
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      const auto raw = tokenizer.id_to_token(static_cast<std::int32_t>(i));
      if (!raw) continue;
      if (*raw == "<|im_end|>" || *raw == "<|endoftext|>") { ends.push_back(int(i)); continue; }
      if (raw->starts_with("<|") && raw->ends_with("|>")) continue;
      std::string error;
      require(tokenizer.decode({int(i)}, bytes[i], error), error);
      add(int(i));
    }
    require(!ends.empty(), "Constrained generation requires an EOS token");
  }
  // A small synthetic vocabulary is useful for exhaustive constraint tests.
  TokenVocabulary(std::vector<std::string> pieces, std::vector<std::int32_t> eos)
      : bytes(std::move(pieces)), ends(std::move(eos)) {
    for (std::size_t i = 0; i < bytes.size(); ++i)
      if (std::find(ends.begin(), ends.end(), int(i)) == ends.end()) add(int(i));
  }
  void add(int id) {
    require(bytes[id].size() <= 8192, "Tokenizer piece exceeds byte limit");
    if (bytes[id].empty()) return;
    std::size_t node = 0;
    for (unsigned char c : bytes[id]) {
      auto &edges = trie[node].edges;
      auto found = std::find_if(edges.begin(), edges.end(), [&](const auto &edge) { return edge.first == c; });
      if (found != edges.end()) node = found->second;
      else {
        const auto next = trie.size(); edges.emplace_back(c, next);
        trie.emplace_back(); node = next;
      }
    }
    trie[node].ids.push_back(id);
  }
};
class JsonConstraint final : public CpuSamplingConstraint {
  std::shared_ptr<const ByteGrammar> grammar_;
  std::shared_ptr<const TokenVocabulary> vocab_;
  ByteGrammar::State state_;
  bool greedy_;
  bool allows(int token) const {
    if (std::find(vocab_->ends.begin(), vocab_->ends.end(), token) != vocab_->ends.end()) return complete();
    if (token < 0 || std::size_t(token) >= vocab_->bytes.size() || vocab_->bytes[token].empty()) return false;
    auto state = state_;
    return grammar_->consume(state, vocab_->bytes[token]);
  }
  void walk(std::size_t node, const ByteGrammar::State &state, std::vector<bool> &allowed) const {
    for (auto id : vocab_->trie[node].ids) if (std::size_t(id) < allowed.size()) allowed[id] = true;
    for (const auto &[byte, child] : vocab_->trie[node].edges) {
      auto next = grammar_->advance(state, byte);
      if (!next.empty()) walk(child, next, allowed);
    }
  }
public:
  JsonConstraint(std::shared_ptr<const ByteGrammar> grammar, std::shared_ptr<const TokenVocabulary> vocab, bool greedy)
      : grammar_(std::move(grammar)), vocab_(std::move(vocab)), state_(grammar_->initial()), greedy_(greedy) {}
  bool complete() const { return grammar_->complete(state_); }
  bool apply(float *logits, std::size_t count, std::string &error) override {
    try {
      const auto negative = -std::numeric_limits<float>::infinity();
      if (greedy_) {
        // With no repetition penalty, selecting the highest valid raw logit is
        // exactly constrained greedy sampling; avoid walking the entire trie.
        std::vector<int> candidates(count); std::iota(candidates.begin(), candidates.end(), 0);
        auto less = [&](int a, int b) {
          const auto x = std::isfinite(logits[a]) ? logits[a] : negative;
          const auto y = std::isfinite(logits[b]) ? logits[b] : negative;
          return x == y ? a > b : x < y;
        };
        std::make_heap(candidates.begin(), candidates.end(), less);
        int chosen = -1;
        while (!candidates.empty()) {
          const int id = candidates.front();
          if (!std::isfinite(logits[id])) break;
          if (allows(id)) { chosen = id; break; }
          std::pop_heap(candidates.begin(), candidates.end(), less); candidates.pop_back();
        }
        require(chosen >= 0, "No finite token satisfies the output grammar");
        const float value = logits[chosen]; std::fill(logits, logits + count, negative); logits[chosen] = value;
      } else {
        std::vector<bool> allowed(count, false); walk(0, state_, allowed);
        if (complete()) for (auto id : vocab_->ends) if (std::size_t(id) < count) allowed[id] = true;
        bool finite = false;
        for (std::size_t i = 0; i < count; ++i) {
          if (!allowed[i] || !std::isfinite(logits[i])) logits[i] = negative;
          else finite = true;
        }
        require(finite, "No finite token satisfies the output grammar");
      }
      return true;
    } catch (const std::exception &e) { error = e.what(); return false; }
  }
  bool accept(std::int32_t token, std::string &error) override {
    try {
      require(allows(token), "Selected token violates output grammar");
      if (std::find(vocab_->ends.begin(), vocab_->ends.end(), token) == vocab_->ends.end())
        grammar_->consume(state_, vocab_->bytes[token]);
      return true;
    } catch (const std::exception &e) { error = e.what(); return false; }
  }
};
} // namespace qwen35x::server
