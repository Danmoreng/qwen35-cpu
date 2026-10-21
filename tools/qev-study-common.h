// Text-only offline qev evaluation protocol. Input token IDs are prepared by Python.
#pragma once
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace qev_study {
struct Job { std::string id, path; };
inline std::vector<std::int32_t> tokens(const std::string &path) {
  std::ifstream file(path); if (!file) throw std::runtime_error("Cannot open tokens: " + path);
  std::string text((std::istreambuf_iterator<char>(file)), {});
  for (auto &c : text) if (c == ',') c = ' ';
  std::istringstream in(text); std::vector<std::int32_t> out; std::int32_t value;
  while (in >> value) out.push_back(value);
  if (!in.eof() || out.empty()) throw std::runtime_error("Invalid token list: " + path);
  return out;
}
inline std::vector<Job> jobs(const std::string &path) {
  std::ifstream file(path); if (!file) throw std::runtime_error("Cannot open jobs");
  std::vector<Job> out; std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    auto tab = line.find('\t'); if (tab == std::string::npos) throw std::runtime_error("Jobs require id TAB token-path");
    Job job{line.substr(0, tab), line.substr(tab + 1)};
    if (job.id.empty() || job.id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != std::string::npos)
      throw std::runtime_error("Unsafe job ID");
    out.push_back(job);
  }
  if (out.empty()) throw std::runtime_error("Empty jobs");
  return out;
}
inline void write(std::ostream &out, const std::string &id, const float *logits, std::size_t count,
                  const std::vector<std::int32_t> &letters) {
  out << id;
  for (auto token : letters) {
    if (token < 0 || std::size_t(token) >= count || !std::isfinite(logits[token]))
      throw std::runtime_error("Invalid option logit");
    out << '\t' << std::setprecision(9) << logits[token];
  }
  out << '\n'; out.flush(); if (!out) throw std::runtime_error("Cannot write option logits");
}
}
