// Native oracle for the exact browser protocol and chunked execution path.
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "qwen35x/runtime/browser_decision.h"
int main(int argc, char **argv) {
  if (argc < 3) { std::cerr << "Usage: qev_browser_reference MODEL_DIR REQUESTS_JSONL [--threads N] [--weights PATH] [--profile-json PATH]\n"; return 2; }
  int threads = 1; std::string profile;
  for (int i = 3; i < argc; i += 2) {
    if (i + 1 == argc) return 2;
    const std::string option = argv[i];
    if (option == "--threads") threads = std::stoi(argv[i + 1]);
    else if (option == "--profile-json") profile = argv[i + 1];
    else if (option == "--weights") {
      if (!std::filesystem::equivalent(std::filesystem::path(argv[1]) / "model.q35h", argv[i + 1])) return 2;
    } else return 2;
  }
  const std::string loaded = qev_load(argv[1], threads, 2048);
  if (loaded.find("\"error\"") != std::string::npos) { std::cerr << loaded << '\n'; return 1; }
  std::ifstream input(argv[2]); if (!input) return 2;
  std::vector<std::string> requests;
  for (std::string line; std::getline(input, line);) if (!line.empty()) requests.push_back(std::move(line));
  using Json = nlohmann::ordered_json;
  auto run = [&](const std::string &line) {
    const auto start = std::chrono::steady_clock::now();
    std::string result = qev_begin(line.c_str());
    while (result.find("\"status\":\"running\"") != std::string::npos) result = qev_step();
    auto record = Json{{"result", Json::parse(result)}, {"debug", Json::parse(qev_debug())}};
    record["worker_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return record;
  };
  if (!profile.empty() && !requests.empty()) run(requests.front());
  Json outputs = Json::array();
  for (const auto &line : requests) {
    auto record = run(line);
    if (record["result"].contains("error")) { std::cerr << record.dump() << '\n'; return 1; }
    if (profile.empty()) std::cout << record.dump() << '\n';
    else outputs.push_back(std::move(record));
  }
  if (!profile.empty()) {
    Json report = {{"engine", "native-cpu-c-abi"}, {"threads", threads}, {"requests", requests.size()},
      {"session_warmup", true}, {"ready", Json::parse(loaded)}, {"outputs", outputs}};
    for (std::size_t i = 0; i < outputs.size(); ++i) report["request_" + std::to_string(i) + "_ms"] = outputs[i]["worker_ms"];
    std::ofstream file(profile); if (!file) return 2;
    file << report.dump(2) << '\n';
  }
}
