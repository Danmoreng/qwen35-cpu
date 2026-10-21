#!/usr/bin/env bash
# Build an external CPU/WASM comparison fixture, without modifying llama.cpp.
set -euo pipefail
cd "$(dirname "$0")/.."
llama_source=${1:-.cache/llama-current}
json_include=${2:-build-linux/_deps/wasm_json-src/single_include}
repo_root=$(pwd)
source_path=$(realpath "$llama_source")
json_path=$(realpath "$json_include")
mkdir -p build-llama-wasm-benchmark
# Keep upstream compiler probes inside the ignored build directory.
cd build-llama-wasm-benchmark
emcmake cmake -S "$repo_root/tools/llama-wasm-benchmark" -B . \
  -DCMAKE_BUILD_TYPE=Release -DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm32 \
  -DLLAMA_SOURCE="$source_path" -DJSON_INCLUDE_DIR="$json_path"
cmake --build . --target llama_browser --parallel "${BUILD_JOBS:-6}"
cd "$repo_root"
mkdir -p dist/wasm/llama-bench/runtime
cp build-llama-wasm-benchmark/llama_browser.{js,wasm} dist/wasm/llama-bench/runtime/
git -C "$source_path" rev-parse HEAD > dist/wasm/llama-bench/runtime/upstream.txt
cp "$source_path/LICENSE" dist/wasm/llama-bench/runtime/llama-LICENSE
