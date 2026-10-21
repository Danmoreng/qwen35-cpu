#!/usr/bin/env bash
# Build local-only browser artifacts. Requires an activated Emscripten SDK.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v emcmake >/dev/null || { echo 'Activate emsdk_env.sh first.' >&2; exit 1; }
mode=${1:-all}
if [ "$mode" = all ]; then
  for variant in scalar simd threads relaxed relaxed-threads fma fma-threads; do
    bash scripts/build-wasm.sh "$variant"
  done
  exit 0
fi
relaxed=OFF
fma=OFF
case "$mode" in
  scalar) simd=OFF; threads=OFF ;;
  simd) simd=ON; threads=OFF ;;
  threads) simd=ON; threads=ON ;;
  relaxed) simd=ON; threads=OFF; relaxed=ON ;;
  relaxed-threads) simd=ON; threads=ON; relaxed=ON ;;
  fma) simd=ON; threads=OFF; relaxed=ON; fma=ON ;;
  fma-threads) simd=ON; threads=ON; relaxed=ON; fma=ON ;;
  *) echo 'Usage: scripts/build-wasm.sh [all|scalar|simd|threads|relaxed|relaxed-threads|fma|fma-threads]' >&2; exit 2 ;;
esac
emcmake cmake -S . -B "build-wasm-$mode" -DCMAKE_BUILD_TYPE=Release \
  -DQWEN35_BUILD_WASM=ON -DQWEN35_WASM_SIMD="$simd" -DQWEN35_WASM_THREADS="$threads" \
  -DQWEN35_WASM_Q4_WORD_SHIFT="${WASM_Q4_WORD_SHIFT:-OFF}" -DQWEN35_WASM_Q4_HALF_ROWS="${WASM_Q4_HALF_ROWS:-OFF}" \
  -DQWEN35_WASM_Q4_HALF_DOTS="${WASM_Q4_HALF_DOTS:-OFF}" \
  -DQWEN35_WASM_Q8_SIMD="${WASM_Q8_SIMD:-ON}" -DQWEN35_WASM_RELAXED_DOT="$relaxed" -DQWEN35_WASM_RELAXED_FMA="$fma"
cmake --build "build-wasm-$mode" --parallel "${BUILD_JOBS:-4}"
node "build-wasm-$mode/wasm_kernel_test.js"
node "build-wasm-$mode/wasm_delta_test.js"
node -e 'require(process.argv[1])()' "./build-wasm-$mode/wasm_quantization_test.js"
mkdir -p "dist/wasm/$mode"
cp "build-wasm-$mode/qev_wasm.js" "build-wasm-$mode/qev_wasm.wasm" "dist/wasm/$mode/"
mkdir -p "dist/wasm/review/quantization/$mode"
cp "build-wasm-$mode/wasm_quantization_test.js" "build-wasm-$mode/wasm_quantization_test.wasm" "dist/wasm/review/quantization/$mode/"
cp tests/wasm_quantization_worker.js "dist/wasm/review/quantization/$mode/worker.js"
cp web/* dist/wasm/
mkdir -p dist/wasm/licenses
cp "build-wasm-$mode/wasm-licenses/"* dist/wasm/licenses/
emcc --version > "dist/wasm/$mode/toolchain.txt"
echo "Local build: dist/wasm/$mode (nothing uploaded)"
