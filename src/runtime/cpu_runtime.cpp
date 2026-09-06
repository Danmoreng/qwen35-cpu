#include "qwen35x/runtime/cpu_engine.h"

#include "qwen35x/cpu/activation.h"

#include "qwen35x/cpu/q4_0.h"
#include "qwen35x/cpu/q4_dot4.h"
#include "qwen35x/cpu/q4_h128.h"
#include "qwen35x/cpu/q8_0.h"
#include "qwen35x/cpu/executor.h"
#include "qwen35x/cpu/delta_net.h"
#include "qwen35x/cpu/full_attention.h"
#include "qwen35x/cpu/kv_pages.h"
#include "qwen35x/cpu/full_attention_tiled.h"
#include "qwen35x/weights/q4_h128_artifact.h"
#include "qwen35x/weights/safetensors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <span>
#include <unordered_set>
#include <unordered_map>

namespace qwen35x {
#include "weights.inl"
#include "layers.inl"
#include "forward.inl"
#include "prefill.inl"
#include "cpu_engine.inl"
}
