#include "qwen35x/cpu/q4_0.h"
#include "qwen35x/cpu/q4_dot4.h"
#include "qwen35x/cpu/q4_h128.h"
#include "qwen35x/weights/q4_h128_artifact.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool expect(const bool condition, const char * message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

} // namespace

int main() {
  namespace fs = std::filesystem;
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path path = fs::temp_directory_path() /
    ("qwen35x-q4-h128-artifact-" + std::to_string(nonce) + ".bin");

  qwen35x::Q4H128ArtifactMetadata metadata;
  metadata.num_hidden_layers = 24;
  metadata.hidden_size = 1024;
  metadata.intermediate_size = 3584;
  metadata.vocabulary_size = 248320;
  metadata.sign_seed = qwen35x::cpu::q4_h128_default_sign_seed;

  qwen35x::Q4H128TensorInfo norm;
  norm.name = "norm.weight";
  norm.shape = {128};
  norm.encoding = qwen35x::Q4H128TensorEncoding::f32;

  qwen35x::Q4H128TensorInfo projection;
  projection.name = "projection.weight";
  projection.shape = {8, 128};
  projection.encoding = qwen35x::Q4H128TensorEncoding::q4_h128;
  projection.transform_size = 128;
  projection.scale_group = 32;
  projection.sign_seed = metadata.sign_seed;

  std::vector<float> norm_data(128);
  for (std::size_t index = 0; index < norm_data.size(); ++index) {
    norm_data[index] = static_cast<float>(index) * 0.03125F;
  }
  std::vector<qwen35x::cpu::Q4_0Block> projection_data(
    8 * 128 / qwen35x::cpu::q4_0_values_per_block);
  for (std::size_t block = 0; block < projection_data.size(); ++block) {
    projection_data[block].d = static_cast<std::uint16_t>(0x3400U + block);
    for (std::size_t index = 0; index < 16; ++index) {
      projection_data[block].qs[index] = static_cast<std::uint8_t>(block + 3 * index);
    }
  }

  bool ok = true;
  std::string error;
  auto packed_projection = projection;
  packed_projection.name = "packed.projection.weight";
  packed_projection.encoding = qwen35x::Q4H128TensorEncoding::q4_h128_cpu_x8;
  auto packed_embedding = packed_projection;
  packed_embedding.name = "packed.embedding.weight";
  packed_embedding.encoding = qwen35x::Q4H128TensorEncoding::q4_0_cpu_x8;
  packed_embedding.transform_size = 0;
  packed_embedding.sign_seed = 0;
  auto dot4_projection = packed_projection;
  dot4_projection.name = "dot4.projection.weight";
  dot4_projection.encoding = qwen35x::Q4H128TensorEncoding::q4_h128_cpu_dot4;
  auto dot4_embedding = packed_embedding;
  dot4_embedding.name = "dot4.embedding.weight";
  dot4_embedding.encoding = qwen35x::Q4H128TensorEncoding::q4_0_cpu_dot4;
  std::vector<qwen35x::cpu::Q4_0BlockX8> dot4_data(4);
  qwen35x::cpu::q4_dot4_pack_rows_8(projection_data.data(), dot4_data.data(), 8, 4);
  std::vector<qwen35x::cpu::Q4_0BlockX8> packed_data(4);
  qwen35x::cpu::q4_0_pack_rows_8(projection_data.data(), packed_data.data(), 8, 4);
  for (const auto encoding : {packed_projection.encoding, packed_embedding.encoding, dot4_projection.encoding, dot4_embedding.encoding}) {
    ok = expect(qwen35x::q4_h128_payload_size(encoding, {7, 128}, error) == 0,
                "packed encoding accepted incomplete row tile") && ok;
    ok = expect(qwen35x::q4_h128_payload_size(encoding, {8, 129}, error) == 0,
                "packed encoding accepted incomplete column tile") && ok;
    ok = expect(qwen35x::q4_h128_payload_size(encoding, {1024}, error) == 0,
                "packed encoding accepted non-matrix") && ok;
  }
  ok = expect(qwen35x::q4_h128_payload_size(packed_projection.encoding, {8, 32}, error) == 0,
              "packed H128 accepted partial transform") && ok;
  ok = expect(qwen35x::q4_h128_payload_size(packed_embedding.encoding, {8, 32}, error) == 144,
              "untransformed packed Q4 rejected 32-column tile") && ok;
  error.clear();
  qwen35x::Q4H128ArtifactWriter writer;
  auto invalid_projection = packed_projection;
  invalid_projection.transform_size = 64;
  ok = expect(!writer.open(path.string(), metadata, {invalid_projection}, error),
              "packed H128 accepted incompatible transform metadata") && ok;
  auto invalid_embedding = packed_embedding;
  invalid_embedding.scale_group = 64;
  ok = expect(!writer.open(path.string(), metadata, {invalid_embedding}, error),
              "packed embedding accepted incompatible scale group") && ok;
  error.clear();
  ok = expect(writer.open(path.string(), metadata, {norm, projection, packed_projection, packed_embedding, dot4_projection, dot4_embedding}, error),
              error.c_str()) && ok;
  ok = expect(writer.write_tensor(
                norm.name, norm_data.data(), norm_data.size() * sizeof(float), error),
              error.c_str()) && ok;
  ok = expect(writer.write_tensor(
                projection.name, projection_data.data(),
                projection_data.size() * sizeof(qwen35x::cpu::Q4_0Block), error),
              error.c_str()) && ok;
  for (const auto & info : {packed_projection, packed_embedding, dot4_projection, dot4_embedding}) {
    const auto & payload = qwen35x::q4_h128_encoding_dot4(info.encoding) ? dot4_data : packed_data;
    ok = expect(writer.write_tensor(info.name, payload.data(),
                  packed_data.size() * sizeof(packed_data[0]), error), error.c_str()) && ok;
  }
  ok = expect(writer.finalize(error), error.c_str()) && ok;

  qwen35x::Q4H128ArtifactReader reader;
  ok = expect(reader.open(path.string(), error), error.c_str()) && ok;
  ok = expect(reader.metadata().hidden_size == metadata.hidden_size,
              "artifact metadata mismatch") && ok;
  const qwen35x::Q4H128TensorInfo * loaded = reader.find_tensor(projection.name);
  ok = expect(loaded != nullptr && loaded->encoding == projection.encoding &&
                loaded->shape == projection.shape && loaded->data_offset % 64 == 0,
              "artifact tensor metadata mismatch") && ok;
  std::vector<std::uint8_t> bytes;
  ok = expect(reader.read_tensor_bytes(projection.name, bytes, error), error.c_str()) && ok;
  ok = expect(bytes.size() == projection_data.size() * sizeof(qwen35x::cpu::Q4_0Block) &&
                std::memcmp(bytes.data(), projection_data.data(), bytes.size()) == 0,
              "artifact tensor payload mismatch") && ok;
  const std::uint64_t projection_offset = loaded == nullptr ? 0 : loaded->data_offset;
  const auto * packed_info = reader.find_tensor(packed_projection.name);
  const std::uint64_t packed_offset = packed_info == nullptr ? 0 : packed_info->data_offset;
  std::vector<qwen35x::cpu::Q4_0BlockX8> direct(4);
  for (const auto & info : {packed_projection, packed_embedding, dot4_projection, dot4_embedding}) {
    const auto & payload = qwen35x::q4_h128_encoding_dot4(info.encoding) ? dot4_data : packed_data;
    ok = expect(reader.read_tensor_into(info.name, direct.data(),
                  direct.size() * sizeof(direct[0]), error), error.c_str()) && ok;
    ok = expect(std::memcmp(direct.data(), payload.data(), direct.size() * sizeof(direct[0])) == 0,
                "direct CPU-packed payload differs") && ok;
  }
  ok = expect(!reader.read_tensor_into(packed_projection.name, direct.data(), 1, error),
              "direct read accepted wrong buffer size") && ok;
  ok = expect(!reader.read_tensor_into(packed_projection.name, nullptr, 576, error),
              "direct read accepted null buffer") && ok;
  reader.close();

  // A payload mutation must pass structural indexing but fail checksum
  // validation when the tensor is read.
  if (projection_offset != 0) {
    std::fstream corrupt(path, std::ios::binary | std::ios::in | std::ios::out);
    corrupt.seekg(static_cast<std::streamoff>(projection_offset + 3), std::ios::beg);
    char value = 0;
    corrupt.read(&value, 1);
    corrupt.clear();
    corrupt.seekp(static_cast<std::streamoff>(projection_offset + 3), std::ios::beg);
    value ^= 0x5a;
    corrupt.write(&value, 1);
  }
  ok = expect(reader.open(path.string(), error), "mutated artifact header was rejected") && ok;
  ok = expect(!reader.read_tensor_bytes(projection.name, bytes, error),
              "mutated payload passed checksum validation") && ok;
  reader.close();
  if (packed_offset != 0) {
    std::fstream corrupt(path, std::ios::binary | std::ios::in | std::ios::out);
    corrupt.seekp(static_cast<std::streamoff>(packed_offset), std::ios::beg);
    const char value = 0x7f;
    corrupt.write(&value, 1);
  }
  ok = expect(reader.open(path.string(), error), "packed corruption broke header") && ok;
  ok = expect(!reader.read_tensor_into(packed_projection.name, direct.data(),
                direct.size() * sizeof(direct[0]), error),
              "direct read accepted corrupted packed payload") && ok;
  reader.close();
  fs::remove(path);

  if (ok) {
    std::cout << "Q4_H128 artifact tests passed\n";
    return 0;
  }
  return 1;
}
