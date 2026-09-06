#include "qwen35x/weights/gguf.h"

#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

constexpr std::uint32_t gguf_type_uint8 = 0;
constexpr std::uint32_t gguf_type_int8 = 1;
constexpr std::uint32_t gguf_type_uint16 = 2;
constexpr std::uint32_t gguf_type_int16 = 3;
constexpr std::uint32_t gguf_type_uint32 = 4;
constexpr std::uint32_t gguf_type_int32 = 5;
constexpr std::uint32_t gguf_type_float32 = 6;
constexpr std::uint32_t gguf_type_bool = 7;
constexpr std::uint32_t gguf_type_string = 8;
constexpr std::uint32_t gguf_type_array = 9;
constexpr std::uint32_t gguf_type_uint64 = 10;
constexpr std::uint32_t gguf_type_int64 = 11;
constexpr std::uint32_t gguf_type_float64 = 12;

bool expect(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << "gguf reader test failure: " << message << '\n';
  }
  return condition;
}

template <typename T>
void write_le(std::ofstream & stream, const T value) {
  static_assert(std::is_unsigned_v<T>);
  std::array<char, sizeof(T)> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((value >> (i * 8U)) & static_cast<T>(0xffU));
  }
  stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void write_string(std::ofstream & stream, const std::string_view value) {
  write_le(stream, static_cast<std::uint64_t>(value.size()));
  stream.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_kv_prefix(
  std::ofstream & stream,
  const std::string_view key,
  const std::uint32_t type) {
  write_string(stream, key);
  write_le(stream, type);
}

void pad_to(std::ofstream & stream, const std::uint64_t alignment) {
  const auto position = static_cast<std::uint64_t>(static_cast<std::streamoff>(stream.tellp()));
  const std::uint64_t padding = (alignment - position % alignment) % alignment;
  for (std::uint64_t i = 0; i < padding; ++i) {
    stream.put('\0');
  }
}

enum class SyntheticVariant {
  valid,
  truncated_payload,
  invalid_q8_row,
  invalid_alignment,
};

bool write_synthetic_gguf(
  const std::filesystem::path & path,
  const SyntheticVariant variant) {
  std::ofstream stream(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!stream) {
    return false;
  }

  stream.write("GGUF", 4);
  write_le(stream, std::uint32_t{3});
  write_le(stream, std::uint64_t{3});
  write_le(stream, std::uint64_t{15});

  write_kv_prefix(stream, "general.alignment", gguf_type_uint32);
  write_le(stream, variant == SyntheticVariant::invalid_alignment ? std::uint32_t{3} : std::uint32_t{32});

  write_kv_prefix(stream, "test.u8", gguf_type_uint8);
  write_le(stream, std::uint8_t{1});
  write_kv_prefix(stream, "test.i8", gguf_type_int8);
  write_le(stream, std::uint8_t{0xff});
  write_kv_prefix(stream, "test.u16", gguf_type_uint16);
  write_le(stream, std::uint16_t{2});
  write_kv_prefix(stream, "test.i16", gguf_type_int16);
  write_le(stream, std::uint16_t{0xfffe});
  write_kv_prefix(stream, "test.u32", gguf_type_uint32);
  write_le(stream, std::uint32_t{3});
  write_kv_prefix(stream, "test.i32", gguf_type_int32);
  write_le(stream, std::uint32_t{0xfffffffdU});
  write_kv_prefix(stream, "test.f32", gguf_type_float32);
  write_le(stream, std::bit_cast<std::uint32_t>(1.25F));
  write_kv_prefix(stream, "test.bool", gguf_type_bool);
  write_le(stream, std::uint8_t{1});
  write_kv_prefix(stream, "test.string", gguf_type_string);
  write_string(stream, "metadata safely skipped");

  write_kv_prefix(stream, "test.u16_array", gguf_type_array);
  write_le(stream, gguf_type_uint16);
  write_le(stream, std::uint64_t{3});
  write_le(stream, std::uint16_t{10});
  write_le(stream, std::uint16_t{20});
  write_le(stream, std::uint16_t{30});

  write_kv_prefix(stream, "test.u64", gguf_type_uint64);
  write_le(stream, std::uint64_t{4});
  write_kv_prefix(stream, "test.i64", gguf_type_int64);
  write_le(stream, std::uint64_t{0xfffffffffffffffcULL});
  write_kv_prefix(stream, "test.f64", gguf_type_float64);
  write_le(stream, std::bit_cast<std::uint64_t>(2.5));

  write_kv_prefix(stream, "test.string_array", gguf_type_array);
  write_le(stream, gguf_type_string);
  write_le(stream, std::uint64_t{2});
  write_string(stream, "alpha");
  write_string(stream, "beta");

  write_string(stream, "norm");
  write_le(stream, std::uint32_t{1});
  write_le(stream, std::uint64_t{4});
  write_le(stream, std::uint32_t{0});
  write_le(stream, std::uint64_t{0});

  write_string(stream, "matrix");
  write_le(stream, std::uint32_t{2});
  write_le(
    stream,
    variant == SyntheticVariant::invalid_q8_row ? std::uint64_t{31} : std::uint64_t{32});
  write_le(stream, std::uint64_t{2});
  write_le(stream, std::uint32_t{8});
  write_le(stream, std::uint64_t{32});

  write_string(stream, "matrix_q4");
  write_le(stream, std::uint32_t{2});
  write_le(stream, std::uint64_t{32});
  write_le(stream, std::uint64_t{2});
  write_le(stream, std::uint32_t{2});
  write_le(stream, std::uint64_t{128});

  pad_to(stream, 32);
  constexpr std::array<float, 4> f32_values{1.0F, -2.5F, 0.0F, 4.25F};
  for (const float value : f32_values) {
    write_le(stream, std::bit_cast<std::uint32_t>(value));
  }
  for (std::uint32_t i = 0; i < 16; ++i) {
    stream.put('\0');
  }

  for (std::uint32_t block = 0; block < 2; ++block) {
    write_le(stream, std::uint16_t{0x3c00});
    for (std::int32_t i = 0; i < 32; ++i) {
      const std::int8_t quant = static_cast<std::int8_t>(block == 0 ? i - 16 : 31 - i);
      stream.put(static_cast<char>(quant));
    }
  }
  for (std::uint32_t i = 0; i < 28; ++i) {
    stream.put('\0');
  }
  for (std::uint32_t block = 0; block < 2; ++block) {
    write_le(stream, std::uint16_t{0x3c00});
    for (std::uint32_t i = 0; i < 16; ++i) {
      stream.put(static_cast<char>((block * 16U + i) & 0xffU));
    }
  }
  stream.close();

  if (variant == SyntheticVariant::truncated_payload) {
    const std::uintmax_t size = std::filesystem::file_size(path);
    std::filesystem::resize_file(path, size - 1);
  }
  return true;
}

struct TempFiles {
  std::vector<std::filesystem::path> paths;

  ~TempFiles() {
    std::error_code ignored;
    for (const auto & path : paths) {
      std::filesystem::remove(path, ignored);
    }
  }
};

std::filesystem::path make_temp_path(TempFiles & files, const std::string_view suffix) {
  const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::filesystem::path path = std::filesystem::temp_directory_path() /
    ("qwen35x-gguf-reader-" + std::to_string(nonce) + "-" + std::string(suffix) + ".gguf");
  files.paths.push_back(path);
  return path;
}

bool test_valid_file(TempFiles & files) {
  const std::filesystem::path path = make_temp_path(files, "valid");
  if (!expect(write_synthetic_gguf(path, SyntheticVariant::valid), "could not create synthetic GGUF")) {
    return false;
  }

  qwen35x::GgufReader reader;
  std::string error;
  bool ok = expect(reader.open(path.string(), error), error) &&
    expect(reader.is_open(), "reader must be open") &&
    expect(reader.version() == 3, "version mismatch") &&
    expect(reader.alignment() == 32, "alignment mismatch") &&
    expect(reader.metadata_count() == 15, "metadata count mismatch") &&
    expect(reader.tensors().size() == 3, "tensor count mismatch");

  const qwen35x::GgufTensorInfo * matrix = reader.find_tensor("matrix");
  ok = expect(matrix != nullptr, "matrix missing from index") && ok;
  if (matrix != nullptr) {
    ok = expect(matrix->is_q8_0(), "matrix type mismatch") &&
      expect(matrix->shape == std::vector<std::uint64_t>({2, 32}), "logical shape was not reversed") &&
      expect(matrix->ggml_shape == std::vector<std::uint64_t>({32, 2}), "serialized shape mismatch") &&
      expect(matrix->element_count == 64, "matrix element count mismatch") &&
      expect(matrix->relative_offset == 32, "matrix relative offset mismatch") &&
      expect(matrix->data_size == 68, "matrix byte size mismatch") && ok;
  }

  qwen35x::GgufTensorF32 norm;
  ok = expect(reader.read_f32_tensor("norm", norm, error), error) &&
    expect(norm.shape == std::vector<std::uint64_t>({4}), "F32 shape mismatch") &&
    expect(norm.data == std::vector<float>({1.0F, -2.5F, 0.0F, 4.25F}), "F32 payload mismatch") && ok;

  qwen35x::GgufTensorQ8_0 q8;
  ok = expect(reader.read_q8_0_tensor("matrix", q8, error), error) &&
    expect(q8.blocks.size() == 2, "Q8_0 block count mismatch") && ok;
  if (q8.blocks.size() == 2) {
    ok = expect(q8.blocks[0].d == 0x3c00, "Q8_0 scale mismatch") &&
      expect(q8.blocks[0].qs[0] == -16, "first Q8_0 quant mismatch") &&
      expect(q8.blocks[0].qs[31] == 15, "last first-block Q8_0 quant mismatch") &&
      expect(q8.blocks[1].qs[0] == 31, "first second-block Q8_0 quant mismatch") &&
      expect(q8.blocks[1].qs[31] == 0, "last Q8_0 quant mismatch") && ok;
  }

  const qwen35x::GgufTensorInfo * matrix_q4 = reader.find_tensor("matrix_q4");
  ok = expect(matrix_q4 != nullptr, "Q4_0 matrix missing from index") && ok;
  if (matrix_q4 != nullptr) {
    ok = expect(matrix_q4->is_q4_0(), "Q4_0 matrix type mismatch") &&
      expect(matrix_q4->shape == std::vector<std::uint64_t>({2, 32}), "Q4_0 shape mismatch") &&
      expect(matrix_q4->data_size == 36, "Q4_0 byte size mismatch") && ok;
  }
  qwen35x::GgufTensorQ4_0 q4;
  ok = expect(reader.read_q4_0_tensor("matrix_q4", q4, error), error) &&
    expect(q4.blocks.size() == 2, "Q4_0 block count mismatch") && ok;
  if (q4.blocks.size() == 2) {
    ok = expect(q4.blocks[0].d == 0x3c00, "Q4_0 scale mismatch") &&
      expect(q4.blocks[0].qs[0] == 0, "first Q4_0 byte mismatch") &&
      expect(q4.blocks[1].qs[15] == 31, "last Q4_0 byte mismatch") && ok;
  }

  std::vector<std::uint8_t> raw;
  ok = expect(reader.read_tensor_bytes("matrix", raw, error), error) &&
    expect(raw.size() == 68, "raw Q8_0 payload size mismatch") &&
    expect(!reader.read_f32_tensor("matrix", norm, error), "wrong typed read must fail") &&
    expect(!reader.read_q8_0_tensor("missing", q8, error), "missing tensor read must fail") && ok;
  return ok;
}

bool test_rejected_file(
  TempFiles & files,
  const SyntheticVariant variant,
  const std::string_view suffix,
  const std::string_view expected_error_fragment) {
  const std::filesystem::path path = make_temp_path(files, suffix);
  if (!expect(write_synthetic_gguf(path, variant), "could not create rejected synthetic GGUF")) {
    return false;
  }
  qwen35x::GgufReader reader;
  std::string error;
  return expect(!reader.open(path.string(), error), "invalid GGUF unexpectedly opened") &&
    expect(!reader.is_open(), "reader remained open after a parse error") &&
    expect(error.find(expected_error_fragment) != std::string::npos, "unexpected rejection reason");
}

} // namespace

bool test_k_types(TempFiles& files) {
  for (std::uint32_t type : {12U,13U,14U}) for (int mode=0;mode<3;++mode) {
    const auto path=make_temp_path(files,"k-quant");
    std::ofstream stream(path,std::ios::binary);
    stream.write("GGUF",4);write_le(stream,3U);
    write_le(stream,std::uint64_t{1});write_le(stream,std::uint64_t{0});
    write_string(stream,"k.weight");write_le(stream,2U);
    write_le(stream,std::uint64_t(mode==1?255:256));write_le(stream,std::uint64_t{1});
    write_le(stream,type);write_le(stream,std::uint64_t{0});pad_to(stream,32);
    const std::size_t bytes=type==12?144:type==13?176:210;
    for(std::size_t i=0;i<bytes-(mode==2);++i)stream.put(static_cast<char>(i));
    stream.close();
    qwen35x::GgufReader reader;std::string error;
    if(mode) {
      if(!expect(!reader.open(path.string(),error),"Malformed K tensor accepted"))return false;
    } else {
      if(!expect(reader.open(path.string(),error),error))return false;
      std::vector<std::uint8_t> data;
      if(!expect(reader.read_tensor_bytes("k.weight",data,error),error) ||
         !expect(data.size()==bytes,"K payload size mismatch"))return false;
      for(std::size_t i=0;i<bytes;++i)
        if(!expect(data[i]==static_cast<std::uint8_t>(i),"K payload changed"))return false;
    }
  }
  return true;
}

int main() {
  TempFiles files;
  bool ok = test_valid_file(files);
  ok = test_k_types(files) && ok;
  ok = test_rejected_file(
         files,
         SyntheticVariant::truncated_payload,
         "truncated",
         "extends beyond") && ok;
  ok = test_rejected_file(
         files,
         SyntheticVariant::invalid_q8_row,
         "invalid-row",
         "not divisible by 32") && ok;
  ok = test_rejected_file(
         files,
         SyntheticVariant::invalid_alignment,
         "invalid-alignment",
         "not a nonzero power of two") && ok;

  if (ok) {
    std::cout << "synthetic GGUF-v3 reader tests passed\n";
    return 0;
  }
  return 1;
}
