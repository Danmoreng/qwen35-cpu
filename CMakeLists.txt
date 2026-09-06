cmake_minimum_required(VERSION 3.24)
project(qwen35_cpu VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)
include(CheckCXXCompilerFlag)
add_library(qwen35x_core
  src/cpu/activation.cpp
  src/cpu/delta_net.cpp
  src/cpu/executor.cpp
  src/cpu/full_attention.cpp
  src/cpu/full_attention_tiled.cpp
  src/cpu/q4_dot4.cpp
  src/cpu/q4_0.cpp
  src/cpu/q4_h128.cpp
  src/cpu/q8_0.cpp
  src/compiler/profile_loader.cpp
  src/runtime/cpu_runtime.cpp
  src/tokenizer/tokenizer.cpp
  src/weights/q4_h128_artifact.cpp
  src/weights/safetensors.cpp
)
set(QWEN35X_Q8_0_HAS_AVX2_TU OFF)
set(QWEN35X_Q8_0_HAS_AVX_VNNI_TU OFF)
set(QWEN35X_Q8_0_HAS_AVX512_TU OFF)
set(QWEN35X_Q8_0_HAS_AVX512_VNNI_TU OFF)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" QWEN35X_SYSTEM_PROCESSOR_LOWER)
# Keep qwen35x_core on the portable compiler baseline. Only the isolated
# translation unit below may contain AVX2/FMA/F16C instructions.
if (QWEN35X_SYSTEM_PROCESSOR_LOWER MATCHES "^(x86_64|amd64|i[3-6]86)$" OR
    CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
  if (MSVC)
    set(QWEN35X_Q8_0_HAS_AVX2_TU ON)
    set_source_files_properties(
      src/cpu/activation_avx2.cpp src/cpu/q4_0_avx2.cpp src/cpu/q4_dot4_avx2.cpp src/cpu/q4_h128_avx2.cpp src/cpu/q8_0_avx2.cpp src/cpu/delta_net_avx2.cpp src/cpu/full_attention_avx2.cpp src/cpu/full_attention_tiled_avx2.cpp
      PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    # The AVX-VNNI intrinsic spellings used by the isolated backend are
    # available in the VS 2022 toolset. Older MSVCs keep the AVX2 backend.
    if (MSVC_VERSION GREATER_EQUAL 1930)
      set(QWEN35X_Q8_0_HAS_AVX_VNNI_TU ON)
      set(QWEN35X_Q8_0_HAS_AVX512_TU ON)
      set(QWEN35X_Q8_0_HAS_AVX512_VNNI_TU ON)
      set_source_files_properties(
        src/cpu/q4_0_vnni.cpp src/cpu/q4_dot4_vex.cpp src/cpu/q8_0_vnni.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
      set_source_files_properties(
        src/cpu/activation_avx512.cpp src/cpu/q4_h128_avx512.cpp src/cpu/full_attention_tiled_avx512.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
      set_source_files_properties(
        src/cpu/q4_0_avx512_vnni.cpp src/cpu/q4_0_evex_vnni.cpp src/cpu/q4_dot4_evex.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    endif ()
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    set(QWEN35X_Q8_0_HAS_AVX2_TU ON)
    set_source_files_properties(
      src/cpu/activation_avx2.cpp src/cpu/q4_0_avx2.cpp src/cpu/q4_dot4_avx2.cpp src/cpu/q4_h128_avx2.cpp src/cpu/q8_0_avx2.cpp src/cpu/delta_net_avx2.cpp src/cpu/full_attention_avx2.cpp src/cpu/full_attention_tiled_avx2.cpp
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-mtune=skylake")
    check_cxx_compiler_flag("-mavxvnni" QWEN35X_COMPILER_HAS_MAVXVNNI)
    if (QWEN35X_COMPILER_HAS_MAVXVNNI)
      set(QWEN35X_Q8_0_HAS_AVX_VNNI_TU ON)
      set_source_files_properties(
        src/cpu/q4_0_vnni.cpp src/cpu/q4_dot4_vex.cpp src/cpu/q8_0_vnni.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mavxvnni;-mfma;-mf16c")
    endif ()
    check_cxx_compiler_flag("-mavx512f" QWEN35X_COMPILER_HAS_MAVX512F)
    if (QWEN35X_COMPILER_HAS_MAVX512F)
      set(QWEN35X_Q8_0_HAS_AVX512_TU ON)
      set_source_files_properties(
        src/cpu/activation_avx512.cpp src/cpu/q4_h128_avx512.cpp src/cpu/full_attention_tiled_avx512.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx2;-mf16c;-mfma")
      check_cxx_compiler_flag("-mavx512vnni" QWEN35X_COMPILER_HAS_MAVX512VNNI)
      check_cxx_compiler_flag("-mavx512vl" QWEN35X_COMPILER_HAS_MAVX512VL)
      if (QWEN35X_COMPILER_HAS_MAVX512VNNI AND QWEN35X_COMPILER_HAS_MAVX512VL)
        set(QWEN35X_Q8_0_HAS_AVX512_VNNI_TU ON)
        set_source_files_properties(
          src/cpu/q4_0_avx512_vnni.cpp src/cpu/q4_0_evex_vnni.cpp src/cpu/q4_dot4_evex.cpp
          PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512vl;-mavx512vnni;-mfma;-mf16c")
      endif ()
    endif ()
  endif ()
endif ()

if (QWEN35X_Q8_0_HAS_AVX2_TU)
  target_sources(
    qwen35x_core PRIVATE
    src/cpu/activation_avx2.cpp src/cpu/q4_0_avx2.cpp src/cpu/q4_dot4_avx2.cpp src/cpu/q4_h128_avx2.cpp src/cpu/q8_0_avx2.cpp src/cpu/delta_net_avx2.cpp src/cpu/full_attention_avx2.cpp src/cpu/full_attention_tiled_avx2.cpp)
  target_compile_definitions(qwen35x_core PRIVATE QWEN35X_Q8_0_HAS_AVX2_TU=1)
else ()
  target_compile_definitions(qwen35x_core PRIVATE QWEN35X_Q8_0_HAS_AVX2_TU=0)
endif ()

if (QWEN35X_Q8_0_HAS_AVX_VNNI_TU)
  target_sources(qwen35x_core PRIVATE src/cpu/q4_0_vnni.cpp src/cpu/q4_dot4_vex.cpp src/cpu/q8_0_vnni.cpp)
  target_compile_definitions(qwen35x_core PRIVATE QWEN35X_Q8_0_HAS_AVX_VNNI_TU=1)
else ()
  target_compile_definitions(qwen35x_core PRIVATE QWEN35X_Q8_0_HAS_AVX_VNNI_TU=0)
endif ()

if (QWEN35X_Q8_0_HAS_AVX512_TU)
  target_sources(qwen35x_core PRIVATE src/cpu/activation_avx512.cpp src/cpu/q4_h128_avx512.cpp src/cpu/full_attention_tiled_avx512.cpp)
  target_compile_definitions(qwen35x_core PRIVATE QWEN35X_Q8_0_HAS_AVX512_TU=1)
else ()
  target_compile_definitions(qwen35x_core PRIVATE QWEN35X_Q8_0_HAS_AVX512_TU=0)
endif ()

if (QWEN35X_Q8_0_HAS_AVX512_VNNI_TU)
  target_sources(qwen35x_core PRIVATE src/cpu/q4_0_avx512_vnni.cpp src/cpu/q4_0_evex_vnni.cpp src/cpu/q4_dot4_evex.cpp)
  target_compile_definitions(qwen35x_core PRIVATE QWEN35X_Q8_0_HAS_AVX512_VNNI_TU=1)
else ()
  target_compile_definitions(qwen35x_core PRIVATE QWEN35X_Q8_0_HAS_AVX512_VNNI_TU=0)
endif ()

target_include_directories(qwen35x_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(qwen35x_core PRIVATE Threads::Threads)


add_executable(qwen35_cpu src/main.cpp)
target_link_libraries(qwen35_cpu PRIVATE qwen35x_core)
option(QWEN35_BUILD_SERVER "Build the native HTTP server" ON)
if (QWEN35_BUILD_SERVER)
  include(FetchContent)
  FetchContent_Declare(http_headers
    URL https://codeload.github.com/yhirose/cpp-httplib/tar.gz/refs/tags/v0.54.1
    URL_HASH SHA256=7310f5312e1423830d649b38ed028e9db86303a979ccbfdbd1c4b1574f422dfb
    SOURCE_SUBDIR unused)
  FetchContent_Declare(json_headers
    URL https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v3.12.0
    URL_HASH SHA256=4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187
    SOURCE_SUBDIR unused)
  FetchContent_MakeAvailable(http_headers json_headers)
  add_executable(qwen35_cpu_server src/server.cpp)
  target_include_directories(qwen35_cpu_server PRIVATE
    ${http_headers_SOURCE_DIR} ${json_headers_SOURCE_DIR}/single_include)
  target_link_libraries(qwen35_cpu_server PRIVATE qwen35x_core Threads::Threads)
  if (WIN32)
    target_link_libraries(qwen35_cpu_server PRIVATE ws2_32 crypt32)
  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(qwen35_cpu_server PRIVATE -static-libgcc -static-libstdc++)
  endif ()
  install(TARGETS qwen35_cpu_server RUNTIME DESTINATION .)
  install(FILES ${http_headers_SOURCE_DIR}/LICENSE DESTINATION licenses RENAME cpp-httplib.txt)
  install(FILES ${json_headers_SOURCE_DIR}/LICENSE.MIT DESTINATION licenses RENAME nlohmann-json.txt)
endif ()
install(TARGETS qwen35_cpu RUNTIME DESTINATION .)
install(FILES README.md LICENSE PROVENANCE.md DESTINATION .)
install(DIRECTORY docs/ DESTINATION docs)
install(PROGRAMS scripts/download-model.sh DESTINATION .)
install(FILES scripts/download-model.ps1 DESTINATION .)
add_executable(qwen35_cpu_bench src/cpu_engine_bench_main.cpp)
target_link_libraries(qwen35_cpu_bench PRIVATE qwen35x_core)
add_executable(qwen35_cpu_pack src/q4_h128_convert_main.cpp)
target_link_libraries(qwen35_cpu_pack PRIVATE qwen35x_core)
enable_testing()
add_executable(full_attention_tiled_test tests/full_attention_tiled_test.cpp)
target_link_libraries(full_attention_tiled_test PRIVATE qwen35x_core)
add_test(NAME full_attention_tiled COMMAND full_attention_tiled_test)
add_executable(q4_dot4_test tests/q4_dot4_test.cpp)
target_link_libraries(q4_dot4_test PRIVATE qwen35x_core)
add_test(NAME q4_dot4 COMMAND q4_dot4_test)
add_executable(q4_0_test tests/q4_0_test.cpp)
target_link_libraries(q4_0_test PRIVATE qwen35x_core)
add_test(NAME q4_0 COMMAND q4_0_test)
add_executable(q4_h128_test tests/q4_h128_test.cpp)
target_link_libraries(q4_h128_test PRIVATE qwen35x_core)
add_test(NAME q4_h128 COMMAND q4_h128_test)
add_executable(q8_0_test tests/q8_0_test.cpp)
target_link_libraries(q8_0_test PRIVATE qwen35x_core)
add_test(NAME q8_0 COMMAND q8_0_test)
add_executable(q4_h128_artifact_test tests/q4_h128_artifact_test.cpp)
target_link_libraries(q4_h128_artifact_test PRIVATE qwen35x_core)
add_test(NAME q4_h128_artifact COMMAND q4_h128_artifact_test)
add_executable(delta_net_test tests/delta_net_test.cpp)
target_link_libraries(delta_net_test PRIVATE qwen35x_core)
add_test(NAME delta_net COMMAND delta_net_test)
add_executable(kv_pages_test tests/kv_pages_test.cpp)
target_link_libraries(kv_pages_test PRIVATE qwen35x_core)
add_test(NAME kv_pages COMMAND kv_pages_test)
add_executable(cpu_prefix_model_test tests/cpu_prefix_model_test.cpp)
target_link_libraries(cpu_prefix_model_test PRIVATE qwen35x_core)
add_executable(cpu_scheduler_model_test tests/cpu_scheduler_model_test.cpp)
target_link_libraries(cpu_scheduler_model_test PRIVATE qwen35x_core)
