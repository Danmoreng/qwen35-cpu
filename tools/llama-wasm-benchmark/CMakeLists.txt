cmake_minimum_required(VERSION 3.20)
project(llama_wasm_benchmark LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT EMSCRIPTEN OR NOT LLAMA_SOURCE OR NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "Use Emscripten and supply LLAMA_SOURCE and JSON_INCLUDE_DIR")
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_WASM_MEM64 OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_HTML OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_WEBGPU OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
add_compile_options(-pthread -msimd128)
add_subdirectory("${LLAMA_SOURCE}" llama)
add_executable(llama_browser main.cpp)
target_include_directories(llama_browser PRIVATE "${JSON_INCLUDE_DIR}")
target_compile_options(llama_browser PRIVATE -fexceptions)
target_link_libraries(llama_browser PRIVATE llama)
set_target_properties(llama_browser PROPERTIES SUFFIX ".js")
target_link_options(llama_browser PRIVATE -pthread -msimd128 -fexceptions
  -sMODULARIZE=1 -sEXPORT_NAME=createLlamaModule -sENVIRONMENT=web,worker
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=67108864 -sMAXIMUM_MEMORY=4294967296
  -sSTACK_SIZE=2097152 -sPTHREAD_POOL_SIZE=7 -sPTHREAD_POOL_SIZE_STRICT=2
  -sFORCE_FILESYSTEM=1 -lworkerfs.js
  "-sINCOMING_MODULE_JS_API=['locateFile','printErr','mainScriptUrlOrBlob']"
  "-sEXPORTED_FUNCTIONS=['_bench_load','_bench_score','_bench_unload']"
  "-sEXPORTED_RUNTIME_METHODS=['ccall','FS','WORKERFS']")
