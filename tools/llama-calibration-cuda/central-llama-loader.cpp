#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
struct ConfigureLlamaDllSearch {
  ConfigureLlamaDllSearch() {
    if (!SetDllDirectoryA(CENTRAL_LLAMA_BIN)) {
      std::fputs("Unable to configure the shared llama.cpp DLL directory.\n", stderr);
      std::abort();
    }
    // Load exact central paths before the delay loader can see older DLLs next
    // to an adapter executable. Dependencies are resolved from the same bin.
    for (const char * name : {"ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll", "llama.dll"}) {
      const std::string path = std::string(CENTRAL_LLAMA_BIN) + "/" + name;
      if (!LoadLibraryExA(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)) {
        std::fprintf(stderr, "Unable to load shared llama.cpp library %s (error %lu).\n",
                     path.c_str(), GetLastError());
        std::abort();
      }
    }
  }
};
ConfigureLlamaDllSearch configure_llama_dll_search;
}
