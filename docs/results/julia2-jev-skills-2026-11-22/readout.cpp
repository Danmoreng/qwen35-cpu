// Standalone quality evaluator for llama.cpp 9b56ccf. Input: row ID and Julia token IDs.
// Output: row ID followed by all 255 raw Julia head logits, tab separated.
#include "ggml-backend.h"
#include "llama.h"
#include "llama-ext.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char **argv) try {
    constexpr int width = 1024, classes = 255, context = 16384;
    check(argc == 3, "Expected text GGUF and FP32 readout file");
    ggml_backend_load_all();
    llama_backend_init();
    auto gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    check(gpu && std::string(ggml_backend_dev_name(gpu)) == "CUDA0", "CUDA0 required");
    ggml_backend_dev_t devices[] = {gpu, nullptr};
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 999;
    model_params.devices = devices;
    auto model = llama_model_load_from_file(argv[1], model_params);
    check(model && llama_model_n_embd(model) == width, "Model/width mismatch");

    auto params = llama_context_default_params();
    params.n_ctx = context;
    params.n_batch = context;
    params.n_ubatch = 512;
    params.n_seq_max = 1;
    params.n_threads = 4;
    params.n_threads_batch = 4;
    params.embeddings = false;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
    params.type_k = GGML_TYPE_F32;
    params.type_v = GGML_TYPE_F32;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto ctx = llama_init_from_model(model, params);
    check(ctx, "llama_init_from_model");
    // In this revision Qwen3.5 exports the post-RMSNorm hidden state as h_nextn.
    llama_set_embeddings_nextn(ctx, true, true);

    std::vector<float> head((width + 1) * classes), logits(classes);
    std::ifstream file(argv[2], std::ios::binary);
    check(bool(file.read(reinterpret_cast<char *>(head.data()), head.size() * sizeof(float)))
          && file.peek() == EOF, "Invalid readout file");
    constexpr size_t matrix_bytes = width * classes * sizeof(float);
    constexpr size_t bias_bytes = classes * sizeof(float);
    float *weights, *hidden, *scores;
    check(cudaMalloc(&weights, matrix_bytes) == cudaSuccess, "Weight allocation");
    check(cudaMalloc(&hidden, width * sizeof(float)) == cudaSuccess, "Hidden allocation");
    check(cudaMalloc(&scores, bias_bytes) == cudaSuccess, "Logit allocation");
    check(cudaMemcpy(weights, head.data(), matrix_bytes, cudaMemcpyHostToDevice) == cudaSuccess,
          "Weight upload");
    cublasHandle_t handle;
    check(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cuBLAS init");
    check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH) == CUBLAS_STATUS_SUCCESS,
          "FP32 readout arithmetic");

    auto batch = llama_batch_init(context, 0, 1);
    const int vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float one = 1.0f;
    size_t count = 0;
    std::cout << std::setprecision(9);
    for (std::string line; std::getline(std::cin, line);) {
        std::istringstream row(line);
        std::string id;
        row >> id;
        batch.n_tokens = 0;
        for (int token; row >> token;) {
            int index = batch.n_tokens++;
            check(index < context && token >= 0 && token < vocab_size, "Invalid tokens");
            batch.token[index] = token;
            batch.pos[index] = index;
            batch.n_seq_id[index] = 1;
            batch.seq_id[index][0] = 0;
            batch.logits[index] = 0;
        }
        check(row.eof() && !id.empty() && batch.n_tokens > 0, "Invalid row");
        batch.logits[batch.n_tokens - 1] = 1;
        llama_memory_clear(llama_get_memory(ctx), true);
        check(llama_decode(ctx, batch) == 0, "llama_decode");
        auto embedding = llama_get_embeddings_nextn_ith(ctx, -1);
        check(embedding, "Final normalized hidden state unavailable");
        check(cudaMemcpy(hidden, embedding, width * sizeof(float), cudaMemcpyHostToDevice)
              == cudaSuccess, "Hidden state upload");
        check(cudaMemcpy(scores, head.data() + width * classes, bias_bytes,
                         cudaMemcpyHostToDevice) == cudaSuccess, "Bias upload");
        // The row-major head is a width x classes column-major matrix in cuBLAS.
        check(cublasSgemv(handle, CUBLAS_OP_T, width, classes, &one, weights, width,
                          hidden, 1, &one, scores, 1) == CUBLAS_STATUS_SUCCESS, "FP32 readout");
        check(cudaMemcpy(logits.data(), scores, bias_bytes, cudaMemcpyDeviceToHost)
              == cudaSuccess, "Logit download");
        std::cout << id;
        for (float value : logits) {
            check(std::isfinite(value), "Non-finite logit");
            std::cout << '\t' << value;
        }
        std::cout << '\n';
        if (++count % 32 == 0) std::cout.flush();
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    cublasDestroy(handle);
    cudaFree(weights);
    cudaFree(hidden);
    cudaFree(scores);
    llama_backend_free();
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
