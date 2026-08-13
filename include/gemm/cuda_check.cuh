#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace gemm {

inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << file << ':' << line << " CUDA call failed: " << expression << " ("
            << cudaGetErrorString(status) << ')';
    throw std::runtime_error(message.str());
}

inline void check_cublas(cublasStatus_t status, const char* expression, const char* file, int line) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }
    std::ostringstream message;
    message << file << ':' << line << " cuBLAS call failed: " << expression
            << " (status=" << static_cast<int>(status) << ')';
    throw std::runtime_error(message.str());
}

}  // namespace gemm

#define CUDA_CHECK(expression) ::gemm::check_cuda((expression), #expression, __FILE__, __LINE__)
#define CUBLAS_CHECK(expression) ::gemm::check_cublas((expression), #expression, __FILE__, __LINE__)
