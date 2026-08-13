#pragma once

#include <cuda_runtime_api.h>

#include <string_view>

namespace gemm {

enum class KernelKind {
    kNaive,
    kTiled,
    kRegisterTiled,
};

const char* kernel_name(KernelKind kind);
KernelKind parse_kernel(std::string_view name);

// Computes row-major C[M, N] = A[M, K] * B[K, N].
// The register-tiled implementation selects a vectorized fast path for aligned
// 128x128x8 tiles and an edge-safe path for all other positive shapes.
void launch_gemm(KernelKind kind,
                 const float* a,
                 const float* b,
                 float* c,
                 int m,
                 int n,
                 int k,
                 cudaStream_t stream = nullptr);

bool uses_vectorized_fast_path(KernelKind kind, int m, int n, int k);

}  // namespace gemm
