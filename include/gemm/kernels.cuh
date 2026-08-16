#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <string_view>

namespace gemm {

enum class KernelKind {
    kNaive,
    kTiled,
    kRegisterTiled,
    kCpAsync,
    kWmmaBf16,
};

const char* kernel_name(KernelKind kind);
KernelKind parse_kernel(std::string_view name);

// Computes row-major C[M, N] = A[M, K] * B[K, N].
// The register-tiled and cp.async implementations select a vectorized fast path
// for aligned 128x128x8 tiles and an edge-safe path for all other positive
// shapes. The WMMA BF16 kernel is exposed separately below so the benchmark can
// convert inputs once and time only the Tensor Core GEMM itself.
void launch_gemm(KernelKind kind,
                 const float* a,
                 const float* b,
                 float* c,
                 int m,
                 int n,
                 int k,
                 cudaStream_t stream = nullptr);

// Converts a device FP32 matrix to BF16. The destination must hold `elements`
// entries.
void convert_fp32_to_bf16(const float* source,
                          __nv_bfloat16* destination,
                          std::size_t elements,
                          cudaStream_t stream = nullptr);

// WMMA BF16 Tensor Core GEMM. Requires M multiple of 128, N multiple of 16 and
// K multiple of 16. Inputs must already be BF16 on the device.
void launch_wmma_bf16(const __nv_bfloat16* a,
                      const __nv_bfloat16* b,
                      float* c,
                      int m,
                      int n,
                      int k,
                      cudaStream_t stream = nullptr);

bool uses_vectorized_fast_path(KernelKind kind, int m, int n, int k);

}  // namespace gemm
