#include "gemm/kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <stdexcept>
#include <string>

namespace gemm {
namespace {

constexpr int kTile = 16;

__global__ void naive_sgemm(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c,
                            int m,
                            int n,
                            int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }

    float accumulator = 0.0F;
    for (int inner = 0; inner < k; ++inner) {
        accumulator = fmaf(a[row * k + inner], b[inner * n + col], accumulator);
    }
    c[row * n + col] = accumulator;
}

__global__ void tiled_sgemm(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c,
                            int m,
                            int n,
                            int k) {
    __shared__ float a_tile[kTile][kTile];
    __shared__ float b_tile[kTile][kTile];

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    float accumulator = 0.0F;

    for (int tile_k = 0; tile_k < k; tile_k += kTile) {
        const int a_col = tile_k + threadIdx.x;
        const int b_row = tile_k + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : 0.0F;
        b_tile[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[b_row * n + col] : 0.0F;
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < kTile; ++inner) {
            accumulator = fmaf(a_tile[threadIdx.y][inner],
                               b_tile[inner][threadIdx.x],
                               accumulator);
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = accumulator;
    }
}

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 8;
constexpr int kThreadM = 8;
constexpr int kThreadN = 8;
constexpr int kThreads = 256;

struct LoadFragment {
    float a[4];
    float b[4];
};

template <bool kAligned>
__device__ __forceinline__ LoadFragment load_fragment(const float* __restrict__ a,
                                                       const float* __restrict__ b,
                                                       int tile_k,
                                                       int block_row,
                                                       int block_col,
                                                       int m,
                                                       int n,
                                                       int k) {
    const int tid = threadIdx.x;
    const int a_row = tid >> 1;
    const int a_col = (tid & 1) << 2;
    const int b_row = tid >> 5;
    const int b_col = (tid & 31) << 2;
    LoadFragment fragment{};

    if constexpr (kAligned) {
        const float4 a_values = *reinterpret_cast<const float4*>(
            a + (block_row + a_row) * k + tile_k + a_col);
        const float4 b_values = *reinterpret_cast<const float4*>(
            b + (tile_k + b_row) * n + block_col + b_col);
        fragment.a[0] = a_values.x;
        fragment.a[1] = a_values.y;
        fragment.a[2] = a_values.z;
        fragment.a[3] = a_values.w;
        fragment.b[0] = b_values.x;
        fragment.b[1] = b_values.y;
        fragment.b[2] = b_values.z;
        fragment.b[3] = b_values.w;
    } else {
#pragma unroll
        for (int vector_element = 0; vector_element < 4; ++vector_element) {
            const int global_a_row = block_row + a_row;
            const int global_a_col = tile_k + a_col + vector_element;
            fragment.a[vector_element] =
                (global_a_row < m && global_a_col < k)
                    ? a[global_a_row * k + global_a_col]
                    : 0.0F;

            const int global_b_row = tile_k + b_row;
            const int global_b_col = block_col + b_col + vector_element;
            fragment.b[vector_element] =
                (global_b_row < k && global_b_col < n)
                    ? b[global_b_row * n + global_b_col]
                    : 0.0F;
        }
    }
    return fragment;
}

__device__ __forceinline__ void store_fragment(
    const LoadFragment& fragment,
    float (&a_shared)[2][kBlockK][kBlockM],
    float (&b_shared)[2][kBlockK][kBlockN],
    int stage) {
    const int tid = threadIdx.x;
    const int a_row = tid >> 1;
    const int a_col = (tid & 1) << 2;
    const int b_row = tid >> 5;
    const int b_col = (tid & 31) << 2;
#pragma unroll
    for (int vector_element = 0; vector_element < 4; ++vector_element) {
        a_shared[stage][a_col + vector_element][a_row] = fragment.a[vector_element];
        b_shared[stage][b_row][b_col + vector_element] = fragment.b[vector_element];
    }
}

template <bool kAligned>
__global__ __launch_bounds__(kThreads, 2) void register_tiled_sgemm(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k) {
    __shared__ __align__(16) float a_shared[2][kBlockK][kBlockM];
    __shared__ __align__(16) float b_shared[2][kBlockK][kBlockN];

    const int tid = threadIdx.x;
    const int thread_row = (tid >> 4) * kThreadM;
    const int thread_col = (tid & 15) * kThreadN;
    const int block_row = blockIdx.y * kBlockM;
    const int block_col = blockIdx.x * kBlockN;

    float accumulators[kThreadM][kThreadN] = {0.0F};
    float a_registers[kThreadM];
    float b_registers[kThreadN];

    const LoadFragment first_fragment =
        load_fragment<kAligned>(a, b, 0, block_row, block_col, m, n, k);
    store_fragment(first_fragment, a_shared, b_shared, 0);
    __syncthreads();

    const int tile_count = (k + kBlockK - 1) / kBlockK;
    int read_stage = 0;
    for (int tile = 0; tile < tile_count; ++tile) {
        LoadFragment next_fragment{};
        if (tile + 1 < tile_count) {
            // Prefetch the next global fragment before consuming the current
            // shared-memory stage. The fragment stays in registers while the
            // outer products execute.
            next_fragment = load_fragment<kAligned>(a,
                                                    b,
                                                    (tile + 1) * kBlockK,
                                                    block_row,
                                                    block_col,
                                                    m,
                                                    n,
                                                    k);
        }
#pragma unroll
        for (int inner = 0; inner < kBlockK; ++inner) {
            const float4 a_low = *reinterpret_cast<const float4*>(
                &a_shared[read_stage][inner][thread_row]);
            const float4 a_high = *reinterpret_cast<const float4*>(
                &a_shared[read_stage][inner][thread_row + 4]);
            const float4 b_low = *reinterpret_cast<const float4*>(
                &b_shared[read_stage][inner][thread_col]);
            const float4 b_high = *reinterpret_cast<const float4*>(
                &b_shared[read_stage][inner][thread_col + 4]);

            a_registers[0] = a_low.x;
            a_registers[1] = a_low.y;
            a_registers[2] = a_low.z;
            a_registers[3] = a_low.w;
            a_registers[4] = a_high.x;
            a_registers[5] = a_high.y;
            a_registers[6] = a_high.z;
            a_registers[7] = a_high.w;
            b_registers[0] = b_low.x;
            b_registers[1] = b_low.y;
            b_registers[2] = b_low.z;
            b_registers[3] = b_low.w;
            b_registers[4] = b_high.x;
            b_registers[5] = b_high.y;
            b_registers[6] = b_high.z;
            b_registers[7] = b_high.w;

#pragma unroll
            for (int row = 0; row < kThreadM; ++row) {
#pragma unroll
                for (int col = 0; col < kThreadN; ++col) {
                    accumulators[row][col] =
                        fmaf(a_registers[row], b_registers[col], accumulators[row][col]);
                }
            }
        }

        if (tile + 1 < tile_count) {
            const int write_stage = read_stage ^ 1;
            store_fragment(next_fragment, a_shared, b_shared, write_stage);
            __syncthreads();
            read_stage = write_stage;
        }
    }

    const int output_row = block_row + thread_row;
    const int output_col = block_col + thread_col;
    if constexpr (kAligned) {
#pragma unroll
        for (int row = 0; row < kThreadM; ++row) {
            *reinterpret_cast<float4*>(c + (output_row + row) * n + output_col) =
                make_float4(accumulators[row][0],
                            accumulators[row][1],
                            accumulators[row][2],
                            accumulators[row][3]);
            *reinterpret_cast<float4*>(c + (output_row + row) * n + output_col + 4) =
                make_float4(accumulators[row][4],
                            accumulators[row][5],
                            accumulators[row][6],
                            accumulators[row][7]);
        }
    } else {
#pragma unroll
        for (int row = 0; row < kThreadM; ++row) {
#pragma unroll
            for (int col = 0; col < kThreadN; ++col) {
                if (output_row + row < m && output_col + col < n) {
                    c[(output_row + row) * n + output_col + col] = accumulators[row][col];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// cp.async multi-stage pipeline (SM80+)
//
// Reuses the 128x128x8 register-tiled data flow but replaces the synchronous
// global -> register -> shared-memory path with 16-byte cp.async.  A fixed
// number of shared-memory stages are filled asynchronously ahead of the FMA
// loop so that global loads of tile T+1 overlap the outer products of tile T.
// A is stored row-major with a 4-column pad so the row stride stays a multiple
// of 16 bytes (cp.async alignment) while column-wise reads stay bank-conflict
// free; B is row-major with no padding.  Only the aligned fast path is exposed;
// unaligned shapes fall back to the guarded register kernel.
// ---------------------------------------------------------------------------
constexpr int kPipelineStages = 4;
constexpr int kCpAsyncPad = 4;  // row stride = 8 + 4 = 12 floats = 48 bytes

__global__ __launch_bounds__(kThreads, 2) void cp_async_sgemm(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k) {
    __shared__ __align__(16) float a_shared[kPipelineStages][kBlockM][kBlockK + kCpAsyncPad];
    __shared__ __align__(16) float b_shared[kPipelineStages][kBlockK][kBlockN];

    const int tid = threadIdx.x;
    const int thread_row = (tid >> 4) * kThreadM;
    const int thread_col = (tid & 15) * kThreadN;
    const int block_row = blockIdx.y * kBlockM;
    const int block_col = blockIdx.x * kBlockN;

    const int a_row = tid >> 1;
    const int a_col = (tid & 1) << 2;
    const int b_row = tid >> 5;
    const int b_col = (tid & 31) << 2;

    float accumulators[kThreadM][kThreadN] = {0.0F};
    float a_registers[kThreadM];
    float b_registers[kThreadN];

    const int tile_count = k / kBlockK;

    auto issue_async_loads = [&](int tile_k, int stage) {
        // A row-major 128x8: each thread moves one 16-byte half-row.
        __pipeline_memcpy_async(&a_shared[stage][a_row][a_col],
                                &a[(block_row + a_row) * k + tile_k + a_col],
                                sizeof(float4));
        // B row-major 8x128: each thread moves one 16-byte column slice.
        __pipeline_memcpy_async(&b_shared[stage][b_row][b_col],
                                &b[(tile_k + b_row) * n + block_col + b_col],
                                sizeof(float4));
        __pipeline_commit();
    };

    // Prologue: asynchronously fill the first `kPipelineStages` stages.
    for (int stage = 0; stage < kPipelineStages; ++stage) {
        if (stage < tile_count) {
            issue_async_loads(stage * kBlockK, stage);
        }
    }

    for (int tile = 0; tile < tile_count; ++tile) {
        const int stage = tile % kPipelineStages;
        __pipeline_wait_prior(kPipelineStages - 1);
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < kBlockK; ++inner) {
#pragma unroll
            for (int row = 0; row < kThreadM; ++row) {
                a_registers[row] = a_shared[stage][thread_row + row][inner];
            }
            const float4 b_low = *reinterpret_cast<const float4*>(
                &b_shared[stage][inner][thread_col]);
            const float4 b_high = *reinterpret_cast<const float4*>(
                &b_shared[stage][inner][thread_col + 4]);
            b_registers[0] = b_low.x;
            b_registers[1] = b_low.y;
            b_registers[2] = b_low.z;
            b_registers[3] = b_low.w;
            b_registers[4] = b_high.x;
            b_registers[5] = b_high.y;
            b_registers[6] = b_high.z;
            b_registers[7] = b_high.w;

#pragma unroll
            for (int row = 0; row < kThreadM; ++row) {
#pragma unroll
                for (int col = 0; col < kThreadN; ++col) {
                    accumulators[row][col] =
                        fmaf(a_registers[row], b_registers[col], accumulators[row][col]);
                }
            }
        }
        __syncthreads();

        // Refill this stage for the tile that arrives `kPipelineStages` ahead.
        const int next_tile = tile + kPipelineStages;
        if (next_tile < tile_count) {
            issue_async_loads(next_tile * kBlockK, stage);
        }
    }

    const int output_row = block_row + thread_row;
    const int output_col = block_col + thread_col;
#pragma unroll
    for (int row = 0; row < kThreadM; ++row) {
        *reinterpret_cast<float4*>(c + (output_row + row) * n + output_col) =
            make_float4(accumulators[row][0],
                        accumulators[row][1],
                        accumulators[row][2],
                        accumulators[row][3]);
        *reinterpret_cast<float4*>(c + (output_row + row) * n + output_col + 4) =
            make_float4(accumulators[row][4],
                        accumulators[row][5],
                        accumulators[row][6],
                        accumulators[row][7]);
    }
}

// ---------------------------------------------------------------------------
// WMMA BF16 Tensor Core kernel (SM80+)
//
// 16x16x16 BF16 outer-product accumulation on Tensor Cores.  Each warp owns a
// single 16x16 output tile and reduces the full K dimension before writing the
// FP32 accumulator back to C.  A 128x16 output block is covered by eight warps
// stacked along M so every warp's A/B fragments are fully independent; the
// shared B tile could be promoted to shared memory later, but this keeps the
// first Tensor Core path simple and auditable.  Inputs are converted from FP32
// to BF16 on the device first; the caller owns those conversions.
// ---------------------------------------------------------------------------
using WmmaFragA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                                         16,
                                         16,
                                         16,
                                         __nv_bfloat16,
                                         nvcuda::wmma::row_major>;
using WmmaFragB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                                         16,
                                         16,
                                         16,
                                         __nv_bfloat16,
                                         nvcuda::wmma::row_major>;
using WmmaFragC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>;

constexpr int kWmmaBlockM = 128;
constexpr int kWmmaBlockN = 16;
constexpr int kWmmaTile = 16;
constexpr int kWmmaWarpsPerBlock = kWmmaBlockM / kWmmaTile;  // 8
constexpr int kWmmaThreads = kWmmaWarpsPerBlock * 32;        // 256

__global__ __launch_bounds__(kWmmaThreads) void wmma_bf16_sgemm(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k) {
    // The eight warps share the same B tile, so it is staged in shared memory
    // once per K-group instead of being re-read from global memory by every
    // warp.  A fragments stay in per-warp registers because each warp owns a
    // different 16-row band.
    __shared__ __nv_bfloat16 b_tile[kWmmaTile][kWmmaTile];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int tile_row = blockIdx.y * kWmmaBlockM + warp_id * kWmmaTile;
    const int tile_col = blockIdx.x * kWmmaBlockN;

    WmmaFragC accumulator;
    nvcuda::wmma::fill_fragment(accumulator, 0.0F);

    for (int tile_k = 0; tile_k < k; tile_k += kWmmaTile) {
        // Cooperative load of the shared 16x16 B tile (256 elements, one per
        // thread).
        const int load_index = tid;
        b_tile[load_index / kWmmaTile][load_index % kWmmaTile] =
            b[(tile_k + load_index / kWmmaTile) * n + tile_col + load_index % kWmmaTile];
        __syncthreads();

        WmmaFragA fragment_a;
        WmmaFragB fragment_b;
        nvcuda::wmma::load_matrix_sync(
            fragment_a, a + tile_row * k + tile_k, k);
        nvcuda::wmma::load_matrix_sync(
            fragment_b, &b_tile[0][0], kWmmaTile);
        nvcuda::wmma::mma_sync(accumulator, fragment_a, fragment_b, accumulator);
        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(
        c + tile_row * n + tile_col, accumulator, n, nvcuda::wmma::mem_row_major);
}

// Convert FP32 row-major matrices to BF16 on the device.  This keeps the
// conversion cost explicit and reusable by the benchmark.
__global__ void convert_fp32_to_bf16_kernel(const float* __restrict__ src,
                                            __nv_bfloat16* __restrict__ dst,
                                            std::size_t elements) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        dst[index] = __float2bfloat16(src[index]);
    }
}

}  // namespace

const char* kernel_name(KernelKind kind) {
    switch (kind) {
        case KernelKind::kNaive:
            return "naive";
        case KernelKind::kTiled:
            return "tiled";
        case KernelKind::kRegisterTiled:
            return "register";
        case KernelKind::kCpAsync:
            return "cp-async";
        case KernelKind::kWmmaBf16:
            return "wmma-bf16";
    }
    return "unknown";
}

KernelKind parse_kernel(std::string_view name) {
    if (name == "naive") {
        return KernelKind::kNaive;
    }
    if (name == "tiled") {
        return KernelKind::kTiled;
    }
    if (name == "register") {
        return KernelKind::kRegisterTiled;
    }
    if (name == "cp-async" || name == "cpasync") {
        return KernelKind::kCpAsync;
    }
    if (name == "wmma-bf16" || name == "wmma") {
        return KernelKind::kWmmaBf16;
    }
    throw std::invalid_argument("unknown CUDA kernel: " + std::string(name));
}

bool uses_vectorized_fast_path(KernelKind kind, int m, int n, int k) {
    if (kind == KernelKind::kCpAsync) {
        return m % kBlockM == 0 && n % kBlockN == 0 && k % kBlockK == 0;
    }
    if (kind == KernelKind::kWmmaBf16) {
        return m % kWmmaBlockM == 0 && n % kWmmaBlockN == 0 && k % kWmmaTile == 0;
    }
    return kind == KernelKind::kRegisterTiled && m % kBlockM == 0 && n % kBlockN == 0 &&
           k % kBlockK == 0;
}

void launch_gemm(KernelKind kind,
                 const float* a,
                 const float* b,
                 float* c,
                 int m,
                 int n,
                 int k,
                 cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        throw std::invalid_argument("M, N, and K must be positive");
    }

    switch (kind) {
        case KernelKind::kNaive: {
            const dim3 block(kTile, kTile);
            const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
            naive_sgemm<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
            return;
        }
        case KernelKind::kTiled: {
            const dim3 block(kTile, kTile);
            const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
            tiled_sgemm<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
            return;
        }
        case KernelKind::kRegisterTiled: {
            const dim3 block(kThreads);
            const dim3 grid((n + kBlockN - 1) / kBlockN,
                            (m + kBlockM - 1) / kBlockM);
            if (uses_vectorized_fast_path(kind, m, n, k)) {
                register_tiled_sgemm<true><<<grid, block, 0, stream>>>(a, b, c, m, n, k);
            } else {
                register_tiled_sgemm<false><<<grid, block, 0, stream>>>(a, b, c, m, n, k);
            }
            return;
        }
        case KernelKind::kCpAsync: {
            if (!uses_vectorized_fast_path(kind, m, n, k)) {
                throw std::invalid_argument(
                    "cp-async requires M/N multiples of 128 and K a multiple of 8");
            }
            const dim3 block(kThreads);
            const dim3 grid((n + kBlockN - 1) / kBlockN,
                            (m + kBlockM - 1) / kBlockM);
            cp_async_sgemm<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
            return;
        }
        case KernelKind::kWmmaBf16: {
            throw std::invalid_argument(
                "wmma-bf16 uses a dedicated entry point (launch_wmma_bf16)");
        }
    }
}

void convert_fp32_to_bf16(const float* source,
                          __nv_bfloat16* destination,
                          std::size_t elements,
                          cudaStream_t stream) {
    constexpr int kConvertThreads = 256;
    const dim3 block(kConvertThreads);
    const dim3 grid((elements + kConvertThreads - 1) / kConvertThreads);
    convert_fp32_to_bf16_kernel<<<grid, block, 0, stream>>>(source, destination, elements);
}

void launch_wmma_bf16(const __nv_bfloat16* a,
                      const __nv_bfloat16* b,
                      float* c,
                      int m,
                      int n,
                      int k,
                      cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        throw std::invalid_argument("M, N, and K must be positive");
    }
    if (m % kWmmaBlockM != 0 || n % kWmmaBlockN != 0 || k % kWmmaTile != 0) {
        throw std::invalid_argument(
            "wmma-bf16 requires M multiple of 128, N multiple of 16, "
            "and K a multiple of 16");
    }
    const dim3 block(kWmmaThreads);
    const dim3 grid((n + kWmmaBlockN - 1) / kWmmaBlockN,
                    (m + kWmmaBlockM - 1) / kWmmaBlockM);
    wmma_bf16_sgemm<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}

}  // namespace gemm
