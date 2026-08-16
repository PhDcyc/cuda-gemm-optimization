# Optimization notes

## 1. Establish the contract first

All implementations compute row-major `C[M,N] = A[M,K] × B[K,N]`. The earlier examples assumed `M=N=K` and did not compare `C` with a reference, which allowed incomplete shared-memory loads to look like successful performance runs. The benchmark now generates one pedantic FP32 cuBLAS result and validates every selected kernel against it.

This ordering matters: an incorrect GEMM can appear faster simply because it moved or multiplied less data.

## 2. Naive kernel

One thread owns one output element and walks over K. Loads from B are coalesced across a warp, while each A value is redundantly requested by many threads. The kernel is useful as a correctness baseline but has little explicit data reuse.

## 3. Shared-memory tiling

The tiled kernel cooperatively loads 16×16 tiles of A and B. Each loaded value participates in up to 16 FMAs before the next tile is fetched. Zero-filling out-of-range elements keeps the implementation correct for arbitrary dimensions.

Two barriers are required per K tile:

1. after cooperative loads, before consumers read shared memory;
2. after computation, before producers overwrite the tile for the next iteration.

## 4. Register blocking

The register kernel maps 256 threads to a 128×128 output block. A logical 16×16 thread arrangement gives every thread an 8×8 output micro-tile, or 64 accumulators. Each inner-K step loads eight A and eight B values from shared memory and performs a 64-FMA outer product.

This raises arithmetic intensity at the cost of register pressure. `__launch_bounds__(256, 2)` communicates an intended lower bound of two resident blocks per SM to the compiler; actual occupancy still depends on architecture and compiler allocation, so it must be checked with `ptxas -v` or Nsight Compute.

## 5. Coalescing and shared layout

For a 128×128×8 block tile:

- every thread loads one aligned `float4` from A and one from B;
- A is transposed while entering shared memory so the eight A values used by a thread are contiguous;
- B remains K-major with contiguous N values;
- output uses two `float4` stores per micro-tile row.

The fast path is selected only when M and N are multiples of 128 and K is a multiple of 8. The edge-safe path uses guarded scalar transactions and zero fill, preserving the same arithmetic structure without out-of-bounds access.

## 6. Double buffering

Two shared-memory stages alternate. While one stage feeds FMAs, the next global tile is moved through per-thread load registers into the other stage. Because producers and consumers touch different stages, one block-wide barrier after the next-stage writes is sufficient before the stage switch.

This is a software pipeline at the shared-memory level. It does not use asynchronous `cp.async`; therefore overlap depends on compiler scheduling and available independent instructions.

## 7. Honest comparisons

The custom kernels use FP32 CUDA cores. cuBLAS can choose highly tuned architecture-specific implementations, and Tensor Core modes change both precision and attainable throughput. The benchmark sets `CUBLAS_PEDANTIC_MATH` so validation is an FP32 comparison rather than an accidental TF32 comparison.

A useful report should include:

- GPU model and compute capability;
- CUDA, driver and compiler versions;
- matrix shapes and data type;
- warm-up and repetition counts;
- kernel execution path;
- absolute/relative error;
- clock/power conditions when doing serious tuning.

Do not compare a single cold launch against a warmed library run, and do not include allocation or host-device copies in only one side of the comparison.

## 8. cp.async multi-stage pipeline

The register kernel prefetches the next global fragment into per-thread registers before consuming the current shared-memory stage. That overlap is software-scheduled and still occupies registers for the in-flight data. The `cp.async` kernel moves those global loads out of the register file: each tile is copied directly from global to shared memory with `__pipeline_memcpy_async`, and `__pipeline_wait_prior` gates consumption.

Four shared-memory stages are used. The prologue fills the first four K-tiles, then each loop iteration waits for stage `tile % 4`, computes the 8×8 outer products, and refills the same stage for the tile that is four ahead. The key differences from the register kernel:

- B uses one 16-byte `cp.async` per thread (row-major, no transposition).
- A is also row-major but padded to a 12-float row stride (8 + 4). The padding keeps every row start 16-byte aligned for `cp.async` and spreads the column-wise A reads across different banks so they stay conflict-free.

The measured effect is shape-dependent. At 8192³ the pipelined kernel reaches 41.3 TFLOP/s versus 41.2 TFLOP/s for the register kernel, a small but reproducible gain because the largest tiles amortize the pipeline. At 4096³ and below the two are within noise, which is honest: for this FP32 CUDA-core workload the register kernel is already close to the memory-bound limit, so `cp.async` mostly buys register-headroom and cleaner overlap rather than a step-change in throughput.

## 9. WMMA BF16 Tensor Core

The `wmma-bf16` kernel switches arithmetic from FP32 CUDA cores to 16×16×16 BF16 Tensor Core outer products (`nvcuda::wmma`). Because BF16 keeps only an 8-bit mantissa, the correctness comparison uses a wider tolerance: absolute 0.5 and relative 5e-2, versus 1e-2/1e-3 for the FP32 kernels.

Each block is 128×16 output, covered by eight warps stacked along M. The eight warps share one 16×16 B tile, so B is staged in shared memory once per K-group instead of being re-read from global memory by every warp (an 8× reduction in B traffic). A fragments stay per-warp because each warp owns a different 16-row band.

At 4096³ this path reaches 43.4 TFLOP/s (0.88× the FP32 cuBLAS baseline), which is a large fraction of cuBLAS even though the kernel still uses a single K-loop with no software pipelining. At 512³ it matches cuBLAS. The remaining gap to the ~165 TFLOP/s BF16 peak comes from the missing multi-stage pipeline and split-K; this kernel is a minimal, dependency-free Tensor Core reference rather than a tuned replacement.

## 10. Why Nsight Compute is absent

`ncu` is installed on the benchmark host, but the container blocks access to GPU performance counters with `ERR_NVGPUCTRPERM`. Rather than invent counter values, the repository records what is available without the profiler: `ptxas -v` register and shared-memory allocation per kernel, plus the full timing/error CSV. The `register` and `cp-async` kernels both use 127–128 registers and no spills; the WMMA kernel uses 42 registers and 512 B of shared memory. Occupancy and bank-conflict numbers should be added on a host that grants perf-counter access.
