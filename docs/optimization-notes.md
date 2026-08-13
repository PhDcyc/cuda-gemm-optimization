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
