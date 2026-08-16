# CUDA GEMM Optimization

[![CUDA build](https://github.com/PhDcyc/cuda-gemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/PhDcyc/cuda-gemm-optimization/actions/workflows/ci.yml)
![CUDA](https://img.shields.io/badge/CUDA-SM80%20%7C%20SM86%20%7C%20SM89-76B900?logo=nvidia&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A correctness-first CUDA GEMM laboratory that shows how a row-major FP32 matrix multiplication evolves from one-output-per-thread code into a vectorized, register-tiled kernel, then onto `cp.async` multi-stage pipelining and a WMMA BF16 Tensor Core path. Every custom kernel is checked against a pedantic FP32 cuBLAS reference before its performance number is accepted.

> 中文简介：这是一个面向 AI Infra 学习与作品集展示的 CUDA GEMM 项目。重点不是给出脱离环境的“峰值数字”，而是把正确性、优化路径、benchmark 方法和硬件相关结果分开记录。

## What changed

The original repository contained three standalone square-matrix examples. The current version fixes incomplete tile loads and hard-coded dimensions, then builds one reproducible benchmark around six comparable implementations:

| Kernel | Main idea | Shape support |
|---|---|---|
| `naive` | One thread computes one output | Arbitrary positive M/N/K |
| `tiled` | 16×16 cooperative shared-memory tiles | Arbitrary positive M/N/K |
| `register` | 128×128×8 block tile, 8×8/thread, double-buffered shared memory | Vectorized aligned fast path plus edge-safe fallback |
| `cp-async` | 128×128×8 with a 4-stage `cp.async` pipeline (SM80+) | Aligned M/N=128k, K=8k |
| `wmma-bf16` | 16×16×16 BF16 Tensor Core outer products with a shared B tile | M=128k, N=16k, K=16k |
| `cublas` | Pedantic FP32 library reference and performance baseline | Arbitrary positive M/N/K |

The benchmark reports **GFLOP/s**, not “FLOPs,” and never treats an unvalidated kernel result as meaningful performance evidence.

## Evidence status

| Evidence | Status |
|---|---|
| SM80/SM86/SM89 compilation | GitHub CI verified |
| Arbitrary-shape validation path | Verified on RTX 4090 against pedantic FP32 cuBLAS |
| RTX 4090 runtime sweep | Measured on a real RTX 4090 (SM 89, see below) |
| Nsight Compute counters | Blocked by container perf-counter permission (`ERR_NVGPUCTRPERM`); `ptxas -v` register/smem data recorded instead |

### RTX 4090 measurements

Checked-in results: [`results/rtx4090-4096.csv`](results/rtx4090-4096.csv), [`results/rtx4090-8192.csv`](results/rtx4090-8192.csv) and [`results/rtx4090-correctness.csv`](results/rtx4090-correctness.csv), produced by `gemm_benchmark` on an NVIDIA GeForce RTX 4090 (compute capability 8.9, driver 580.82.07).

4096×4096×4096 sweep (warmup=10, iterations=50):

| Kernel | Path | Time | GFLOP/s | vs cuBLAS |
|---|---:|---:|---:|---:|
| naive | edge-safe | 25.056 ms | 5,485 | 0.111× |
| tiled | edge-safe | 28.158 ms | 4,881 | 0.099× |
| register | vectorized | 3.298 ms | 41,677 | 0.846× |
| cp-async | vectorized | 3.328 ms | 41,301 | 0.838× |
| wmma-bf16 | tensor-core | 3.169 ms | 43,377 | 0.880× |
| cuBLAS | library | 2.789 ms | 49,283 | 1.000× |

8192×8192×8192 sweep (warmup=5, iterations=20):

| Kernel | Path | Time | GFLOP/s | vs cuBLAS |
|---|---:|---:|---:|---:|
| register | vectorized | 26.699 ms | 41,182 | 0.788× |
| cp-async | vectorized | 26.623 ms | 41,299 | 0.790× |
| wmma-bf16 | tensor-core | 29.379 ms | 37,425 | 0.716× |
| cuBLAS | library | 21.042 ms | 52,252 | 1.000× |

Arbitrary-shape correctness sweep (257×263×269, edge-safe paths): all four FP32 kernels matched the pedantic FP32 cuBLAS reference (max abs err ≤ 1.5e-05, zero mismatches, `passed=true`). The `wmma-bf16` path is validated against a BF16-aware tolerance (abs 0.5, rel 5e-2) because its 8-bit mantissa introduces a rounding error that accumulates over a 4096-long reduction.

Compile-time resource use from `ptxas -v` for SM89:

| Kernel | Registers/thread | Shared memory | Spills |
|---|---:|---:|---:|
| naive | 40 | 0 B | 0 |
| tiled | 36 | 2,048 B | 0 |
| register | 127 | 16,384 B | 0 |
| cp-async | 128 | 40,960 B | 0 |
| wmma-bf16 | 42 | 512 B | 0 |

No throughput number is committed until the executing GPU, shape, path, timing configuration and numerical error are recorded together.

## Register-tiled data flow

```mermaid
flowchart LR
    GM[Global A and B] -->|coalesced float4 loads| LR[Load registers]
    LR -->|transpose A tile| SM[Double-buffered shared memory]
    SM -->|8 A + 8 B values/thread| RR[Thread registers]
    RR -->|8x8 outer products| ACC[64 FP32 accumulators]
    ACC -->|vectorized float4 stores| GC[Global C]
```

For aligned shapes, a 256-thread block computes a 128×128 output tile. Each thread accumulates an 8×8 micro-tile, increasing data reuse while keeping global transactions coalesced. Shapes not divisible by 128×128×8 use the same register blocking with guarded scalar loads/stores.

## Requirements

- NVIDIA GPU with compute capability 8.0 or newer recommended
- CUDA Toolkit 11.8 or newer
- CMake 3.22+
- C++17 compiler

The default build emits SM80, SM86 and SM89 code. Override `CMAKE_CUDA_ARCHITECTURES` for another GPU.

For an RTX 4090, the checked-in preset keeps the build reproducible and compiles only SM89 code:

```bash
cmake --preset release-sm89
cmake --build --preset release-sm89
```

## Build

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

For an RTX 4090, SM89 is the relevant target.

## Correctness run

Exercise all kernels on an awkward shape so the edge paths are covered:

```bash
./build/gemm_benchmark \
  --m 257 --n 263 --k 269 \
  --kernel all \
  --warmup 2 --iterations 5
```

The process exits with status `2` if any output exceeds:

```text
abs(actual - reference) <= atol + rtol * abs(reference)
```

Defaults are `atol=1e-2` and `rtol=1e-3`; both can be changed on the command line.

## Performance run

```bash
./build/gemm_benchmark \
  --m 8192 --n 8192 --k 8192 \
  --kernel tiled,register,cublas \
  --warmup 10 --iterations 50 \
  --csv results/rtx4090-8192.csv
```

Example output format (numbers intentionally omitted because they must come from the executing GPU):

```text
kernel      path           time (ms)       GFLOP/s   vs cuBLAS   max abs err     valid
register    vectorized          ...             ...        ...            ...       yes
cublas      library             ...             ...     1.000x       0.00e+00       yes
```

Run the checked-in shape sweep:

```bash
python3 tools/run_sweep.py \
  --binary build/gemm_benchmark \
  --output results/rtx4090.csv
```

The sweep includes square GEMMs and LLM-like rectangular projection shapes.

## Benchmark methodology

- Inputs are deterministic uniform FP32 values with seed `2026`.
- cuBLAS uses `CUBLAS_PEDANTIC_MATH` as the correctness reference.
- Warm-up launches are completed before CUDA-event timing starts.
- Reported latency is the mean of repeated asynchronous launches synchronized by the stop event.
- FLOP count is `2 × M × N × K`; reported throughput is decimal GFLOP/s.
- GPU name, compute capability, exact shape, execution path and validation errors are written to CSV.
- H2D allocation/copies and host validation copies are outside the timed region.

See [Optimization notes](docs/optimization-notes.md) for the reasoning and limitations behind each kernel.

## Project layout

```text
include/gemm/          Public launch API and CUDA/cuBLAS error handling
src/kernels.cu         Naive, tiled, register-tiled, cp.async and WMMA BF16 kernels
src/benchmark.cu       CLI, cuBLAS reference, timing, validation and CSV output
tools/run_sweep.py     Repeatable multi-shape benchmark driver
docs/                  Optimization and measurement notes
.github/workflows/     CUDA compile CI using an official NVIDIA development image
```

## CI and GPU tests

GitHub CI compiles SM80/86/89 code but does not claim runtime performance because hosted runners do not provide an NVIDIA GPU. On a GPU machine, enable CTest correctness cases with:

```bash
cmake -S . -B build -DGEMM_ENABLE_GPU_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Scope and next steps

This repository studies FP32 CUDA-core SGEMM alongside a BF16 Tensor Core path. It is not intended to beat Tensor Core cuBLAS on modern accelerators. Completed extensions:

- ✅ BF16 Tensor Core kernel via WMMA (shared B tile, no CUTLASS dependency)
- ✅ `cp.async` 4-stage pipeline for SM80+
- ✅ RTX 4090 runtime sweep at 4096³ and 8192³

Remaining natural extensions:

- Split-K and grouped GEMM for inference workloads
- CUTLASS/CuTe-based Tensor Core kernels with multi-stage software pipelines
- Nsight Compute reports (occupancy, bank conflicts, memory transactions) on a host with perf-counter access

## License

MIT
