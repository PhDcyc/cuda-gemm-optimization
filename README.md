# CUDA GEMM Optimization

A correctness-first CUDA SGEMM laboratory that shows how a row-major FP32 matrix multiplication evolves from one-output-per-thread code into a vectorized, register-tiled kernel. Every custom kernel is checked against a pedantic FP32 cuBLAS reference before its performance number is accepted.

> 中文简介：这是一个面向 AI Infra 学习与作品集展示的 CUDA GEMM 项目。重点不是给出脱离环境的“峰值数字”，而是把正确性、优化路径、benchmark 方法和硬件相关结果分开记录。

## What changed

The original repository contained three standalone square-matrix examples. The current version fixes incomplete tile loads and hard-coded dimensions, then builds one reproducible benchmark around four comparable implementations:

| Kernel | Main idea | Shape support |
|---|---|---|
| `naive` | One thread computes one output | Arbitrary positive M/N/K |
| `tiled` | 16×16 cooperative shared-memory tiles | Arbitrary positive M/N/K |
| `register` | 128×128×8 block tile, 8×8/thread, double-buffered shared memory | Vectorized aligned fast path plus edge-safe fallback |
| `cublas` | Pedantic FP32 library reference and performance baseline | Arbitrary positive M/N/K |

The benchmark reports **GFLOP/s**, not “FLOPs,” and never treats an unvalidated kernel result as meaningful performance evidence.

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
- CUDA Toolkit 12.x
- CMake 3.22+
- C++17 compiler

The default build emits SM80, SM86 and SM89 code. Override `CMAKE_CUDA_ARCHITECTURES` for another GPU.

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
src/kernels.cu         Naive, tiled and register-tiled CUDA kernels
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

This repository studies FP32 CUDA-core SGEMM. It is not intended to beat Tensor Core cuBLAS on modern accelerators. Natural extensions are:

- TF32/BF16 Tensor Core kernels via WMMA or CUTLASS/CuTe
- `cp.async` multi-stage pipelines for SM80+
- Split-K and grouped GEMM for inference workloads
- Nsight Compute reports for occupancy, bank conflicts and memory transactions

## License

MIT
