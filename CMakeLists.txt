cmake_minimum_required(VERSION 3.22)

project(cuda_gemm_optimization VERSION 1.0.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(gemm_kernels STATIC src/kernels.cu)
target_include_directories(gemm_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(gemm_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
set_target_properties(gemm_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

add_executable(gemm_benchmark src/benchmark.cu)
target_include_directories(gemm_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gemm_benchmark PRIVATE
  gemm_kernels
  CUDA::cudart
  CUDA::cublas
)
target_compile_options(gemm_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

include(CTest)
option(GEMM_ENABLE_GPU_TESTS "Enable tests that require an NVIDIA GPU" OFF)
if(BUILD_TESTING AND GEMM_ENABLE_GPU_TESTS)
  add_test(
    NAME correctness_edge_shapes
    COMMAND gemm_benchmark --m 257 --n 263 --k 269 --kernel naive,tiled,register --warmup 1 --iterations 1
  )
  add_test(
    NAME correctness_fast_path
    COMMAND gemm_benchmark --m 512 --n 512 --k 512 --kernel register --warmup 1 --iterations 2
  )
  add_test(
    NAME correctness_cp_async
    COMMAND gemm_benchmark --m 1024 --n 1024 --k 1024 --kernel cp-async --warmup 1 --iterations 2
  )
  add_test(
    NAME correctness_wmma_bf16
    COMMAND gemm_benchmark --m 1024 --n 1024 --k 1024 --kernel wmma-bf16 --warmup 1 --iterations 2
  )
endif()
