#include "gemm/cuda_check.cuh"
#include "gemm/kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Options {
    int m = 1024;
    int n = 1024;
    int k = 1024;
    int warmup = 5;
    int iterations = 20;
    unsigned int seed = 2026;
    float absolute_tolerance = 1.0e-2F;
    float relative_tolerance = 1.0e-3F;
    bool validate = true;
    std::vector<std::string> kernels = {"register", "cublas"};
    std::string csv_path;
};

struct ValidationResult {
    float max_absolute_error = 0.0F;
    float max_relative_error = 0.0F;
    std::size_t mismatches = 0;
    bool passed = true;
};

struct BenchmarkResult {
    std::string kernel;
    std::string path;
    float milliseconds = 0.0F;
    double gflops = 0.0;
    double relative_to_cublas = std::numeric_limits<double>::quiet_NaN();
    ValidationResult validation;
};

void print_help(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n\n"
        << "Benchmarks row-major FP32 C[M,N] = A[M,K] * B[K,N].\n\n"
        << "Options:\n"
        << "  --m INT                 Rows of A and C (default: 1024)\n"
        << "  --n INT                 Columns of B and C (default: 1024)\n"
        << "  --k INT                 Reduction dimension (default: 1024)\n"
        << "  --kernel LIST           Comma-separated naive,tiled,register,cublas,all\n"
        << "                          (default: register,cublas)\n"
        << "  --warmup INT            Warm-up launches (default: 5)\n"
        << "  --iterations INT        Timed launches (default: 20)\n"
        << "  --seed INT              Input RNG seed (default: 2026)\n"
        << "  --atol FLOAT            Absolute validation tolerance (default: 1e-2)\n"
        << "  --rtol FLOAT            Relative validation tolerance (default: 1e-3)\n"
        << "  --no-validate           Skip cuBLAS reference comparison\n"
        << "  --csv PATH              Write machine-readable results; use - for stdout\n"
        << "  --help                  Show this help\n";
}

int parse_positive_int(const std::string& value, const char* flag) {
    std::size_t consumed = 0;
    const long parsed = std::stol(value, &consumed);
    if (consumed != value.size() || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(flag) + " must be a positive integer");
    }
    return static_cast<int>(parsed);
}

std::vector<std::string> split_kernels(const std::string& value) {
    if (value == "all") {
        return {"naive", "tiled", "register", "cublas"};
    }
    std::vector<std::string> kernels;
    std::stringstream stream(value);
    for (std::string kernel; std::getline(stream, kernel, ',');) {
        if (kernel != "naive" && kernel != "tiled" && kernel != "register" &&
            kernel != "cublas") {
            throw std::invalid_argument("unknown kernel: " + kernel);
        }
        if (std::find(kernels.begin(), kernels.end(), kernel) == kernels.end()) {
            kernels.push_back(std::move(kernel));
        }
    }
    if (kernels.empty()) {
        throw std::invalid_argument("--kernel requires at least one kernel");
    }
    return kernels;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        auto next_value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument(flag + " requires a value");
            }
            return argv[index];
        };

        if (flag == "--help" || flag == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else if (flag == "--m") {
            options.m = parse_positive_int(next_value(), "--m");
        } else if (flag == "--n") {
            options.n = parse_positive_int(next_value(), "--n");
        } else if (flag == "--k") {
            options.k = parse_positive_int(next_value(), "--k");
        } else if (flag == "--warmup") {
            options.warmup = parse_positive_int(next_value(), "--warmup");
        } else if (flag == "--iterations") {
            options.iterations = parse_positive_int(next_value(), "--iterations");
        } else if (flag == "--seed") {
            options.seed = static_cast<unsigned int>(
                parse_positive_int(next_value(), "--seed"));
        } else if (flag == "--atol") {
            options.absolute_tolerance = std::stof(next_value());
        } else if (flag == "--rtol") {
            options.relative_tolerance = std::stof(next_value());
        } else if (flag == "--kernel") {
            options.kernels = split_kernels(next_value());
        } else if (flag == "--csv") {
            options.csv_path = next_value();
        } else if (flag == "--no-validate") {
            options.validate = false;
        } else {
            throw std::invalid_argument("unknown option: " + flag);
        }
    }
    if (options.absolute_tolerance < 0.0F || options.relative_tolerance < 0.0F) {
        throw std::invalid_argument("validation tolerances must be non-negative");
    }
    return options;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer_), elements * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return pointer_; }
    const T* get() const { return pointer_; }
    std::size_t bytes() const { return elements_ * sizeof(T); }

private:
    T* pointer_ = nullptr;
    std::size_t elements_ = 0;
};

class CublasHandle {
public:
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH));
    }

    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    cublasHandle_t get() { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

void launch_cublas(cublasHandle_t handle,
                   const float* a,
                   const float* b,
                   float* c,
                   int m,
                   int n,
                   int k) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    // cuBLAS is column-major. Swapping A/B and M/N computes the equivalent
    // row-major product without transposing the stored inputs.
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             n,
                             m,
                             k,
                             &alpha,
                             b,
                             n,
                             a,
                             k,
                             &beta,
                             c,
                             n));
}

template <typename Launch>
float time_launch(Launch&& launch, int warmup, int iterations) {
    for (int index = 0; index < warmup; ++index) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int index = 0; index < iterations; ++index) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsed = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed / static_cast<float>(iterations);
}

ValidationResult validate(const std::vector<float>& actual,
                          const std::vector<float>& reference,
                          float absolute_tolerance,
                          float relative_tolerance) {
    ValidationResult result;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float absolute_error = std::abs(actual[index] - reference[index]);
        const float denominator = std::max(std::abs(reference[index]), 1.0e-7F);
        const float relative_error = absolute_error / denominator;
        result.max_absolute_error = std::max(result.max_absolute_error, absolute_error);
        result.max_relative_error = std::max(result.max_relative_error, relative_error);
        if (absolute_error > absolute_tolerance + relative_tolerance * std::abs(reference[index])) {
            ++result.mismatches;
        }
    }
    result.passed = result.mismatches == 0;
    return result;
}

double calculate_gflops(int m, int n, int k, float milliseconds) {
    const double operations = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                              static_cast<double>(k);
    return operations / (static_cast<double>(milliseconds) * 1.0e6);
}

void write_csv(std::ostream& output,
               const Options& options,
               const cudaDeviceProp& device,
               const std::vector<BenchmarkResult>& results) {
    output << "gpu,compute_capability,m,n,k,kernel,path,time_ms,gflops,relative_to_cublas,"
              "max_abs_error,max_rel_error,mismatches,passed\n";
    for (const auto& result : results) {
        output << '"' << device.name << "\"," << device.major << '.' << device.minor << ','
               << options.m << ',' << options.n << ',' << options.k << ',' << result.kernel << ','
               << result.path << ',' << std::fixed << std::setprecision(6) << result.milliseconds
               << ',' << result.gflops << ',';
        if (std::isnan(result.relative_to_cublas)) {
            output << "";
        } else {
            output << result.relative_to_cublas;
        }
        output << ',' << result.validation.max_absolute_error << ','
               << result.validation.max_relative_error << ',' << result.validation.mismatches << ','
               << (result.validation.passed ? "true" : "false") << '\n';
    }
}

void print_results(const Options& options,
                   const cudaDeviceProp& device,
                   const std::vector<BenchmarkResult>& results) {
    std::cout << "GPU: " << device.name << " (SM " << device.major << device.minor << ")\n"
              << "Shape: M=" << options.m << ", N=" << options.n << ", K=" << options.k
              << " | warmup=" << options.warmup << " | iterations=" << options.iterations << "\n\n"
              << std::left << std::setw(12) << "kernel" << std::setw(12) << "path"
              << std::right << std::setw(12) << "time (ms)" << std::setw(14) << "GFLOP/s"
              << std::setw(12) << "vs cuBLAS" << std::setw(14) << "max abs err"
              << std::setw(10) << "valid" << '\n';
    std::cout << std::string(86, '-') << '\n';
    for (const auto& result : results) {
        std::cout << std::left << std::setw(12) << result.kernel << std::setw(12) << result.path
                  << std::right << std::fixed << std::setprecision(3) << std::setw(12)
                  << result.milliseconds << std::setw(14) << result.gflops;
        if (std::isnan(result.relative_to_cublas)) {
            std::cout << std::setw(12) << "-";
        } else {
            std::ostringstream ratio;
            ratio << std::fixed << std::setprecision(3) << result.relative_to_cublas << 'x';
            std::cout << std::setw(12) << ratio.str();
        }
        std::cout << std::scientific << std::setprecision(2) << std::setw(14)
                  << result.validation.max_absolute_error << std::setw(10)
                  << (options.validate ? (result.validation.passed ? "yes" : "NO") : "skipped")
                  << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        int device_index = 0;
        CUDA_CHECK(cudaGetDevice(&device_index));
        cudaDeviceProp device{};
        CUDA_CHECK(cudaGetDeviceProperties(&device, device_index));

        const std::size_t a_elements = static_cast<std::size_t>(options.m) * options.k;
        const std::size_t b_elements = static_cast<std::size_t>(options.k) * options.n;
        const std::size_t c_elements = static_cast<std::size_t>(options.m) * options.n;
        const std::size_t required_bytes =
            (a_elements + b_elements + 2 * c_elements) * sizeof(float);
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (required_bytes > free_bytes) {
            std::ostringstream message;
            message << "benchmark requires " << required_bytes / (1024.0 * 1024.0)
                    << " MiB, but only " << free_bytes / (1024.0 * 1024.0)
                    << " MiB is currently free";
            throw std::runtime_error(message.str());
        }

        std::mt19937 random_engine(options.seed);
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
        std::vector<float> host_a(a_elements);
        std::vector<float> host_b(b_elements);
        std::generate(host_a.begin(), host_a.end(), [&]() { return distribution(random_engine); });
        std::generate(host_b.begin(), host_b.end(), [&]() { return distribution(random_engine); });

        DeviceBuffer<float> device_a(a_elements);
        DeviceBuffer<float> device_b(b_elements);
        DeviceBuffer<float> device_c(c_elements);
        DeviceBuffer<float> device_reference(c_elements);
        CUDA_CHECK(cudaMemcpy(device_a.get(), host_a.data(), device_a.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_b.get(), host_b.data(), device_b.bytes(), cudaMemcpyHostToDevice));

        CublasHandle cublas;
        launch_cublas(cublas.get(),
                      device_a.get(),
                      device_b.get(),
                      device_reference.get(),
                      options.m,
                      options.n,
                      options.k);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> host_reference;
        if (options.validate) {
            host_reference.resize(c_elements);
            CUDA_CHECK(cudaMemcpy(host_reference.data(),
                                  device_reference.get(),
                                  device_reference.bytes(),
                                  cudaMemcpyDeviceToHost));
        }

        std::vector<BenchmarkResult> results;
        for (const std::string& kernel_name : options.kernels) {
            BenchmarkResult result;
            result.kernel = kernel_name;
            if (kernel_name == "cublas") {
                result.path = "library";
                result.milliseconds = time_launch(
                    [&]() {
                        launch_cublas(cublas.get(),
                                      device_a.get(),
                                      device_b.get(),
                                      device_c.get(),
                                      options.m,
                                      options.n,
                                      options.k);
                    },
                    options.warmup,
                    options.iterations);
            } else {
                const gemm::KernelKind kind = gemm::parse_kernel(kernel_name);
                result.path = gemm::uses_vectorized_fast_path(
                                  kind, options.m, options.n, options.k)
                                  ? "vectorized"
                                  : "edge-safe";
                result.milliseconds = time_launch(
                    [&]() {
                        gemm::launch_gemm(kind,
                                          device_a.get(),
                                          device_b.get(),
                                          device_c.get(),
                                          options.m,
                                          options.n,
                                          options.k);
                    },
                    options.warmup,
                    options.iterations);
            }
            result.gflops = calculate_gflops(options.m, options.n, options.k, result.milliseconds);

            if (options.validate) {
                std::vector<float> host_actual(c_elements);
                CUDA_CHECK(cudaMemcpy(host_actual.data(),
                                      device_c.get(),
                                      device_c.bytes(),
                                      cudaMemcpyDeviceToHost));
                result.validation = validate(host_actual,
                                             host_reference,
                                             options.absolute_tolerance,
                                             options.relative_tolerance);
            }
            results.push_back(std::move(result));
        }

        const auto cublas_result = std::find_if(results.begin(), results.end(), [](const auto& result) {
            return result.kernel == "cublas";
        });
        if (cublas_result != results.end()) {
            for (auto& result : results) {
                result.relative_to_cublas = result.gflops / cublas_result->gflops;
            }
        }

        if (options.csv_path != "-") {
            print_results(options, device, results);
        }
        if (!options.csv_path.empty()) {
            if (options.csv_path == "-") {
                write_csv(std::cout, options, device, results);
            } else {
                std::ofstream csv(options.csv_path);
                if (!csv) {
                    throw std::runtime_error("could not open CSV output: " + options.csv_path);
                }
                write_csv(csv, options, device, results);
                std::cout << "\nWrote " << options.csv_path << '\n';
            }
        }

        const bool all_valid = std::all_of(results.begin(), results.end(), [](const auto& result) {
            return result.validation.passed;
        });
        return all_valid ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
