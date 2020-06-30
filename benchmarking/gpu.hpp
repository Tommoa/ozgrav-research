#pragma once

#include <chrono>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <vector>

#ifdef DEBUG
#ifndef __syncthreads
#include <assert.h>
#define __syncthreads()                                                        \
    { assert(false); }
#endif
#endif

namespace gpu {
/// A custom allocator class to allocate memory on the GPU
template <class T> class Allocator {
  public:
    typedef T value_type;

    Allocator() {}

    template <class U> Allocator(const Allocator<U> &) noexcept {}

    [[nodiscard]] value_type *allocate(std::size_t n) {
        value_type *result = nullptr;

        auto e = cudaMallocManaged(&result, n * sizeof(T), cudaMemAttachGlobal);

        if (e != cudaSuccess) {
            throw thrust::system_error(
                e, thrust::cuda_category(),
                "GpuAllocator::allocate(): cudaMallocManaged");
        }

        return result;
    }

    void deallocate(value_type *ptr, std::size_t) {
        cudaError_t error = cudaFree(ptr);

        if (error != cudaSuccess) {
            throw thrust::system_error(
                error, thrust::cuda_category(),
                "ManagedAllocator::deallocate(): cudaFree");
        }
    }
};

template <class T1, class T2>
bool operator==(const Allocator<T1> &, const Allocator<T2> &) {
    return true;
}

template <class T1, class T2>
bool operator!=(const Allocator<T1> &lhs, const Allocator<T2> &rhs) {
    return false;
}

/// A vector on the GPU
template <class T> using Vector = std::vector<T, Allocator<T>>;

void check_memory() {
    size_t free_bytes, total_bytes;
    long double free_megabytes, total_megabytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    free_megabytes = ((long double)free_bytes) / (1024 * 1024);
    total_megabytes = ((long double)total_bytes) / (1024 * 1024);

    std::cerr << "Free (MB): " << free_megabytes << '\n';
    std::cerr << "Total (MB): " << total_megabytes << '\n';
    std::cerr << "Used (MB): " << total_megabytes - free_megabytes << '\n';
}

void get_info(int index = 0) {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, index);
    std::cerr << "Device: " << properties.name << std::endl;
}

void gpu_assert(cudaError_t code, const char *file, int line,
                bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "GPU failed assertion: " << cudaGetErrorString(code)
                  << '\n';
        std::cerr << "\tFile: " << std::string(file) << '\n';
        std::cerr << "\tLine: " << line << '\n';
        if (abort) {
            exit(code);
        }
    }
}

template <typename F>
double benchmark(int iterations, const std::string mode, F &&function) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        function();
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << mode << ": ";
    return diff.count();
}

#define GPUASSERT(code)                                                        \
    { gpu::gpu_assert((code), __FILE__, __LINE__); }

} // namespace gpu
