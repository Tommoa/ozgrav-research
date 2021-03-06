#pragma once

#include <cassert>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>
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
template <class T> class ManagedAllocator {
  public:
    typedef T value_type;

    ManagedAllocator() {}

    template <class U> ManagedAllocator(const ManagedAllocator<U> &) noexcept {}

    [[nodiscard]] value_type *allocate(std::size_t n) {
        value_type *result = nullptr;

        auto e = cudaMallocManaged(&result, n * sizeof(T), cudaMemAttachGlobal);

        if (e != cudaSuccess) {
            throw thrust::system_error(
                e, thrust::cuda_category(),
                "ManagedAllocator::allocate(): cudaMallocManaged");
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
bool operator==(const ManagedAllocator<T1> &, const ManagedAllocator<T2> &) {
    return true;
}

template <class T1, class T2>
bool operator!=(const ManagedAllocator<T1> &lhs,
                const ManagedAllocator<T2> &rhs) {
    return false;
}

/// A vector on the GPU
template <class T> using ManagedVector = std::vector<T, ManagedAllocator<T>>;

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
    std::cerr << "Device " << index << ": " << properties.name << std::endl;
    std::cerr << "\tClock speed: " << ((float)properties.clockRate / 1000)
              << "MHz" << std::endl;
    std::cerr << "\tCompute capability: " << properties.major << "."
              << properties.minor << std::endl;
    std::cerr << "\tMax threads per block: " << properties.maxThreadsPerBlock
              << std::endl;
    std::cerr << "\tCompute units: " << properties.multiProcessorCount
              << std::endl;
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
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; ++i) {
        function();
    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    std::cout << std::setw(20) << mode << ":\t";
    return time / 1000;
}

#define GPUASSERT(code)                                                        \
    { gpu::gpu_assert((code), __FILE__, __LINE__); }

} // namespace gpu
