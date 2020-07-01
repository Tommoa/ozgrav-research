#include "gpu.hpp"

#include <__clang_cuda_runtime_wrapper.h>
#include <algorithm>
#include <assert.h>
#include <cpuid.h>
#include <cuda_profiler_api.h>
#include <exception>
#include <iostream>
#include <math.h>
#include <random>

void get_cpus() {
    char cpu_model[0x40];
    unsigned int cpu_info[4] = {0, 0, 0, 0};

    __cpuid(0x80000000, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
    unsigned int num_cpus = cpu_info[0];

    memset(cpu_model, 0, sizeof(cpu_model));

    for (unsigned int i = 0x80000000; i <= num_cpus; ++i) {
        __cpuid(i, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);

        if (i == 0x80000002)
            memcpy(cpu_model, cpu_info, sizeof(cpu_info));
        else if (i == 0x80000003)
            memcpy(cpu_model + 16, cpu_info, sizeof(cpu_info));
        else if (i == 0x80000004)
            memcpy(cpu_model + 32, cpu_info, sizeof(cpu_info));
    }

    std::cerr << "CPU: " << cpu_model << std::endl;
}

/// AtomicMax for floats
__device__ static inline void atomicMax(float *address, float val) {
    int *address_as_i = (int *)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

/// An early-exit AtomicMax for floats
__device__ static inline void atomicMaxExit(float *address, float val) {
    if (*address >= val) {
        return;
    }
    atomicMax(address, val);
}

/// A basic max reduce
/// This is basically equivalent to what is done in the majority of `postcoh`
__global__ void reduce_naive(const float *__restrict__ input, const int size,
                             float *out, int *index_out) {
    float cur;
    float max = 0.0;
    int index = 0;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        cur = input[i];
        if (cur > max) {
            index = i;
            max = cur;
        }
    }
    atomicMax(out, max);
    __syncthreads();
    if (max == *out) {
        *index_out = index;
    }
}

__global__ void reduce_basic(const float *__restrict__ input, const int size,
                             float *out, int *index_out) {
    float cur;
    float max = 0.0;
    int index = 0;
    int chunk_size = (size / blockDim.x) + 1;
    for (int i = threadIdx.x * chunk_size;
         i < chunk_size * (threadIdx.x + 1) && i < size; ++i) {
        cur = input[i];
        if (cur > max) {
            index = i;
            max = cur;
        }
    }
    atomicMax(out, max);
    __syncthreads();
    if (max == *out) {
        *index_out = index;
    }
}

__global__ void reduce_basic_exit(const float *__restrict__ input,
                                  const int size, float *out, int *index_out) {
    float cur;
    float max = 0.0;
    int index = 0;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        cur = input[i];
        if (cur > max) {
            index = i;
            max = cur;
        }
    }
    atomicMaxExit(out, max);
    __syncthreads();
    if (max == *out) {
        *index_out = index;
    }
}

__global__ void reduce_shared(const float *__restrict__ input, const int size,
                              float *out, int *index_out) {
    __shared__ float shared_max;
    __shared__ int shared_index;

    if (0 == threadIdx.x) {
        shared_max = 0.f;
        shared_index = 0;
    }

    __syncthreads();

    float max = 0.f;
    int index = 0;

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float val = input[i];

        if (max < val) {
            max = val;
            index = i;
        }
    }

    atomicMax(&shared_max, max);

    __syncthreads();

    if (shared_max == max) {
        shared_index = index;
    }

    __syncthreads();

    if (0 == threadIdx.x) {
        *out = shared_max;
        *index_out = shared_index;
    }
}

__global__ void reduce_blocks(const float *__restrict__ input, const int size,
                              float *out, int *index_out) {
    __shared__ float shared_max;
    __shared__ int shared_index;

    if (0 == threadIdx.x) {
        shared_max = 0.f;
        shared_index = 0;
    }

    __syncthreads();

    float max = 0.f;
    int index = 0;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size;
         i += blockDim.x) {
        float val = input[i];

        if (max < val) {
            max = val;
            index = i;
        }
    }

    atomicMax(&shared_max, max);

    __syncthreads();

    if (shared_max == max) {
        shared_index = index;
    }

    __syncthreads();

    if (0 == threadIdx.x) {
        out[blockIdx.x] = shared_max;
        index_out[blockIdx.x] = shared_index;
    }
}

__global__ void reduce_warp(const float *__restrict__ input, const int size,
                            float *maxOut, int *maxIdxOut) {
    float local_max = 0.f;
    int index = 0;

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float val = input[i];

        if (local_max < val) {
            local_max = val;
            index = i;
        }
    }

    const unsigned int MASK = -1;
    float warp_max = local_max;
    for (int mask = (warpSize >> 1); mask > 0; mask >>= 1) {
        warp_max = max(warp_max, __shfl_xor_sync(MASK, warp_max, mask));
    }
    unsigned int mask = __ballot_sync(MASK, local_max == warp_max);
    index = __shfl_sync(MASK, index, __ffs(mask) - 1);

    int lane = 0;
    lane = threadIdx.x & (warpSize - 1);

    if (lane == 0) {
        int warp_index = threadIdx.x / warpSize;
        maxOut[warp_index] = warp_max;
        maxIdxOut[warp_index] = index;
    }
}

int main(int argc, char **argv) {
    get_cpus();
    gpu::get_info();
    const long long N = 25600000;
    const int iterations = 1000;

    gpu::Vector<float> input(N);
    gpu::Vector<float> output(1024 / 32);
    gpu::Vector<int> output_index(1024 / 32);

    std::default_random_engine engine;
    std::uniform_real_distribution<> distribution(0, 1000);
    std::generate(input.begin(), input.end(),
                  [&]() { return distribution(engine); });

    cudaMemPrefetchAsync(input.data(), input.size() * sizeof(float), 0);
    cudaMemPrefetchAsync(output.data(), output.size() * sizeof(float), 0);
    cudaMemPrefetchAsync(output_index.data(), output_index.size() * sizeof(int),
                         0);

    GPUASSERT(cudaDeviceSynchronize());

    gpu::check_memory();

    auto bench = [&](double baseline, double other) {
        return ((other - baseline) / (other + baseline)) * 100;
    };

    auto standard = std::max_element(input.begin(), input.end());
    auto pos = standard - input.begin();
    auto actual_max = *standard;
    auto baseline = gpu::benchmark(iterations, "cpu_naive", [&]() {
        auto standard = std::max_element(input.begin(), input.end());
        actual_max = *standard;
    });
    std::cout << baseline << std::endl;

    auto next = gpu::benchmark(iterations, "reduce_naive", [&]() {
        reduce_naive<<<1, 1024>>>(input.data(), input.size(), output.data(),
                                  output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    std::cout << next << "\t(" << bench(baseline, next) << "%)" << std::endl;
    assert(output[0] == actual_max);
    output[0] = -1;

    next = gpu::benchmark(iterations, "reduce_basic", [&]() {
        reduce_basic<<<1, 1024>>>(input.data(), input.size(), output.data(),
                                  output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    std::cout << next << "\t(" << bench(baseline, next) << "%)" << std::endl;
    assert(output[0] == actual_max);
    output[0] = -1;

    next = gpu::benchmark(iterations, "reduce_basic_exit", [&]() {
        reduce_basic_exit<<<1, 1024>>>(input.data(), input.size(),
                                       output.data(), output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    std::cout << next << "\t(" << bench(baseline, next) << "%)" << std::endl;
    assert(output[0] == actual_max);
    output[0] = -1;

    next = gpu::benchmark(iterations, "reduce_shared", [&]() {
        reduce_shared<<<1, 1024>>>(input.data(), input.size(), output.data(),
                                   output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    std::cout << next << "\t(" << bench(baseline, next) << "%)" << std::endl;
    assert(output[0] == actual_max);

    next = gpu::benchmark(iterations, "reduce_blocks", [&]() {
        reduce_blocks<<<4, 1024>>>(input.data(), input.size(), output.data(),
                                   output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    std::cout << next << "\t(" << bench(baseline, next) << "%)" << std::endl;
    assert(output[0] == actual_max);
    for (auto &a : output) {
        a = -1;
    }

    next = gpu::benchmark(iterations, "reduce_warp", [&]() {
        reduce_warp<<<4, 1024>>>(input.data(), input.size(), output.data(),
                                 output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    std::cout << next << "\t(" << bench(baseline, next) << "%)" << std::endl;
    assert(*std::max_element(output.begin(), output.end()) == actual_max);
}
