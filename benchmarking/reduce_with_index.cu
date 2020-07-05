#include "cpu.hpp"
#include "gpu.hpp"

#include <algorithm>
#include <cuda_profiler_api.h>
#if __cplusplus >= 201703L
#include <execution>
#endif
#include <iostream>
#include <random>

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

__global__ void reduce_naive_nobranch(const float *__restrict__ input,
                                      const int size, float *out,
                                      int *index_out) {
    float max = 0.0;
    int index = 0;
    int tmp = reinterpret_cast<int &>(max);
    float cur;
    int mask;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        cur = input[i];
        mask = -(max < cur);
        // x ^ ((x ^ y) & mask)
        // gives y when mask = -1, x if mask = 0
        index = index ^ ((index ^ i) & mask);
        // SPIIR does `max = (max + cur)*0.5 + (max - cur)*((max > cur)*0.5)
        // this would be fine except for floating point inaccuracies
        // The below does effectively the same but by just taking the bits of
        // the winning float
        tmp = tmp ^ ((tmp ^ reinterpret_cast<int &>(cur)) & mask);
        max = reinterpret_cast<float &>(tmp);
    }
    atomicMax(out, max);
    __syncthreads();
    if (max == *out) {
        *index_out = index;
    }
}

__global__ void reduce_early_exit(const float *__restrict__ input,
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
    int chunk = blockDim.x * gridDim.x;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += chunk) {
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

__global__ void reduce_chunked(const float *__restrict__ input, const int size,
                               float *out, int *index_out) {
    float cur;
    float max = 0.0;
    int index = 0;
    int chunk = blockDim.x << 3;
    for (int start = threadIdx.x << 3, end = (threadIdx.x + 1) << 3;
         start < size; start += chunk, end += chunk) {
        for (int i = start; i < end && i != size; ++i) {
            cur = input[i];
            if (cur > max) {
                index = i;
                max = cur;
            }
        }
    }
    atomicMax(out, max);
    __syncthreads();
    if (max == *out) {
        *index_out = index;
    }
}

__global__ void reduce_chunked_exit(const float *__restrict__ input,
                                    const int size, float *out,
                                    int *index_out) {
    float cur;
    float max = 0.0;
    int index = 0;
    int chunk = blockDim.x << 3;
    for (int start = threadIdx.x << 3, end = (threadIdx.x + 1) << 3;
         start < size; start += chunk, end += chunk) {
        for (int i = start; i < end && i != size; ++i) {
            cur = input[i];
            if (cur > max) {
                index = i;
                max = cur;
            }
        }
    }
    atomicMaxExit(out, max);
    __syncthreads();
    if (max == *out) {
        *index_out = index;
    }
}

__global__ void reduce_chunked_shared(const float *__restrict__ input,
                                      const int size, float *out,
                                      int *index_out) {
    __shared__ float shared_max;
    __shared__ int shared_index;

    if (0 == threadIdx.x) {
        shared_max = 0.f;
        shared_index = 0;
    }

    __syncthreads();

    float max = 0.f;
    int index = 0;

    int chunk = blockDim.x << 3;
    for (int start = threadIdx.x << 3, end = (threadIdx.x + 1) << 3;
         start < size; start += chunk, end += chunk) {
        for (int i = start; i < end && i != size; ++i) {
            float cur = input[i];
            if (cur > max) {
                index = i;
                max = cur;
            }
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

__global__ void reduce_chunked_blocks(const float *__restrict__ input,
                                      const int size, float *out,
                                      int *index_out) {
    __shared__ float shared_max;
    __shared__ int shared_index;

    if (0 == threadIdx.x) {
        shared_max = 0.f;
        shared_index = 0;
    }

    __syncthreads();

    float max = 0.f;
    int index = 0;
    int chunk = (gridDim.x * blockDim.x) << 3;
    int start = (threadIdx.x + blockIdx.x * blockDim.x) << 3;
    int end = start + 8;
    for (; start < size; start += chunk, end += chunk) {
        end = end ^ ((end ^ size) & -(end > size)); // min(end, size);
        for (int i = start; i < end; ++i) {
            float cur = input[i];
            if (cur > max) {
                index = i;
                max = cur;
            }
        }
    }

    /*
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size;
         i += blockDim.x) {
        float val = input[i];

        if (max < val) {
            max = val;
            index = i;
        }
    }
    */

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

int main(int argc, char **argv) {
    cpu::get_cpus();
    gpu::get_info();
    const long long N = 25600000;
    const int iterations = 1000;

    gpu::ManagedVector<float> input(N);
    gpu::ManagedVector<float> output(1024 / 32);
    gpu::ManagedVector<int> output_index(1024 / 32);

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

    std::cerr << std::endl;
    std::cout << std::setprecision(5) << std::fixed;

    auto standard = std::max_element(input.begin(), input.end());
    auto pos = standard - input.begin();
    auto actual_max = *standard;
    auto baseline = cpu::benchmark(iterations, "cpu_sequential", [&]() {
        auto standard = std::max_element(input.begin(), input.end());
        actual_max = *standard;
        cpu::do_not_optimize(standard);
        cpu::clobber();
    });
    std::cout << baseline << std::endl;

    double next;
#if __cplusplus >= 201703L
    next = cpu::benchmark(iterations, "cpu_parallel", [&]() {
        auto standard =
            std::max_element(std::execution::par, input.begin(), input.end());
        actual_max = *standard;
        cpu::do_not_optimize(standard);
        cpu::clobber();
    });
    std::cout << next << "\t(" << cpu::bench(baseline, next) << "%)" << std::endl;
#endif

    next = gpu::benchmark(iterations, "naive", [&]() {
        reduce_naive<<<1, 1024>>>(input.data(), input.size(), output.data(),
                                  output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    cpu::finish_benchmark(baseline, next, actual_max, output);

    next = gpu::benchmark(iterations, "nobranch", [&]() {
        reduce_naive_nobranch<<<1, 1024>>>(input.data(), input.size(),
                                           output.data(), output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    cpu::finish_benchmark(baseline, next, actual_max, output);

    next = gpu::benchmark(iterations, "early_exit", [&]() {
        reduce_early_exit<<<1, 1024>>>(input.data(), input.size(),
                                       output.data(), output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    cpu::finish_benchmark(baseline, next, actual_max, output);

    next = gpu::benchmark(iterations, "shared", [&]() {
        reduce_shared<<<1, 1024>>>(input.data(), input.size(), output.data(),
                                   output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    cpu::finish_benchmark(baseline, next, actual_max, output);

    next = gpu::benchmark(iterations, "blocks", [&]() {
        reduce_blocks<<<4, 1024>>>(input.data(), input.size(), output.data(),
                                   output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    cpu::finish_benchmark(baseline, next, actual_max, output);

    next = gpu::benchmark(iterations, "warp", [&]() {
        reduce_warp<<<4, 1024>>>(input.data(), input.size(), output.data(),
                                 output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    cpu::finish_benchmark(baseline, next, actual_max, output);

    next = gpu::benchmark(iterations, "chunked", [&]() {
        reduce_chunked<<<1, 1024>>>(input.data(), input.size(), output.data(),
                                    output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    cpu::finish_benchmark(baseline, next, actual_max, output);

    next = gpu::benchmark(iterations, "chunked_exit", [&]() {
        reduce_chunked_exit<<<1, 1024>>>(input.data(), input.size(),
                                         output.data(), output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    cpu::finish_benchmark(baseline, next, actual_max, output);

    next = gpu::benchmark(iterations, "chunked_shared", [&]() {
        reduce_chunked_shared<<<1, 1024>>>(input.data(), input.size(),
                                           output.data(), output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    cpu::finish_benchmark(baseline, next, actual_max, output);

    next = gpu::benchmark(iterations, "chunked_blocks", [&]() {
        reduce_chunked_blocks<<<4, 1024>>>(input.data(), input.size(),
                                           output.data(), output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
    cpu::finish_benchmark(baseline, next, actual_max, output);
}
