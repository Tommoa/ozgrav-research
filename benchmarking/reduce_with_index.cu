#include "gpu.hpp"

#include <algorithm>
#include <cuda_profiler_api.h>
#include <exception>
#include <iostream>
#include <math.h>
#include <random>

/// AtomicMax for floats
__device__ static inline float atomicMax(float *address, float val) {
    int *address_as_i = (int *)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/// A basic max reduce
/// This is basically equivalent to what is done in the majority of `postcoh`
__global__ void reduce_basic(const float *__restrict__ input, const int size,
                             float *out, int *index_out) {
    float cur;
    float max = 0.0;
    int index = 0;
    // for (int i = threadIdx.x*blockDim.x; i < (threadIdx.x+1)*blockDim.x; i
    // ++) {
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

    const unsigned int MASK = 0xffffffff;
    float warp_max = 0.0;
    for (int mask = (warpSize >> 1); mask > 0; mask >>= 1) {
        warp_max = max(local_max, __shfl_xor_sync(MASK, local_max, mask));
    }
    unsigned int mask = __ballot_sync(MASK, local_max == warp_max);
    int lane = 0;
    for (; !(mask & 1); ++lane, mask >>= 1)
        ;
    index = __shfl_sync(MASK, index, lane);

    lane = threadIdx.x & (warpSize - 1);

    if (lane == 0) {
        int warp_index = threadIdx.x / warpSize;
        maxOut[warp_index] = warp_max;
        maxIdxOut[warp_index] = index;
    }
}

int main(int argc, char **argv) {
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

    gpu::benchmark(iterations, "reduce_basic", [&]() {
        reduce_basic<<<1, 1024>>>(input.data(), input.size(), output.data(),
                                  output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });

    gpu::benchmark(iterations, "reduce_shared", [&]() {
        reduce_shared<<<1, 1024>>>(input.data(), input.size(), output.data(),
                                   output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });

    gpu::benchmark(iterations, "reduce_blocks", [&]() {
        reduce_blocks<<<4, 1024>>>(input.data(), input.size(), output.data(),
                                   output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });

    gpu::benchmark(iterations, "reduce_warp", [&]() {
        reduce_warp<<<4, 1024>>>(input.data(), input.size(), output.data(),
                                 output_index.data());
        GPUASSERT(cudaGetLastError());
        GPUASSERT(cudaDeviceSynchronize());
    });
}
