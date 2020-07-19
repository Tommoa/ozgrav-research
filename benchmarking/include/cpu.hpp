#pragma once

#include <algorithm>
#include <cassert>
#include <cpuid.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

namespace cpu {
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
    std::cerr << "\t" << std::thread::hardware_concurrency() << " threads"
              << std::endl;

    uint32_t eax, ebx, ecx, edx;
    if (__get_cpuid(0x80000006, &eax, &ebx, &ecx, &edx)) {
        std::cerr << "\tLine size: " << (ecx & 0xff)
                  << "B, Cache Size: " << ((ecx >> 16) & 0xffff) << "KB"
                  << std::endl;
    }
}

// Tell the compiler to not optimize this away
template <typename T> void do_not_optimize(T &ptr) {
    asm volatile("" : : "g"(ptr) : "memory");
}

// Fake a read/write
void clobber() { asm volatile("" : : : "memory"); }

template <typename F>
double benchmark(int iterations, const std::string mode, F &&function) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        function();
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << std::setw(20) << mode << ":\t";
    return diff.count();
}

double bench(double baseline, double other) {
    return ((other - baseline) / baseline) * 100;
}

template <typename T, typename Container>
void finish_benchmark(double baseline, double time, T expected,
                      Container &output) {
    std::cout << time << "\t(" << std::showpos << cpu::bench(baseline, time)
              << "%)" << std::noshowpos << std::endl;
    assert(*std::max_element(output.begin(), output.end()) == expected);
    for (auto &a : output) {
        a = -1;
    }
}

} // namespace cpu
