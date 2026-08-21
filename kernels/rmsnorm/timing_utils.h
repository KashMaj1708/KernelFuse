// Shared host helpers for Phase 3 CUDA-event timing (median of samples).
#pragma once

#include <algorithm>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

inline float median_f(std::vector<float> samples) {
    if (samples.empty()) {
        return 0.0f;
    }
    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();
    if (n % 2 == 1) {
        return samples[n / 2];
    }
    return 0.5f * (samples[n / 2 - 1] + samples[n / 2]);
}

// Time `launch` callable: void() that only enqueues GPU work (no sync inside).
template <typename LaunchFn>
inline float bench_cuda_events(int warmup, int iters, LaunchFn&& launch) {
    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    cudaDeviceSynchronize();

    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        launch();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        samples.push_back(ms);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return median_f(std::move(samples));
}
