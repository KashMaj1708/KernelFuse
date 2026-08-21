// Phase 2 fused RMSNorm (fixes_1 Step 4: fp32 accumulator by default).
// One block per row: block-reduce sum-of-squares → inv_rms in shared memory →
// normalize × weight. inv_rms never written to global memory.
//
// Build with -DACC_DOUBLE to restore the Phase-2 double accumulator (numerical
// reference only — do not time).
//
// CLI:
//   rmsnorm_fused.exe <rows> <cols> <eps> <x.bin> <weight.bin> <out.bin>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "timing_utils.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_err));                                 \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static constexpr int kMaxThreads = 256;

#ifdef ACC_DOUBLE
using AccT = double;
#else
using AccT = float;
#endif

__global__ void rmsnorm_fused(const float* __restrict__ x,
                              const float* __restrict__ weight,
                              float* __restrict__ out, int cols, float eps) {
    __shared__ AccT spartial[kMaxThreads];
    __shared__ float sinv_rms;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_x = x + static_cast<size_t>(row) * cols;
    float* row_out = out + static_cast<size_t>(row) * cols;

    AccT sum_sq = AccT(0);
    for (int c = tid; c < cols; c += blockDim.x) {
#ifdef ACC_DOUBLE
        double v = static_cast<double>(row_x[c]);
        sum_sq += v * v;
#else
        float v = row_x[c];
        sum_sq = fmaf(v, v, sum_sq);
#endif
    }
    spartial[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            spartial[tid] += spartial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean_sq = static_cast<float>(spartial[0] / static_cast<AccT>(cols));
        sinv_rms = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float inv_rms = sinv_rms;
    for (int c = tid; c < cols; c += blockDim.x) {
        row_out[c] = row_x[c] * inv_rms * weight[c];
    }
}

static void read_bin(const char* path, float* dst, size_t count) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open %s for read\n", path);
        std::exit(1);
    }
    size_t n = std::fread(dst, sizeof(float), count, f);
    std::fclose(f);
    if (n != count) {
        fprintf(stderr, "short read on %s (%zu / %zu floats)\n", path, n, count);
        std::exit(1);
    }
}

static void write_bin(const char* path, const float* src, size_t count) {
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "failed to open %s for write\n", path);
        std::exit(1);
    }
    size_t n = std::fwrite(src, sizeof(float), count, f);
    std::fclose(f);
    if (n != count) {
        fprintf(stderr, "short write on %s (%zu / %zu floats)\n", path, n, count);
        std::exit(1);
    }
}

static void run_fused(const float* h_x, const float* h_w, float* h_out, int rows,
                      int cols, float eps) {
    size_t n = static_cast<size_t>(rows) * cols;
    float *d_x = nullptr, *d_w = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, cols * sizeof(float), cudaMemcpyHostToDevice));

    rmsnorm_fused<<<rows, kMaxThreads>>>(d_x, d_w, d_out, cols, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_out));
}

int main(int argc, char** argv) {
    const bool is_bench = (argc >= 2 && std::strcmp(argv[1], "bench") == 0);
    if (!is_bench && argc != 7) {
        fprintf(stderr,
                "usage:\n"
                "  %s <rows> <cols> <eps> <x.bin> <weight.bin> <out.bin>\n"
                "  %s bench <rows> <cols> <eps> <warmup> <iters> <x.bin> <w.bin>\n",
                argv[0], argv[0]);
        return 2;
    }
    if (is_bench && argc != 9) {
        fprintf(stderr,
                "usage: %s bench <rows> <cols> <eps> <warmup> <iters> <x.bin> <w.bin>\n",
                argv[0]);
        return 2;
    }

    const int a0 = is_bench ? 2 : 1;
    int rows = std::atoi(argv[a0]);
    int cols = std::atoi(argv[a0 + 1]);
    float eps = static_cast<float>(std::atof(argv[a0 + 2]));
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "rows and cols must be positive\n");
        return 2;
    }

    size_t n = static_cast<size_t>(rows) * cols;
    std::vector<float> h_x(n), h_w(cols);

    if (is_bench) {
        int warmup = std::atoi(argv[5]);
        int iters = std::atoi(argv[6]);
        read_bin(argv[7], h_x.data(), n);
        read_bin(argv[8], h_w.data(), cols);

        float *d_x = nullptr, *d_w = nullptr, *d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w, cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

        auto launch = [&]() {
            rmsnorm_fused<<<rows, kMaxThreads>>>(d_x, d_w, d_out, cols, eps);
        };
        float med = bench_cuda_events(warmup, iters, launch);
        std::printf("PATH fused\n");
        std::printf("MEDIAN_MS %.6f\n", med);
        std::fflush(stdout);

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_w));
        CUDA_CHECK(cudaFree(d_out));
        return 0;
    }

    std::vector<float> h_out(n);
    read_bin(argv[4], h_x.data(), n);
    read_bin(argv[5], h_w.data(), cols);
    run_fused(h_x.data(), h_w.data(), h_out.data(), rows, cols, eps);
    std::printf("PATH fused\n");
    std::fflush(stdout);
    write_bin(argv[6], h_out.data(), n);
    return 0;
}
