// Phase 1: naive (unfused) RMSNorm — correctness only.
// Pass 1: one thread per row → inv_rms[row]
// Pass 2: one thread per element → out = x * inv_rms * weight
//
// CLI:
//   rmsnorm_naive.exe <rows> <cols> <eps> <x.bin> <weight.bin> <out.bin>
// All tensors are row-major float32. x: [rows, cols], weight: [cols], out: [rows, cols].

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

__global__ void rmsnorm_inv_rms_naive(const float* __restrict__ x,
                                      float* __restrict__ inv_rms,
                                      int rows, int cols, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float* row_ptr = x + static_cast<size_t>(row) * cols;
    // fixes_1 Step 4: fp32 accumulation (matches production kernels / timing story).
    float sum_sq = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float v = row_ptr[c];
        sum_sq = fmaf(v, v, sum_sq);
    }
    float mean_sq = sum_sq / static_cast<float>(cols);
    inv_rms[row] = rsqrtf(mean_sq + eps);
}

__global__ void rmsnorm_normalize_naive(const float* __restrict__ x,
                                        const float* __restrict__ inv_rms,
                                        const float* __restrict__ weight,
                                        float* __restrict__ out,
                                        int rows, int cols) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t n = static_cast<size_t>(rows) * cols;
    if (idx >= n) {
        return;
    }
    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    out[idx] = x[idx] * inv_rms[row] * weight[col];
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

static void run_naive(const float* h_x, const float* h_w, float* h_out,
                      int rows, int cols, float eps) {
    size_t n = static_cast<size_t>(rows) * cols;
    float *d_x = nullptr, *d_w = nullptr, *d_out = nullptr, *d_inv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv, rows * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, cols * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 256;
    int reduce_blocks = (rows + threads - 1) / threads;
    rmsnorm_inv_rms_naive<<<reduce_blocks, threads>>>(d_x, d_inv, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());

    int norm_blocks = static_cast<int>((n + threads - 1) / threads);
    rmsnorm_normalize_naive<<<norm_blocks, threads>>>(d_x, d_inv, d_w, d_out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_inv));
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

        float *d_x = nullptr, *d_w = nullptr, *d_out = nullptr, *d_inv = nullptr;
        CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w, cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_inv, rows * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

        const int threads = 256;
        int reduce_blocks = (rows + threads - 1) / threads;
        int norm_blocks = static_cast<int>((n + threads - 1) / threads);
        auto launch = [&]() {
            rmsnorm_inv_rms_naive<<<reduce_blocks, threads>>>(d_x, d_inv, rows, cols,
                                                              eps);
            rmsnorm_normalize_naive<<<norm_blocks, threads>>>(d_x, d_inv, d_w, d_out,
                                                              rows, cols);
        };

        float med = bench_cuda_events(warmup, iters, launch);
        std::printf("PATH naive\n");
        std::printf("MEDIAN_MS %.6f\n", med);
        std::fflush(stdout);

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_w));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_inv));
        return 0;
    }

    std::vector<float> h_out(n);
    read_bin(argv[4], h_x.data(), n);
    read_bin(argv[5], h_w.data(), cols);
    run_naive(h_x.data(), h_w.data(), h_out.data(), rows, cols, eps);
    std::printf("PATH naive\n");
    std::fflush(stdout);
    write_bin(argv[6], h_out.data(), n);
    return 0;
}
