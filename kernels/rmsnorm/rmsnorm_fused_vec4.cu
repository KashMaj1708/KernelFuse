// Shared-memory fused RMSNorm with float4 bulk + scalar tail.
// Per-row: if 16-byte aligned, vectorize floor(N/4)*4 then scalar tail; else
// full scalar. Smem budget queried per-device (same as fused_smem).
//
// stdout:
//   MAX_SMEM_COLS <n>
//   PATH vec4|vec4_tail|global
//
// CLI:
//   rmsnorm_fused_vec4.exe <rows> <cols> <eps> <x.bin> <weight.bin> <out.bin>

#include <cmath>
#include <cstdint>
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
static constexpr int kStaticSmemReserve = 2048;

__device__ __forceinline__ bool ptr16_aligned(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

__global__ void rmsnorm_fused_global(const float* __restrict__ x,
                                     const float* __restrict__ weight,
                                     float* __restrict__ out, int cols,
                                     float eps) {
    __shared__ float spartial[kMaxThreads];
    __shared__ float sinv_rms;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_x = x + static_cast<size_t>(row) * cols;
    float* row_out = out + static_cast<size_t>(row) * cols;

    float sum_sq = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = row_x[c];
        sum_sq = fmaf(v, v, sum_sq);
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
        sinv_rms = rsqrtf(spartial[0] / static_cast<float>(cols) + eps);
    }
    __syncthreads();

    const float inv_rms = sinv_rms;
    for (int c = tid; c < cols; c += blockDim.x) {
        row_out[c] = row_x[c] * inv_rms * weight[c];
    }
}

__global__ void rmsnorm_fused_smem_vec4(const float* __restrict__ x,
                                        const float* __restrict__ weight,
                                        float* __restrict__ out, int cols,
                                        float eps) {
    extern __shared__ float srow[];
    __shared__ float spartial[kMaxThreads];
    __shared__ float sinv_rms;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_x = x + static_cast<size_t>(row) * cols;
    float* row_out = out + static_cast<size_t>(row) * cols;
    const float* row_w = weight;

    const bool row_aligned = ptr16_aligned(row_x) && ptr16_aligned(row_out);
    const int n4 = cols / 4;
    const int bulk = n4 * 4;

    float sum_sq = 0.0f;
    if (row_aligned && n4 > 0) {
        const float4* in4 = reinterpret_cast<const float4*>(row_x);
        float4* s4 = reinterpret_cast<float4*>(srow);
        for (int i = tid; i < n4; i += blockDim.x) {
            float4 v = in4[i];
            s4[i] = v;
            sum_sq = fmaf(v.x, v.x, sum_sq);
            sum_sq = fmaf(v.y, v.y, sum_sq);
            sum_sq = fmaf(v.z, v.z, sum_sq);
            sum_sq = fmaf(v.w, v.w, sum_sq);
        }
        for (int c = bulk + tid; c < cols; c += blockDim.x) {
            float v = row_x[c];
            srow[c] = v;
            sum_sq = fmaf(v, v, sum_sq);
        }
    } else {
        for (int c = tid; c < cols; c += blockDim.x) {
            float v = row_x[c];
            srow[c] = v;
            sum_sq = fmaf(v, v, sum_sq);
        }
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
        sinv_rms = rsqrtf(spartial[0] / static_cast<float>(cols) + eps);
    }
    __syncthreads();

    const float inv_rms = sinv_rms;
    if (row_aligned && n4 > 0) {
        float4* out4 = reinterpret_cast<float4*>(row_out);
        const float4* s4 = reinterpret_cast<const float4*>(srow);
        const float4* w4 = reinterpret_cast<const float4*>(row_w);
        for (int i = tid; i < n4; i += blockDim.x) {
            float4 v = s4[i];
            float4 w = w4[i];
            float4 o;
            o.x = v.x * inv_rms * w.x;
            o.y = v.y * inv_rms * w.y;
            o.z = v.z * inv_rms * w.z;
            o.w = v.w * inv_rms * w.w;
            out4[i] = o;
        }
        for (int c = bulk + tid; c < cols; c += blockDim.x) {
            row_out[c] = srow[c] * inv_rms * weight[c];
        }
    } else {
        for (int c = tid; c < cols; c += blockDim.x) {
            row_out[c] = srow[c] * inv_rms * weight[c];
        }
    }
}

static int max_smem_cols_from_bytes(int max_smem_bytes) {
    int usable = max_smem_bytes - kStaticSmemReserve;
    if (usable < static_cast<int>(sizeof(float))) {
        return 0;
    }
    return usable / static_cast<int>(sizeof(float));
}

static int resolve_smem_budget(int* out_bytes) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int optin = 0;
    int deflt = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &deflt, cudaDevAttrMaxSharedMemoryPerBlock, dev));

    int bytes = optin > 0 ? optin : deflt;
    cudaError_t attr = cudaFuncSetAttribute(
        rmsnorm_fused_smem_vec4, cudaFuncAttributeMaxDynamicSharedMemorySize,
        bytes);
    if (attr != cudaSuccess) {
        cudaGetLastError();
        bytes = deflt;
    }
    *out_bytes = bytes;
    return max_smem_cols_from_bytes(bytes);
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

static const char* run(const float* h_x, const float* h_w, float* h_out, int rows,
                       int cols, float eps, int max_cols) {
    size_t n = static_cast<size_t>(rows) * cols;
    float *d_x = nullptr, *d_w = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, cols * sizeof(float), cudaMemcpyHostToDevice));

    const char* path = "global";
    if (cols <= max_cols) {
        size_t shmem = static_cast<size_t>(cols) * sizeof(float);
        rmsnorm_fused_smem_vec4<<<rows, kMaxThreads, shmem>>>(d_x, d_w, d_out, cols,
                                                              eps);
        path = (cols % 4 == 0) ? "vec4" : "vec4_tail";
    } else {
        rmsnorm_fused_global<<<rows, kMaxThreads>>>(d_x, d_w, d_out, cols, eps);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_out));
    return path;
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

    int max_smem_bytes = 0;
    int max_cols = resolve_smem_budget(&max_smem_bytes);
    std::printf("MAX_SMEM_COLS %d\n", max_cols);
    std::fflush(stdout);

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

        const bool use_smem = cols <= max_cols;
        const char* path =
            use_smem ? ((cols % 4 == 0) ? "vec4" : "vec4_tail") : "global";
        size_t shmem = static_cast<size_t>(cols) * sizeof(float);
        auto launch = [&]() {
            if (use_smem) {
                rmsnorm_fused_smem_vec4<<<rows, kMaxThreads, shmem>>>(
                    d_x, d_w, d_out, cols, eps);
            } else {
                rmsnorm_fused_global<<<rows, kMaxThreads>>>(d_x, d_w, d_out, cols, eps);
            }
        };
        float med = bench_cuda_events(warmup, iters, launch);
        std::printf("PATH %s\n", path);
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
    const char* path = run(h_x.data(), h_w.data(), h_out.data(), rows, cols, eps,
                           max_cols);
    std::printf("PATH %s\n", path);
    std::fflush(stdout);
    write_bin(argv[6], h_out.data(), n);
    return 0;
}
