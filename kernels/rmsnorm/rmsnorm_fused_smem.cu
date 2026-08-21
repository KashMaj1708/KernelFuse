// Fused RMSNorm with row staged in dynamic shared memory.
// Smem budget is queried per-device (cudaDevAttrMaxSharedMemoryPerBlockOptin)
// so the same binary uses smem as far as each GPU allows (1650 vs A100/3090).
//
// stdout:
//   MAX_SMEM_COLS <n>
//   PATH smem|global
//
// CLI:
//   rmsnorm_fused_smem.exe <rows> <cols> <eps> <x.bin> <weight.bin> <out.bin>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

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

__global__ void rmsnorm_fused_smem(const float* __restrict__ x,
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

    float sum_sq = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = row_x[c];
        srow[c] = v;
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
        row_out[c] = srow[c] * inv_rms * weight[c];
    }
}

static int max_smem_cols_from_bytes(int max_smem_bytes) {
    int usable = max_smem_bytes - kStaticSmemReserve;
    if (usable < static_cast<int>(sizeof(float))) {
        return 0;
    }
    return usable / static_cast<int>(sizeof(float));
}

// Resolve usable dynamic smem: prefer opt-in max, fall back if SetAttribute rejects.
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
        rmsnorm_fused_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (attr != cudaSuccess) {
        cudaGetLastError();  // clear sticky error
        bytes = deflt;
        // Default limit needs no opt-in attribute.
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
        rmsnorm_fused_smem<<<rows, kMaxThreads, shmem>>>(d_x, d_w, d_out, cols, eps);
        path = "smem";
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
    if (argc != 7) {
        fprintf(stderr,
                "usage: %s <rows> <cols> <eps> <x.bin> <weight.bin> <out.bin>\n",
                argv[0]);
        return 2;
    }

    int rows = std::atoi(argv[1]);
    int cols = std::atoi(argv[2]);
    float eps = static_cast<float>(std::atof(argv[3]));
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "rows and cols must be positive\n");
        return 2;
    }

    int max_smem_bytes = 0;
    int max_cols = resolve_smem_budget(&max_smem_bytes);
    std::printf("MAX_SMEM_COLS %d\n", max_cols);
    std::fflush(stdout);

    size_t n = static_cast<size_t>(rows) * cols;
    std::vector<float> h_x(n), h_w(cols), h_out(n);
    read_bin(argv[4], h_x.data(), n);
    read_bin(argv[5], h_w.data(), cols);

    const char* path = run(h_x.data(), h_w.data(), h_out.data(), rows, cols, eps,
                           max_cols);
    std::printf("PATH %s\n", path);
    std::fflush(stdout);

    write_bin(argv[6], h_out.data(), n);
    return 0;
}
