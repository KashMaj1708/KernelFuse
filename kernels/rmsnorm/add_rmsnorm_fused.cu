// Phase 8 fused add + RMSNorm — matches vLLM 0.8.5 fused_add_rms_norm_kernel semantics.
// bf16 IO, fp32 variance accumulate. In-place: input -> normalized out, residual -> x+residual.
//
// CLI:
//   add_rmsnorm_fused.exe <rows> <cols> <eps> <x.bin> <residual.bin> <weight.bin>
//     <out_x.bin> <out_residual.bin>
//
// Bins are raw little-endian __nv_bfloat16 (2 bytes/elem).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_bf16.h>
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

static constexpr int kMaxBlock = 1024;

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

struct Bf16Vec8 {
    __nv_bfloat16 val[8];

    __device__ Bf16Vec8& operator+=(const Bf16Vec8& other) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float a = bf16_to_f32(val[i]);
            float b = bf16_to_f32(other.val[i]);
            val[i] = f32_to_bf16(a + b);
        }
        return *this;
    }

    __device__ float sum_squares() const {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float x = bf16_to_f32(val[i]);
            acc = fmaf(x, x, acc);
        }
        return acc;
    }

    __device__ Bf16Vec8& operator*=(float s) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            val[i] = f32_to_bf16(bf16_to_f32(val[i]) * s);
        }
        return *this;
    }

    __device__ Bf16Vec8& mul_weight(const Bf16Vec8& w) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            val[i] = f32_to_bf16(bf16_to_f32(val[i]) * bf16_to_f32(w.val[i]));
        }
        return *this;
    }
};

__device__ void block_reduce_sum(float* smem, float val, int tid, int block_size) {
    smem[tid] = val;
    __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
}

__global__ void fused_add_rms_norm_bf16_scalar(
    __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    float epsilon, int hidden_size) {
    __shared__ float smem[kMaxBlock];
    __shared__ float s_inv_rms;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const size_t base = static_cast<size_t>(row) * hidden_size;

    float variance = 0.0f;
    for (int idx = tid; idx < hidden_size; idx += block_size) {
        const size_t off = base + idx;
        float z = bf16_to_f32(input[off]) + bf16_to_f32(residual[off]);
        variance = fmaf(z, z, variance);
        residual[off] = f32_to_bf16(z);
    }

    block_reduce_sum(smem, variance, tid, block_size);
    if (tid == 0) {
        s_inv_rms = rsqrtf(smem[0] / static_cast<float>(hidden_size) + epsilon);
    }
    __syncthreads();

    const float inv = s_inv_rms;
    for (int idx = tid; idx < hidden_size; idx += block_size) {
        const size_t off = base + idx;
        float x = bf16_to_f32(residual[off]) * inv;
        x *= bf16_to_f32(weight[idx]);
        input[off] = f32_to_bf16(x);
    }
}

__global__ void fused_add_rms_norm_bf16_vec8(
    __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    float epsilon, int hidden_size) {
    __shared__ float smem[kMaxBlock];
    __shared__ float s_inv_rms;

    const int vec_hidden = hidden_size / 8;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    auto* input_v = reinterpret_cast<Bf16Vec8*>(input + static_cast<size_t>(row) * hidden_size);
    auto* residual_v =
        reinterpret_cast<Bf16Vec8*>(residual + static_cast<size_t>(row) * hidden_size);
    auto* weight_v = reinterpret_cast<const Bf16Vec8*>(weight);

    float variance = 0.0f;
    for (int idx = tid; idx < vec_hidden; idx += block_size) {
        Bf16Vec8 temp = input_v[idx];
        temp += residual_v[idx];
        variance += temp.sum_squares();
        residual_v[idx] = temp;
    }

    block_reduce_sum(smem, variance, tid, block_size);
    if (tid == 0) {
        s_inv_rms = rsqrtf(smem[0] / static_cast<float>(hidden_size) + epsilon);
    }
    __syncthreads();

    const float inv = s_inv_rms;
    for (int idx = tid; idx < vec_hidden; idx += block_size) {
        Bf16Vec8 temp = residual_v[idx];
        temp *= inv;
        temp.mul_weight(weight_v[idx]);
        input_v[idx] = temp;
    }
}

static void read_bf16_bin(const char* path, std::vector<__nv_bfloat16>& dst, size_t count) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open %s for read\n", path);
        std::exit(1);
    }
    dst.resize(count);
    size_t n = std::fread(dst.data(), sizeof(__nv_bfloat16), count, f);
    std::fclose(f);
    if (n != count) {
        fprintf(stderr, "short read on %s (%zu / %zu bf16)\n", path, n, count);
        std::exit(1);
    }
}

static void write_bf16_bin(const char* path, const __nv_bfloat16* src, size_t count) {
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "failed to open %s for write\n", path);
        std::exit(1);
    }
    size_t n = std::fwrite(src, sizeof(__nv_bfloat16), count, f);
    std::fclose(f);
    if (n != count) {
        fprintf(stderr, "short write on %s (%zu / %zu bf16)\n", path, n, count);
        std::exit(1);
    }
}

static bool ptr16_aligned(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

static void launch_fused(
    __nv_bfloat16* d_input,
    __nv_bfloat16* d_residual,
    const __nv_bfloat16* d_weight,
    int num_tokens,
    int hidden_size,
    float eps,
    bool use_vec8) {
    const int max_block = (num_tokens < 256) ? 1024 : 256;
    const int block = std::min(hidden_size, max_block);
    dim3 grid(num_tokens);

    if (use_vec8) {
        fused_add_rms_norm_bf16_vec8<<<grid, block>>>(
            d_input, d_residual, d_weight, eps, hidden_size);
    } else {
        fused_add_rms_norm_bf16_scalar<<<grid, block>>>(
            d_input, d_residual, d_weight, eps, hidden_size);
    }
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
    if (argc != 9) {
        fprintf(stderr,
                "usage: %s <rows> <cols> <eps> <x.bin> <residual.bin> <weight.bin> "
                "<out_x.bin> <out_residual.bin>\n",
                argv[0]);
        return 1;
    }

    const int rows = std::atoi(argv[1]);
    const int cols = std::atoi(argv[2]);
    const float eps = static_cast<float>(std::atof(argv[3]));
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "invalid shape\n");
        return 1;
    }

    const size_t n = static_cast<size_t>(rows) * cols;
    std::vector<__nv_bfloat16> h_x, h_r, h_w;
    read_bf16_bin(argv[4], h_x, n);
    read_bf16_bin(argv[5], h_r, n);
    read_bf16_bin(argv[6], h_w, static_cast<size_t>(cols));

    __nv_bfloat16 *d_x = nullptr, *d_r = nullptr, *d_w = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_w, static_cast<size_t>(cols) * sizeof(__nv_bfloat16)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, h_r.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_w, h_w.data(), static_cast<size_t>(cols) * sizeof(__nv_bfloat16),
                   cudaMemcpyHostToDevice));

    const bool aligned =
        ptr16_aligned(d_x) && ptr16_aligned(d_r) && ptr16_aligned(d_w) && (cols % 8 == 0);
    launch_fused(d_x, d_r, d_w, rows, cols, eps, aligned);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_r.data(), d_r, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    write_bf16_bin(argv[7], h_x.data(), n);
    write_bf16_bin(argv[8], h_r.data(), n);

    printf("PATH %s\n", aligned ? "vec8" : "scalar");
    printf("ROWS %d\n", rows);
    printf("COLS %d\n", cols);

    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_w);
    return 0;
}
