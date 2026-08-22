// Shared fused add+RMSNorm device kernels (Phase 8).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace kernelfuse {

static constexpr int kAddRmsNormMaxBlock = 1024;

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

__device__ inline void block_reduce_sum(float* smem, float val, int tid, int block_size) {
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
    __shared__ float smem[kAddRmsNormMaxBlock];
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
    __shared__ float smem[kAddRmsNormMaxBlock];
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

inline void launch_fused_add_rms_norm_bf16(
    __nv_bfloat16* input,
    __nv_bfloat16* residual,
    const __nv_bfloat16* weight,
    int num_tokens,
    int hidden_size,
    float epsilon,
    cudaStream_t stream) {
    const int max_block = (num_tokens < 256) ? 1024 : 256;
    const int block = hidden_size < max_block ? hidden_size : max_block;
    dim3 grid(num_tokens);

    const bool aligned =
        (reinterpret_cast<uintptr_t>(input) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(residual) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(weight) % 16 == 0) && (hidden_size % 8 == 0);

    if (aligned) {
        fused_add_rms_norm_bf16_vec8<<<grid, block, 0, stream>>>(
            input, residual, weight, epsilon, hidden_size);
    } else {
        fused_add_rms_norm_bf16_scalar<<<grid, block, 0, stream>>>(
            input, residual, weight, epsilon, hidden_size);
    }
}

}  // namespace kernelfuse
