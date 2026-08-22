// Shared fused add+RMSNorm device kernels (Phase 8).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace kernelfuse {

static constexpr int kAddRmsNormMaxBlock = 1024;

// Fixed segments/thread so local[] is only indexed by a compile-time loop
// variable after unroll. Dynamic n_local indexing forces a stack frame
// (local memory = DRAM) and recreates the two-pass traffic we are fixing.
static constexpr int kVecPerThread = 2;  // covers hidden=3584 with block>=224

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

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

struct Bf16Vec8 {
    __nv_bfloat16 val[8];
};

__global__ void fused_add_rms_norm_bf16_scalar(
    __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    float epsilon, int hidden_size) {
    // Scalar fallback: stage in shared memory (row fits at typical hidden).
    extern __shared__ float srow[];
    __shared__ float smem_red[kAddRmsNormMaxBlock];
    __shared__ float s_inv_rms;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const size_t base = static_cast<size_t>(row) * hidden_size;

    float variance = 0.0f;
    for (int idx = tid; idx < hidden_size; idx += block_size) {
        const size_t off = base + idx;
        float z = bf16_to_f32(input[off]) + bf16_to_f32(residual[off]);
        srow[idx] = z;
        residual[off] = f32_to_bf16(z);
        variance = fmaf(z, z, variance);
    }

    block_reduce_sum(smem_red, variance, tid, block_size);
    if (tid == 0) {
        s_inv_rms = rsqrtf(smem_red[0] / static_cast<float>(hidden_size) + epsilon);
    }
    __syncthreads();

    const float inv = s_inv_rms;
    for (int idx = tid; idx < hidden_size; idx += block_size) {
        const size_t off = base + idx;
        float x = srow[idx] * inv * bf16_to_f32(weight[idx]);
        input[off] = f32_to_bf16(x);
    }
}

__global__ void fused_add_rms_norm_bf16_vec8(
    __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    float epsilon, int hidden_size) {
    // Post-add values stay in registers (compile-time indices only).
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

    float local[kVecPerThread][8];
    float variance = 0.0f;

#pragma unroll
    for (int t = 0; t < kVecPerThread; ++t) {
        const int idx = tid + t * block_size;
        if (idx < vec_hidden) {
            Bf16Vec8 in = input_v[idx];
            Bf16Vec8 rs = residual_v[idx];
            Bf16Vec8 temp;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float z = bf16_to_f32(in.val[i]) + bf16_to_f32(rs.val[i]);
                local[t][i] = z;
                temp.val[i] = f32_to_bf16(z);
                variance = fmaf(z, z, variance);
            }
            residual_v[idx] = temp;
        }
    }

    block_reduce_sum(smem, variance, tid, block_size);
    if (tid == 0) {
        s_inv_rms = rsqrtf(smem[0] / static_cast<float>(hidden_size) + epsilon);
    }
    __syncthreads();

    const float inv = s_inv_rms;
#pragma unroll
    for (int t = 0; t < kVecPerThread; ++t) {
        const int idx = tid + t * block_size;
        if (idx < vec_hidden) {
            Bf16Vec8 out;
            Bf16Vec8 w = weight_v[idx];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float x = local[t][i] * inv * bf16_to_f32(w.val[i]);
                out.val[i] = f32_to_bf16(x);
            }
            input_v[idx] = out;
        }
    }
}

inline int pick_vec_block(int vec_hidden) {
    // Power-of-two blocks only (block_reduce_sum assumes that).
    // Need ceil(vec_hidden / block) <= kVecPerThread.
    int block = 256;
    while (block < kAddRmsNormMaxBlock &&
           (vec_hidden + block - 1) / block > kVecPerThread) {
        block *= 2;
    }
    if (vec_hidden < block) {
        int b = 32;
        while (b < vec_hidden && b < kAddRmsNormMaxBlock) b *= 2;
        block = b;
    }
    return block;
}

inline void launch_fused_add_rms_norm_bf16(
    __nv_bfloat16* input,
    __nv_bfloat16* residual,
    const __nv_bfloat16* weight,
    int num_tokens,
    int hidden_size,
    float epsilon,
    cudaStream_t stream) {
    dim3 grid(num_tokens);

    const bool aligned =
        (reinterpret_cast<uintptr_t>(input) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(residual) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(weight) % 16 == 0) && (hidden_size % 8 == 0);

    if (aligned) {
        const int block = pick_vec_block(hidden_size / 8);
        fused_add_rms_norm_bf16_vec8<<<grid, block, 0, stream>>>(
            input, residual, weight, epsilon, hidden_size);
    } else {
        // Dynamic smem = one fp32 per hidden element.
        const int block = hidden_size < 256 ? (((hidden_size + 31) / 32) * 32) : 256;
        const size_t smem = static_cast<size_t>(hidden_size) * sizeof(float);
        fused_add_rms_norm_bf16_scalar<<<grid, block, smem, stream>>>(
            input, residual, weight, epsilon, hidden_size);
    }
}

}  // namespace kernelfuse
