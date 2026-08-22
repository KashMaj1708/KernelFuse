// Minimal decode attention: one query vs KV cache (bf16, fp32 softmax).
// nvcc -O3 -o decode_attn_bench decode_attn_bench.cu -lcudart

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

using bf16 = __nv_bfloat16;

__global__ void decode_attn_kernel(
    const bf16* __restrict__ q,
    const bf16* __restrict__ k_cache,
    const bf16* __restrict__ v_cache,
    bf16* __restrict__ out,
    int seq_len,
    int head_dim,
    float scale) {
  extern __shared__ float smem[];
  float* scores = smem;
  float* partial = smem + seq_len;

  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  float max_score = -INFINITY;
  for (int s = tid; s < seq_len; s += stride) {
    float dot = 0.f;
    for (int d = 0; d < head_dim; ++d) {
      dot += __bfloat162float(q[d]) * __bfloat162float(k_cache[s * head_dim + d]);
    }
    dot *= scale;
    scores[s] = dot;
    max_score = fmaxf(max_score, dot);
  }
  partial[tid] = max_score;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) partial[tid] = fmaxf(partial[tid], partial[tid + off]);
    __syncthreads();
  }
  max_score = partial[0];

  float sum = 0.f;
  for (int s = tid; s < seq_len; s += stride) {
    float e = expf(scores[s] - max_score);
    scores[s] = e;
    sum += e;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) partial[tid] += partial[tid + off];
    __syncthreads();
  }
  sum = partial[0];

  for (int d = tid; d < head_dim; d += stride) {
    float acc = 0.f;
    for (int s = 0; s < seq_len; ++s) {
      float w = scores[s] / sum;
      acc += w * __bfloat162float(v_cache[s * head_dim + d]);
    }
    out[d] = __float2bfloat16(acc);
  }
}

static void check(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    exit(1);
  }
}

int main(int argc, char** argv) {
  int seq_len = 512;
  int head_dim = 128;
  int iters = 200;
  if (argc > 1) seq_len = atoi(argv[1]);
  if (argc > 2) head_dim = atoi(argv[2]);

  size_t kv_bytes = (size_t)seq_len * head_dim * sizeof(bf16);
  size_t q_bytes = (size_t)head_dim * sizeof(bf16);
  bf16 *d_q, *d_k, *d_v, *d_out;
  check(cudaMalloc(&d_q, q_bytes), "malloc q");
  check(cudaMalloc(&d_k, kv_bytes), "malloc k");
  check(cudaMalloc(&d_v, kv_bytes), "malloc v");
  check(cudaMalloc(&d_out, q_bytes), "malloc out");

  std::vector<bf16> h_q(head_dim, __float2bfloat16(0.01f));
  std::vector<bf16> h_k(seq_len * head_dim);
  std::vector<bf16> h_v(seq_len * head_dim);
  for (int i = 0; i < seq_len * head_dim; ++i) {
    h_k[i] = __float2bfloat16(0.001f * (i % 17));
    h_v[i] = __float2bfloat16(0.002f * (i % 13));
  }
  check(cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice), "h2d q");
  check(cudaMemcpy(d_k, h_k.data(), kv_bytes, cudaMemcpyHostToDevice), "h2d k");
  check(cudaMemcpy(d_v, h_v.data(), kv_bytes, cudaMemcpyHostToDevice), "h2d v");

  float scale = 1.f / sqrtf((float)head_dim);
  int threads = 256;
  size_t smem = (seq_len + threads) * sizeof(float);
  decode_attn_kernel<<<1, threads, smem>>>(d_q, d_k, d_v, d_out, seq_len, head_dim, scale);
  check(cudaDeviceSynchronize(), "sync");

  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int i = 0; i < 20; ++i)
    decode_attn_kernel<<<1, threads, smem>>>(d_q, d_k, d_v, d_out, seq_len, head_dim, scale);
  check(cudaDeviceSynchronize(), "warmup");
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i)
    decode_attn_kernel<<<1, threads, smem>>>(d_q, d_k, d_v, d_out, seq_len, head_dim, scale);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, a, b);
  printf("decode_attn seq=%d dim=%d avg_us=%.2f\n", seq_len, head_dim, ms * 1000.f / iters);

  std::vector<bf16> h_out(head_dim);
  check(cudaMemcpy(h_out.data(), d_out, q_bytes, cudaMemcpyDeviceToHost), "d2h");
  printf("out0=%f\n", __bfloat162float(h_out[0]));

  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_out);
  return 0;
}
