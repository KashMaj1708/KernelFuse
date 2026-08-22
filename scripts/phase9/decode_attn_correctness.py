#!/usr/bin/env python3
"""Phase 9 — max-abs / max-rel of CUDA decode attention vs decode_attn_ref."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

from kernels.attention.decode_attn_ref import decode_attn_ref

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cmath>

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
    for (int d = 0; d < head_dim; ++d)
      dot += __bfloat162float(q[d]) * __bfloat162float(k_cache[s * head_dim + d]);
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
    for (int s = 0; s < seq_len; ++s)
      acc += (scores[s] / sum) * __bfloat162float(v_cache[s * head_dim + d]);
    out[d] = __float2bfloat16(acc);
  }
}

torch::Tensor decode_attn_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16);
  int seq = k.size(0);
  int dim = q.size(0);
  auto out = torch::empty_like(q);
  float scale = 1.f / sqrtf((float)dim);
  int threads = 256;
  size_t smem = (size_t)(seq + threads) * sizeof(float);
  decode_attn_kernel<<<1, threads, smem>>>(
      reinterpret_cast<bf16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<bf16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<bf16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
      seq, dim, scale);
  return out;
}
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seqs", type=int, nargs="+", default=[128, 512, 2048])
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    mod = load_inline(
        name="decode_attn_corr",
        cpp_sources="torch::Tensor decode_attn_cuda(torch::Tensor, torch::Tensor, torch::Tensor);",
        cuda_sources=_CUDA_SRC,
        functions=["decode_attn_cuda"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )

    torch.manual_seed(args.seed)
    rows: list[dict] = []
    for seq in args.seqs:
        q = torch.randn(args.dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(seq, args.dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(seq, args.dim, device="cuda", dtype=torch.bfloat16)
        ref = decode_attn_ref(q, k, v)
        out = mod.decode_attn_cuda(q.contiguous(), k.contiguous(), v.contiguous())
        torch.cuda.synchronize()
        diff = (out.float() - ref.float()).abs()
        max_abs = float(diff.max().item())
        max_rel = float((diff / ref.float().abs().clamp_min(1e-6)).max().item())
        rows.append(
            {
                "seq": seq,
                "dim": args.dim,
                "max_abs": f"{max_abs:.6e}",
                "max_rel": f"{max_rel:.6e}",
                "pass": int(max_abs < 0.05),
            }
        )
        print(f"seq={seq} max_abs={max_abs:.4e} max_rel={max_rel:.4e} pass={max_abs < 0.05}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    if any(int(r["pass"]) == 0 for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
