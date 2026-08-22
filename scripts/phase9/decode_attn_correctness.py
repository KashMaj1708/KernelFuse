#!/usr/bin/env python3
"""Phase 9 — CUDA decode attention vs ref, compared in fp32 with a real tolerance.

bf16 outputs matching bit-exactly is *expected*: 8 mantissa bits (~0.4% rel) swamp
typical softmax-order disagreement (~1e-4 rel in fp32). Compare fp32 kernel output
vs fp32 reference, and mutate near the tolerance (not 100x above it).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

from kernels.attention.decode_attn_ref import decode_attn_ref

# Kernel writes fp32 so the harness can bound error below the bf16 quantum.
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cmath>

using bf16 = __nv_bfloat16;

__global__ void decode_attn_kernel_f32(
    const bf16* __restrict__ q,
    const bf16* __restrict__ k_cache,
    const bf16* __restrict__ v_cache,
    float* __restrict__ out,
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
    out[d] = acc;
  }
}

torch::Tensor decode_attn_cuda_f32(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16);
  int seq = k.size(0);
  int dim = q.size(0);
  auto out = torch::empty({dim}, q.options().dtype(torch::kFloat32));
  float scale = 1.f / sqrtf((float)dim);
  int threads = 256;
  size_t smem = (size_t)(seq + threads) * sizeof(float);
  decode_attn_kernel_f32<<<1, threads, smem>>>(
      reinterpret_cast<bf16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<bf16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<bf16*>(v.data_ptr<at::BFloat16>()),
      out.data_ptr<float>(),
      seq, dim, scale);
  return out;
}
"""

# Absolute / relative tolerances on fp32 outputs. Softmax order differences of
# ~1e-4 rel are expected; keep headroom but stay well below the bf16 quantum (~4e-3).
ABS_TOL = 5e-3
REL_TOL = 5e-3
# Mutation near the abs tolerance (not 0.1 = 20x above).
MUTATION_EPS = 1.5 * ABS_TOL


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
        name="decode_attn_corr_f32",
        cpp_sources="torch::Tensor decode_attn_cuda_f32(torch::Tensor, torch::Tensor, torch::Tensor);",
        cuda_sources=_CUDA_SRC,
        functions=["decode_attn_cuda_f32"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )

    torch.manual_seed(args.seed)
    rows: list[dict] = []
    for seq in args.seqs:
        q = torch.randn(args.dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(seq, args.dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(seq, args.dim, device="cuda", dtype=torch.bfloat16)
        # fp32 reference (same entrypoint, upcast inside)
        ref = decode_attn_ref(q, k, v).float()
        out = mod.decode_attn_cuda_f32(q.contiguous(), k.contiguous(), v.contiguous())
        torch.cuda.synchronize()
        diff = (out - ref).abs()
        max_abs = float(diff.max().item())
        # Elementwise max_rel is dominated by near-zero ref elements; report
        # max_abs / max|ref| (scale-normalized) and a floored relative max.
        ref_abs = ref.abs()
        max_ref = float(ref_abs.max().item())
        max_abs_over_max_ref = max_abs / max(max_ref, 1e-12)
        floor = 1e-3 * max(max_ref, 1e-12)
        mask = ref_abs >= floor
        if bool(mask.any()):
            max_rel_floored = float((diff[mask] / ref_abs[mask]).max().item())
        else:
            max_rel_floored = float("nan")
        ok = bool(
            torch.allclose(out, ref, rtol=REL_TOL, atol=ABS_TOL, equal_nan=False)
        )
        margin = REL_TOL / max(max_abs_over_max_ref, 1e-12)
        rows.append(
            {
                "seq": seq,
                "dim": args.dim,
                "max_abs": f"{max_abs:.6e}",
                "max_abs_over_max_ref": f"{max_abs_over_max_ref:.6e}",
                "max_rel_floored": f"{max_rel_floored:.6e}",
                "abs_tol": ABS_TOL,
                "rel_tol": REL_TOL,
                "tol_margin": f"{margin:.2f}",
                "pass": int(ok),
            }
        )
        print(
            f"seq={seq} max_abs={max_abs:.4e} "
            f"max_abs/max|ref|={max_abs_over_max_ref:.4e} "
            f"max_rel_floored={max_rel_floored:.4e} "
            f"pass={ok} margin={margin:.2f}x (atol={ABS_TOL} rtol={REL_TOL})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")

    # Mutation near tolerance: must fail closed.
    q = torch.randn(args.dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(512, args.dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(512, args.dim, device="cuda", dtype=torch.bfloat16)
    ref = decode_attn_ref(q, k, v).float()
    out = mod.decode_attn_cuda_f32(q.contiguous(), k.contiguous(), v.contiguous())
    out_mut = out.clone()
    out_mut[0] = out_mut[0] + MUTATION_EPS
    mut_ok = torch.allclose(out_mut, ref, rtol=REL_TOL, atol=ABS_TOL)
    mut_abs = float((out_mut - ref).abs().max().item())
    if mut_ok:
        print(
            f"MUTATION_TEST FAIL: eps={MUTATION_EPS} still allclose "
            f"(max_abs={mut_abs:.4e})"
        )
        return 1
    print(
        f"MUTATION_TEST PASS: eps={MUTATION_EPS} breaks allclose "
        f"(max_abs={mut_abs:.4e})"
    )

    if any(int(r["pass"]) == 0 for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
