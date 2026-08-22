#!/usr/bin/env python3
"""Isolated add+RMSNorm microbench: kernelfuse vs vLLM _custom_ops."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch


def bench_fn(fn, x, r, w, eps: float, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn(x, r, w, eps)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x, r, w, eps)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--cols", type=int, default=3584)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rows, cols = args.rows, args.cols
    eps = 1e-6
    device = "cuda"

    x_k = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    r_k = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    w = torch.ones(cols, device=device, dtype=torch.bfloat16)

    import kernelfuse

    def kf_fn(x, r, w_, e):
        x.copy_(x_k)
        r.copy_(r_k)
        kernelfuse.fused_add_rms_norm(x, r, w_, e)

    from vllm import _custom_ops as vops

    def vllm_fn(x, r, w_, e):
        x.copy_(x_k)
        r.copy_(r_k)
        vops.fused_add_rms_norm(x, r, w_, e)

    kf_us = bench_fn(kf_fn, x_k.clone(), r_k.clone(), w, eps, args.iters, args.warmup)
    vllm_us = bench_fn(vllm_fn, x_k.clone(), r_k.clone(), w, eps, args.iters, args.warmup)

    row = {
        "rows": rows,
        "cols": cols,
        "kernelfuse_us": f"{kf_us:.4f}",
        "vllm_us": f"{vllm_us:.4f}",
        "speedup": f"{vllm_us / kf_us:.4f}",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(row.keys()))
        wcsv.writeheader()
        wcsv.writerow(row)

    print(
        f"rows={rows} cols={cols} kernelfuse={kf_us:.2f}us vllm={vllm_us:.2f}us "
        f"speedup={vllm_us/kf_us:.2f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
