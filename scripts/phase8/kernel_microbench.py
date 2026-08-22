#!/usr/bin/env python3
"""Isolated add+RMSNorm microbench — CUDA events, amortized N-in-graph, rows sweep.

Shapes like [1, 3584] are latency-bound / occupancy-starved (one block, one SM).
The device-wide HBM napkin does not apply. Sweep rows to find the latency→BW knee.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def _make_kf():
    import kernelfuse

    def fn(x, r, w, e):
        kernelfuse.fused_add_rms_norm(x, r, w, e)

    return fn


def _make_vllm():
    from vllm import _custom_ops as vops

    def fn(x, r, w, e):
        vops.fused_add_rms_norm(x, r, w, e)

    return fn


def bench_amortized(
    fn,
    x: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    *,
    warmup: int,
    iters: int,
    trials: int,
    use_graph: bool,
) -> dict[str, float]:
    """N launches inside one timed region (graph or event bracket); report min µs/call."""
    for _ in range(warmup):
        fn(x, r, w, eps)
    torch.cuda.synchronize()

    if use_graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn(x, r, w, eps)
        torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            g.replay()  # one replay = `iters` kernel launches already captured
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1e3 / iters)
    else:
        samples = []
        for _ in range(trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn(x, r, w, eps)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1e3 / iters)

    samples.sort()
    return {"min_us": samples[0], "median_us": samples[len(samples) // 2], "max_us": samples[-1]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cols", type=int, default=3584)
    p.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[1, 8, 64, 512, 4096, 16384],
        help="Row sweep: find latency→bandwidth transition",
    )
    p.add_argument("--iters", type=int, default=200, help="Launches per timed trial / inside graph")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--trials", type=int, default=21)
    p.add_argument("--graph", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = "cuda"
    eps = 1e-6
    cols = args.cols
    kf_fn = _make_kf()
    vllm_fn = _make_vllm()

    rows_out: list[dict] = []
    for rows in args.rows:
        x0 = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
        r0 = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
        w = torch.ones(cols, device=device, dtype=torch.bfloat16)
        x_k, r_k = x0.clone(), r0.clone()
        x_v, r_v = x0.clone(), r0.clone()

        kf = bench_amortized(
            kf_fn, x_k, r_k, w, eps,
            warmup=args.warmup, iters=args.iters, trials=args.trials, use_graph=args.graph,
        )
        vllm = bench_amortized(
            vllm_fn, x_v, r_v, w, eps,
            warmup=args.warmup, iters=args.iters, trials=args.trials, use_graph=args.graph,
        )
        # Rough bytes: read x,r,w + write x,r ≈ 5 * rows * cols * 2
        bytes_ = rows * cols * 2 * 5
        row = {
            "rows": rows,
            "cols": cols,
            "graph": int(args.graph),
            "iters_amortized": args.iters,
            "kernelfuse_min_us": f"{kf['min_us']:.4f}",
            "vllm_min_us": f"{vllm['min_us']:.4f}",
            "speedup_min": f"{vllm['min_us'] / kf['min_us']:.4f}",
            "bytes_approx": bytes_,
            "gb_s_kf": f"{(bytes_ / (kf['min_us'] * 1e-6)) / 1e9:.2f}",
            "gb_s_vllm": f"{(bytes_ / (vllm['min_us'] * 1e-6)) / 1e9:.2f}",
        }
        rows_out.append(row)
        print(
            f"rows={rows:5d} graph={args.graph} "
            f"kf={kf['min_us']:.3f}us vllm={vllm['min_us']:.3f}us "
            f"ratio={vllm['min_us']/kf['min_us']:.3f} "
            f"GB/s_kf={row['gb_s_kf']} GB/s_vllm={row['gb_s_vllm']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows_out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
