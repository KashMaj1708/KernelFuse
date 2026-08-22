#!/usr/bin/env python3
"""Isolated add+RMSNorm microbench with CUDA events (not wall-clock).

For ops this small (~14 KB @ rows=1), wall-clock + synchronize measures
cudaLaunchKernel / host overhead, not kernel time. This harness:

- CUDA Event elapsed time (GPU timeline)
- Warmup discarded
- Optional CUDA Graph capture so launch overhead is amortized
- Reports **minimum** over trials (not mean)
- Copies inputs outside the timed region

Develop/run on any CUDA GPU; A100 numbers still need this methodology.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def _make_kf(x_src, r_src):
    import kernelfuse

    def fn(x, r, w, e):
        kernelfuse.fused_add_rms_norm(x, r, w, e)

    return fn


def _make_vllm(x_src, r_src):
    from vllm import _custom_ops as vops

    def fn(x, r, w, e):
        vops.fused_add_rms_norm(x, r, w, e)

    return fn


def bench_events(
    fn,
    x: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    *,
    warmup: int,
    iters_per_trial: int,
    trials: int,
    use_graph: bool,
) -> dict[str, float]:
    """Return min/median µs per call from CUDA events."""
    for _ in range(warmup):
        fn(x, r, w, eps)
    torch.cuda.synchronize()

    if use_graph:
        g = torch.cuda.CUDAGraph()
        # Static buffers — graph replays same tensors.
        with torch.cuda.graph(g):
            for _ in range(iters_per_trial):
                fn(x, r, w, eps)
        torch.cuda.synchronize()

        samples_us: list[float] = []
        for _ in range(trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            g.replay()
            end.record()
            torch.cuda.synchronize()
            # elapsed_time is ms for the whole replay of iters_per_trial calls
            samples_us.append(start.elapsed_time(end) * 1e3 / iters_per_trial)
    else:
        samples_us = []
        for _ in range(trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters_per_trial):
                fn(x, r, w, eps)
            end.record()
            torch.cuda.synchronize()
            samples_us.append(start.elapsed_time(end) * 1e3 / iters_per_trial)

    samples_us.sort()
    return {
        "min_us": samples_us[0],
        "median_us": samples_us[len(samples_us) // 2],
        "max_us": samples_us[-1],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--cols", type=int, default=3584)
    p.add_argument("--iters", type=int, default=200, help="Launches per timed trial")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--trials", type=int, default=21)
    p.add_argument("--graph", action="store_true", help="CUDA Graph capture + replay")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rows, cols = args.rows, args.cols
    eps = 1e-6
    device = "cuda"

    x0 = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    r0 = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    w = torch.ones(cols, device=device, dtype=torch.bfloat16)

    # Working buffers (mutated in-place by both ops).
    x_k, r_k = x0.clone(), r0.clone()
    x_v, r_v = x0.clone(), r0.clone()

    kf_fn = _make_kf(x0, r0)
    vllm_fn = _make_vllm(x0, r0)

    # Reset to identical inputs before each bench arm (outside timing).
    x_k.copy_(x0)
    r_k.copy_(r0)
    kf = bench_events(
        kf_fn,
        x_k,
        r_k,
        w,
        eps,
        warmup=args.warmup,
        iters_per_trial=args.iters,
        trials=args.trials,
        use_graph=args.graph,
    )
    x_v.copy_(x0)
    r_v.copy_(r0)
    vllm = bench_events(
        vllm_fn,
        x_v,
        r_v,
        w,
        eps,
        warmup=args.warmup,
        iters_per_trial=args.iters,
        trials=args.trials,
        use_graph=args.graph,
    )

    # Bandwidth napkin: 3 tensors × 2 B × rows × cols (x,r,w read; x,r write ≈)
    bytes_touch = rows * cols * 2 * 4  # rough upper bound
    row = {
        "rows": rows,
        "cols": cols,
        "graph": int(args.graph),
        "kernelfuse_min_us": f"{kf['min_us']:.4f}",
        "kernelfuse_median_us": f"{kf['median_us']:.4f}",
        "vllm_min_us": f"{vllm['min_us']:.4f}",
        "vllm_median_us": f"{vllm['median_us']:.4f}",
        "speedup_min": f"{vllm['min_us'] / kf['min_us']:.4f}",
        "bytes_touch_approx": bytes_touch,
        "note": "min over CUDA-event trials; wall-clock benches are not comparable",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(row.keys()))
        wcsv.writeheader()
        wcsv.writerow(row)

    print(
        f"rows={rows} cols={cols} graph={args.graph} "
        f"kf_min={kf['min_us']:.3f}us vllm_min={vllm['min_us']:.3f}us "
        f"speedup_min={vllm['min_us']/kf['min_us']:.3f}x "
        f"(report min; do not claim gain if both << memory bound)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
