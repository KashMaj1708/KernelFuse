#!/usr/bin/env python3
"""Grep Phase 7 server logs for prefix/radix cache hit rates (offline, no GPU)."""
from __future__ import annotations

import argparse
from pathlib import Path

from bench.cache_obs import prefix_hit_rate_from_log


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path, help="Server log (e.g. vllm_server.log, STATUS.txt)")
    p.add_argument(
        "--backend",
        choices=("vllm", "sglang", "trtllm", "auto"),
        default="auto",
    )
    args = p.parse_args()
    backend = args.backend
    if backend == "auto":
        name = args.log.name.lower()
        if "sglang" in name:
            backend = "sglang"
        elif "trt" in name:
            backend = "trtllm"
        else:
            backend = "vllm"
    rate = prefix_hit_rate_from_log(args.log, backend)
    if rate is None:
        print(f"no prefix/radix hit rate found in {args.log}")
        return 1
    print(f"last reported prefix cache hit rate: {rate:.2f}% ({backend})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
