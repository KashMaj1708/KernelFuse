"""Latency percentiles and table formatting."""

from __future__ import annotations

from typing import Sequence


def percentile(sorted_samples: Sequence[float], p: float) -> float:
    """Nearest-rank percentile on a pre-sorted non-empty sequence. p in [0, 100]."""
    if not sorted_samples:
        raise ValueError("empty samples")
    if p <= 0:
        return float(sorted_samples[0])
    if p >= 100:
        return float(sorted_samples[-1])
    # nearest-rank: ceil(p/100 * n) with 1-based rank
    n = len(sorted_samples)
    rank = max(1, int((p / 100.0) * n + 0.999999999))  # ceil without import
    return float(sorted_samples[min(rank, n) - 1])


def summarize_latencies(samples_ms: Sequence[float], ps: Sequence[float] = (50, 90, 99)) -> dict[str, float]:
    s = sorted(float(x) for x in samples_ms)
    return {f"p{int(p)}": percentile(s, float(p)) for p in ps}
