"""Unit tests for percentile math (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.metrics import percentile, summarize_latencies


def test_percentile_odd():
    s = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(s, 50) == 3.0
    assert percentile(s, 100) == 5.0
    assert percentile(s, 0) == 1.0


def test_summarize():
    # 100 samples 1..100 → p50≈50, p90≈90, p99≈99 (nearest-rank)
    s = list(range(1, 101))
    out = summarize_latencies(s, (50, 90, 99))
    assert out["p50"] == 50.0
    assert out["p90"] == 90.0
    assert out["p99"] == 99.0


if __name__ == "__main__":
    test_percentile_odd()
    test_summarize()
    print("ok")
