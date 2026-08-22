"""Prefix-cache observability for serving benchmarks.

Scrape engine-reported cache hit rates from server logs or /metrics counters.
Used to confirm cross-cell cache effects without per-sample TTFT retention.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen

# vLLM V1 LoggingStatLogger (0.8.x): "Prefix cache hit rate: 42.3%"
_VLLM_LOG_HIT_RE = re.compile(
    r"prefix cache hit rate:\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    re.IGNORECASE,
)
# SGLang server log variants
_SGLANG_RADIX_RE = re.compile(
    r"(?:radix cache hit rate|cache hit rate):\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    re.IGNORECASE,
)

_VLLM_HITS_RE = re.compile(
    r"^vllm(?:_v1)?(?:_|:)prefix_cache_hits(?:_total)?\s+([0-9.eE+-]+)",
    re.MULTILINE,
)
_VLLM_QUERIES_RE = re.compile(
    r"^vllm(?:_v1)?(?:_|:)prefix_cache_queries(?:_total)?\s+([0-9.eE+-]+)",
    re.MULTILINE,
)


def _last_match(text: str, pattern: re.Pattern[str]) -> float | None:
    hits = pattern.findall(text)
    if not hits:
        return None
    return float(hits[-1])


def prefix_hit_rate_from_log(
    log_path: Path,
    backend: str,
    *,
    tail_bytes: int = 512_000,
) -> float | None:
    """Return the last reported prefix/radix hit rate (0–100) from a server log tail."""
    if not log_path.is_file():
        return None
    raw = log_path.read_bytes()
    if len(raw) > tail_bytes:
        raw = raw[-tail_bytes:]
    text = raw.decode("utf-8", errors="replace")
    b = backend.lower()
    if b == "vllm":
        val = _last_match(text, _VLLM_LOG_HIT_RE)
    elif b == "sglang":
        val = _last_match(text, _SGLANG_RADIX_RE)
    else:
        val = _last_match(text, _VLLM_LOG_HIT_RE) or _last_match(text, _SGLANG_RADIX_RE)
    return val


def prefix_hit_rate_from_metrics(metrics_url: str, timeout_s: float = 5.0) -> float | None:
    """Compute cumulative prefix hit rate from vLLM /metrics counters (0–100)."""
    try:
        with urlopen(metrics_url, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (URLError, TimeoutError, OSError, ValueError):
        return None
    hits = _parse_prom_counter(body, _VLLM_HITS_RE)
    queries = _parse_prom_counter(body, _VLLM_QUERIES_RE)
    if hits is None or queries is None or queries <= 0:
        return None
    return 100.0 * hits / queries


def _parse_prom_counter(body: str, pattern: re.Pattern[str]) -> float | None:
    total = 0.0
    found = False
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        m = pattern.match(line.strip())
        if m:
            total += float(m.group(1))
            found = True
    return total if found else None


def scrape_prefix_cache_hit_pct(
    backend: str,
    *,
    server_log: Path | None = None,
    metrics_url: str | None = None,
) -> float | None:
    """Best-effort prefix cache hit rate for a cell window (percent 0–100)."""
    if metrics_url:
        val = prefix_hit_rate_from_metrics(metrics_url)
        if val is not None:
            return val
    if server_log:
        return prefix_hit_rate_from_log(server_log, backend)
    return None


def flush_prefix_cache(base_url: str, backend: str, timeout_s: float = 10.0) -> bool:
    """Ask the server to drop prefix/radix cache between cells.

    Returns True only if an admin endpoint accepted the request. On vLLM 0.8.5 a
    successful HTTP response can still leave blocks resident when sequences or
    scheduler state still reference them — so callers that need a *cold* prefill
    must either restart the server or assert a near-zero hit rate on the next
    measurement cell (see ``assert_flush_effective``).
    """
    b = backend.lower()
    paths: Iterable[str]
    if b == "vllm":
        # vLLM admin routes vary by version; try common names.
        paths = ("/reset_prefix_cache", "/flush_cache", "/v1/reset_prefix_cache")
    elif b == "sglang":
        paths = ("/flush_cache",)
    else:
        return False
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    for path in paths:
        url = f"{root}{path}"
        try:
            req = urlopen(url, timeout=timeout_s)  # noqa: S310
            req.read()
            return True
        except (URLError, TimeoutError, OSError, ValueError):
            continue
    return False


def assert_flush_effective(
    flushed: bool,
    *,
    soft: bool = False,
    hit_pct: float | None = None,
    max_hit_pct: float | None = None,
) -> None:
    """Fail closed when a required flush HTTP-no-ops.

    Optionally (``max_hit_pct`` set) also fail when the cell's scraped hit rate
    shows residual cache — for deliberate cold arms only. Do not enable that on
    full sweeps: within-cell prompt reuse routinely yields non-zero hits.
    """
    if soft:
        return
    if not flushed:
        raise RuntimeError(
            "prefix-cache flush failed (HTTP no-op). Restart the server for a "
            "definitive cold arm, or pass --soft-flush to allow best-effort."
        )
    if max_hit_pct is not None and hit_pct is not None and hit_pct > max_hit_pct:
        raise RuntimeError(
            f"post-cell prefix_cache_hit_pct={hit_pct:.1f}% exceeds cold threshold "
            f"{max_hit_pct:.1f}% — flush did not empty the cache. Restart the "
            "server between cold/warm arms."
        )
