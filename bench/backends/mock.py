"""Mock backend — deterministic streaming latency for harness debug (no GPU)."""

from __future__ import annotations

import hashlib
import time

from bench.schema import GenerateRequest, GenerateResult, tpot_from_e2e


def _sleep_ms(ms: float) -> None:
    # time.sleep under ~15ms is unreliable on Windows; busy-wait for harness fidelity.
    if ms <= 0:
        return
    if ms >= 15.0:
        time.sleep(ms / 1000.0)
        return
    deadline = time.perf_counter() + ms / 1000.0
    while time.perf_counter() < deadline:
        pass


class MockBackend:
    name = "mock"
    model_id = "mock-toy"

    def __init__(
        self,
        ttft_ms: float = 5.0,
        per_token_ms: float = 0.5,
    ) -> None:
        self.ttft_ms = ttft_ms
        self.per_token_ms = per_token_ms

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def generate(self, req: GenerateRequest) -> GenerateResult:
        h = hashlib.sha256((req.request_id + req.prompt).encode()).digest()
        jitter = (h[0] / 255.0) * 2.0  # 0–2 ms on TTFT
        n = max(1, int(req.max_new_tokens))

        t0 = time.perf_counter()
        _sleep_ms(self.ttft_ms + jitter)
        ttft = (time.perf_counter() - t0) * 1000.0

        intervals: list[float] = []
        for _ in range(n - 1):
            t_tok = time.perf_counter()
            _sleep_ms(self.per_token_ms)
            intervals.append((time.perf_counter() - t_tok) * 1000.0)

        latency = (time.perf_counter() - t0) * 1000.0
        tpot = tpot_from_e2e(ttft, latency, n)
        return GenerateResult(
            request_id=req.request_id,
            latency_ms=latency,
            output_tokens=n,
            ok=True,
            ttft_ms=ttft,
            tpot_ms=tpot,
            token_intervals_ms=tuple(intervals),
        )
