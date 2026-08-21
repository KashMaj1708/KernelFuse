"""Backend-agnostic serving benchmark types.

Streaming contract (Phase 5+): every backend that can emit tokens incrementally
fills TTFT / TPOT. Phase 6 engines (vLLM, SGLang, TRT-LLM) all support streaming;
mock and hf_local implement it now so the schema does not change under load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence


@dataclass(frozen=True)
class GenerateRequest:
    prompt: str
    max_new_tokens: int
    request_id: str = ""


@dataclass(frozen=True)
class GenerateResult:
    request_id: str
    latency_ms: float
    output_tokens: int
    ok: bool = True
    error: str | None = None
    # Streaming metrics (None if backend could not stream / failed early).
    ttft_ms: float | None = None
    # Mean inter-token gap after the first token: (e2e - ttft) / (n - 1).
    tpot_ms: float | None = None
    token_intervals_ms: tuple[float, ...] = ()


def tpot_from_intervals(intervals_ms: Sequence[float]) -> float | None:
    """TPOT from per-token arrival gaps after the first token (ms)."""
    if len(intervals_ms) < 1:
        return None
    return float(sum(intervals_ms) / len(intervals_ms))


def tpot_from_e2e(ttft_ms: float, latency_ms: float, output_tokens: int) -> float | None:
    if output_tokens < 2:
        return None
    return (latency_ms - ttft_ms) / (output_tokens - 1)


class Backend(Protocol):
    name: str
    model_id: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def generate(self, req: GenerateRequest) -> GenerateResult: ...


@dataclass
class CellSpec:
    matrix_version: str
    backend: str
    model: str
    batch_size: int
    input_tokens: int
    output_tokens: int
    concurrency: int
    warmup_requests: int
    num_requests: int
    seed: int


@dataclass
class CellResult:
    cell: CellSpec
    latencies_ms: list[float] = field(default_factory=list)
    ttft_ms: list[float] = field(default_factory=list)
    tpot_ms: list[float] = field(default_factory=list)
    # Per-request generation rates (output_tokens / e2e_s), not system rate.
    per_request_tok_s: list[float] = field(default_factory=list)
    output_tokens_total: int = 0
    wall_s: float = 0.0
    errors: int = 0

    @property
    def system_throughput_tok_s(self) -> float:
        """Aggregate: total output tokens / measurement-window wall clock."""
        if self.wall_s <= 0:
            return 0.0
        return self.output_tokens_total / self.wall_s

    @property
    def mean_per_request_tok_s(self) -> float:
        if not self.per_request_tok_s:
            return 0.0
        return sum(self.per_request_tok_s) / len(self.per_request_tok_s)

    # Back-compat alias used by older call sites.
    @property
    def throughput_tok_s(self) -> float:
        return self.system_throughput_tok_s
