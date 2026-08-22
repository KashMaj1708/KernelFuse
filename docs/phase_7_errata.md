# Phase 7 errata — cold prefill anchor overturned (Phase 8 preamble)

**Date:** 2026-08-22 (post Phase 8 A100 preamble)

## Overturn

Phase 7 treated **TTFT ≈ 108.8 ms** (`in=2048 out=128 c=1`) as the best **cold** prefill datapoint (~83% MFU vs a corrected 28.3 TFLOPs bound). That claim does **not** survive Phase 8 instrumentation.

### Phase 8 flush arm (same model / A100 class)

| Arm | TTFT p50 | `prefix_cache_hit_pct` | Flush |
|-----|---------:|-----------------------:|-------|
| “Cold” (flush requested) | **129.5 ms** | **25.6%** | `ok=False` |
| Warm (no flush) | 20.8 ms | 58.0% | — |

A quarter of prefill blocks were still resident on the flush arm. That is **not** a cold measurement.

### Back-calculation

If ~25.6% of tokens hit cache, compute scales with ~74.4% of tokens:

`129.5 / 0.744 ≈ 174 ms` implied fully-cold TTFT.

Against 28.3 TFLOPs that is ~**52% MFU** — ordinary for a 2048-token prefill, not heroic.

Scaling the same residual-cache logic backward: **108.8 ms was never cold**. It is an **unknown-cache-state** datapoint; a plausible hit rate ~35–40% would also land near ~175 ms true cold. Both numbers reconcile once residual cache is admitted.

## Honest statement

1. **We have never measured a definitive cold prefill** on this Qwen2.5-7B / vLLM stack.
2. Closest instrumented point: **129.5 ms @ 25.6% hit** — still an **underestimate** of cold cost.
3. **108.8 ms** must not be cited as a cold anchor or as an MFU headline.

## Flush path (actionable)

On vLLM 0.8.5, `/reset_prefix_cache` can HTTP-succeed or fail while leaving blocks referenced by live sequences / scheduler state. Mid-sweep flush often **no-ops** (`CACHE_FLUSH ok=False` in Phase 8 logs).

**Reliable cold arm:** server restart between arms, **or** flush with zero in-flight requests and **assert** effectiveness.

Harness (post-errata):

- Default: **fail the run** if flush is required and HTTP returns failure (`ok=False`).
- Opt-in `--assert-cold-hit`: fail if scraped `prefix_cache_hit_pct` exceeds `--max-cold-hit-pct` (cold arms only — not full sweeps).
- Escape hatch: `--soft-flush` (legacy best-effort; forbids cold claims).

## Scope of retraction

Widens Phase 7’s existing honesty on `in=2048` decode/knee claims: **prefill TTFT absolute levels for “cold” are retracted** until a restart-gated cold arm exists. Decode ranking at `in=128` is unaffected.
