# Phase 9 report — decode attention kernel (ceiling)

## Exit gate

**None required.** Stretch goal only.

## What shipped

- Standalone CUDA microbench: `kernels/attention/decode_attn_bench.cu`
- Torch reference: `kernels/attention/decode_attn_ref.py`
- Correctness CSV path: `scripts/phase9/decode_attn_correctness.py` (CUDA `load_inline` vs ref → max-abs / max-rel)

## A100 timing — CUDA events (re-run)

| Backend | seq=512 min | seq=2048 min | scaling |
|---------|------------:|-------------:|---------|
| torch_ref (events) | 104.7 µs | 106.2 µs | **~1.01×** |
| CUDA kernel (events, load_inline) | 177.1 µs | 562.8 µs | **~3.18×** |

torch_ref stays flat under 4× KV — still an **overhead floor**. CUDA tracks work (~3×). Fair event compare: naive CUDA is **1.7×–5.3× slower** than torch_ref at these shapes; that relative claim is now supported. Absolute “torch is fast at decode attn” is not — its wall/event time barely moves with seq.

Standalone exe earlier (176.8 → 690.4 µs) matches the same scaling story.

## Correctness

`scripts/phase9/decode_attn_correctness.py` on A100:

| seq | max_abs | max_rel | pass |
|----:|--------:|--------:|:----:|
| 128 | 0 | 0 | ✓ |
| 512 | 0 | 0 | ✓ |
| 2048 | 0 | 0 | ✓ |

Exact zeros here mean bf16 outputs matched the reference for these seeds (same math + deterministic reduction on this path). Artifact: `reports/phase9/decode_attn_correctness.csv`, `decode_attn_events.csv`.

## Limits

- No serving integration.
- No FA/GQA-quality kernel.
- Standalone exe timings need the same CUDA-event discipline as Phase 8 microbench before any speed claim.
- Event re-run confirms: CUDA slower than torch_ref **relatively**; torch_ref absolute time is still overhead-dominated.
