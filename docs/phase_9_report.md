# Phase 9 report — decode attention kernel (ceiling)

## Exit gate

**None required.** Stretch goal only.

## What shipped

- Standalone CUDA microbench: `kernels/attention/decode_attn_bench.cu`
- Torch reference: `kernels/attention/decode_attn_ref.py`
- Correctness CSV path: `scripts/phase9/decode_attn_correctness.py` (CUDA `load_inline` vs ref → max-abs / max-rel)

## A100 timing (wall / event-naive exe)

| Backend | seq=512 | seq=2048 |
|---------|--------:|---------:|
| CUDA naive exe | 176.8 µs | 690.4 µs (~3.9×) |
| torch_ref (wall) | 108.7 µs | 120.6 µs (~1.11×) |

### Interpretation (methodology)

torch_ref’s **11%** time growth under **4×** KV traffic is the signature of a **dispatch / overhead floor**, not measured matmul work. The CUDA kernel’s ~3.9× scaling is the only arm clearly tracking work.

Therefore **“naive CUDA is slower than torch” is not established by this table** — it compares a work-dominated path to an overhead-dominated path. The conclusion may still be true after CUDA-event / graph / ncu measurement; this dataset does not show it.

## Correctness

Run (local or rental CUDA):

```bash
python scripts/phase9/decode_attn_correctness.py --out reports/phase9/decode_attn_correctness.csv
```

Pass criterion: max-abs < 0.05 bf16 vs `decode_attn_ref` for listed seq lengths. Artifact is the CSV (closes the “correctness-first without artifact” contradiction).

## Limits

- No serving integration.
- No FA/GQA-quality kernel.
- Standalone exe timings need the same CUDA-event discipline as Phase 8 microbench before any speed claim.
