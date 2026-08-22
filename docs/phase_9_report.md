# Phase 9 report — decode attention kernel (ceiling)

## Exit gate

**None required.** Stretch goal only.

## What shipped

- Standalone CUDA microbench: `kernels/attention/decode_attn_bench.cu`
- Torch reference: `kernels/attention/decode_attn_ref.py`
- Correctness CSV path: `scripts/phase9/decode_attn_correctness.py` (CUDA `load_inline` vs ref → max-abs / max-rel), plus a **mutation test**

## A100 timing — CUDA events (re-run)

| Backend | seq=512 min | seq=2048 min | scaling |
|---------|------------:|-------------:|---------|
| torch_ref (events) | 104.7 µs | 106.2 µs | **~1.01×** |
| CUDA kernel (events, load_inline) | 177.1 µs | 562.8 µs | **~3.18×** |

torch_ref stays flat under 4× KV — still an **overhead floor**. CUDA tracks work (~3×). Fair event compare: naive CUDA is **1.7×–5.3× slower** than torch_ref at these shapes.

**Ratio framing:** with torch_ref overhead-floored at ~105 µs, the measured ratio is a **lower bound on torch’s advantage**, not a measurement of torch’s true kernel time. The same in-loop amortization used in Phase 8 would be needed to resolve absolute torch work time.

Standalone exe earlier (176.8 → 690.4 µs) matches the same scaling story.

## Correctness

`scripts/phase9/decode_attn_correctness.py` on A100:

| seq | max_abs | max_rel | pass |
|----:|--------:|--------:|:----:|
| 128 | 0 | 0 | ✓ |
| 512 | 0 | 0 | ✓ |
| 2048 | 0 | 0 | ✓ |

Exact zeros mean bf16 outputs matched the reference for these seeds. That is stronger agreement than independent softmax reductions usually give; treat it as “matched on these seeds,” not as proof of bitwise-identical algorithms in general.

**Mutation test:** perturb one element of the CUDA output by ~0.1 and re-diff → max-abs ≈ 0.10, harness fails closed. So a correctness CSV that passes is evidence the comparison can fail — not a dead harness. Artifact: `reports/phase9/decode_attn_correctness.csv`.

## Limits

- No serving integration.
- No FA/GQA-quality kernel.
- torch_ref absolute time remains overhead-dominated; reported speedup of torch over CUDA is a lower bound.
- Exact max-abs 0 is surprising; mutation test rules out a no-op comparator, not all shared-path bugs.
