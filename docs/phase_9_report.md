# Phase 9 report — decode attention kernel (ceiling)

## Exit gate

**None required.** Stretch goal only.

## What shipped

- Standalone CUDA microbench: `kernels/attention/decode_attn_bench.cu`
- Torch reference: `kernels/attention/decode_attn_ref.py`
- Correctness: fp32 kernel output vs fp32 ref; floored relative metric; near-tolerance mutation

## A100 timing — CUDA events

| Backend | seq=512 min | seq=2048 min | scaling |
|---------|------------:|-------------:|---------|
| torch_ref (events) | 104.7 µs | 106.2 µs | ~1.01× |
| CUDA kernel (events) | 177.1 µs | 562.8 µs | ~3.18× |

torch_ref stays flat under 4× KV — **overhead floor**. Event ratios are a **lower bound on torch’s advantage**.

## Correctness

bf16 bitwise zeros are **expected**, not surprising: 8 mantissa bits (~0.4% rel quantum) swamp typical softmax-order disagreement (~1e-4 rel in fp32). Compare in fp32.

**Elementwise `max_rel` is the wrong headline metric.** Absolute error fell ~7× from seq=128→2048 while elementwise max_rel sat flat at ~3.6e-3 — the signature of one near-zero reference element dominating the ratio. Report instead:

- `max_abs / max|ref|` (scale-normalized)
- `max_rel_floored` over elements with `|ref| ≥ 1e-3 · max|ref|`

| seq | max_abs | max_abs/max\|ref\| | max_rel_floored | tol_margin (vs rtol) | pass |
|----:|--------:|------------------:|----------------:|---------------------:|:----:|
| 128 | 9.6e-4 | 2.6e-3 | 3.7e-3 | **1.36×** | ✓ |
| 512 | 4.5e-4 | 2.0e-3 | 3.6e-3 | **1.37×** | ✓ |
| 2048 | 1.3e-4 | 1.7e-3 | 3.6e-3 | **1.38×** | ✓ |

atol = rtol = 5e-3. Binding metric is floored rel ≈ **72% of rtol** (~1.4× headroom). A bug that inflated that error by ~1.5× would still pass — tight, not comfortable.

**Mutation:** eps = 1.5×atol = 0.0075 → breaks `allclose` (harness can fail at the tolerance scale).

## Limits

- No serving integration; no FA/GQA-quality kernel.
- torch_ref absolute time remains overhead-dominated.
- Correctness headroom vs rtol is only ~1.4× on the floored relative metric.
