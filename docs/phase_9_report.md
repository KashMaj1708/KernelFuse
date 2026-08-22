# Phase 9 report — decode attention kernel (ceiling)

## Exit gate

**None required.** Stretch goal only.

## What shipped

- Standalone CUDA microbench: `kernels/attention/decode_attn_bench.cu`
- Torch reference: `kernels/attention/decode_attn_ref.py`
- Correctness: `scripts/phase9/decode_attn_correctness.py` — **fp32 kernel output vs fp32 ref**, with atol/rtol and a **near-tolerance mutation test**

## A100 timing — CUDA events

| Backend | seq=512 min | seq=2048 min | scaling |
|---------|------------:|-------------:|---------|
| torch_ref (events) | 104.7 µs | 106.2 µs | ~1.01× |
| CUDA kernel (events) | 177.1 µs | 562.8 µs | ~3.18× |

torch_ref stays flat under 4× KV — **overhead floor**. Measured CUDA/torch ratios are a **lower bound on torch’s advantage**, not torch’s true kernel time.

## Correctness (fp32 compare)

bf16 has 8 mantissa bits (~0.4% relative quantum). Independent softmax reduction orders typically disagree by ~1e-4 relative in fp32 — **below the bf16 quantum** — so bit-exact bf16 zeros are the **expected** outcome of a bf16-output compare, not evidence of a strong test.

Harness now writes **fp32** from the CUDA path, compares with `atol=rtol=5e-3`, and mutates by `1.5×atol` (not 0.1).

| seq | max_abs | max_rel | pass |
|----:|--------:|--------:|:----:|
| 128 | 9.6e-4 | 3.7e-3 | ✓ |
| 512 | 4.5e-4 | 3.6e-3 | ✓ |
| 2048 | 1.3e-4 | 3.6e-3 | ✓ |

**Mutation test:** eps=0.0075 → max_abs≈7.5e-3, breaks `allclose` (harness can fail at the tolerance scale).

## Limits

- No serving integration; no FA/GQA-quality kernel.
- torch_ref absolute time remains overhead-dominated.
- bf16-output bitwise compare is insensitive by construction; use the fp32 path above.
