# Phase 7 predictions (write before measuring)

Target: Vast.ai A100 SXM4 40GB, Qwen/Qwen2.5-7B-Instruct @ `a09a35458c702b33eeacc393d103063234e8bc28`, dtype **bfloat16**, CUDA graphs **on**.

## Batch-1 decode floor

| Quantity | Value |
|----------|-------|
| Params | ~7.6B |
| Weight bytes (bf16/fp16) | ~15.2 GB |
| A100 SXM peak HBM | 1555 GB/s (spec); achievable often ~1200–1400 GB/s |
| Floor TPOT ≈ weights / BW | 15.2e9 / 1.3e12 ≈ **11.7 ms/token** |
| Phase 6 TinyLlama eager TPOT | 26–32 ms (eager tax; different model) |

**Alarm:** batch-1 TPOT ≳ 40–50 ms on A100 with graphs on → likely still eager, wrong dtype path, or bandwidth problem. Re-check `enforce_eager` and `nvidia-smi` clocks before trusting the sweep.

## KV / concurrency knee (order-of-magnitude)

Qwen2.5-7B uses GQA (28 Q / 4 KV). Rough KV footprint ~100–150 KB/token (bf16, both layers).

| Free HBM after weights (~15 GB used of 40) | ~24 GB |
| Concurrent tokens at 125 KB/tok | ~24e9/125e3 ≈ **190k tokens** |
| At max_model_len 4096, concurrent seqs | 190k/4096 ≈ **~45 seqs** (order of magnitude) |

Expect the **throughput knee somewhere in c=16–64**, not at c=4 (Phase 6 linear region). If still scaling at 64, extend to 128 for a few (in,out) pairs only.

## Regime contrast

| Regime | Shape | Expect |
|--------|-------|--------|
| Decode-heavy | in≤512, out=128/512 | TTFT ≪ e2e; sys tok/s rises with c until knee |
| Prefill-heavy | in=2048, out=32 | TTFT large fraction of e2e; engines may separate on prefill |

## Cell / time budget

| Block | Cells / backend | Est. wall (napkin) |
|-------|----------------:|--------------------|
| Main grid | 42 | 1.5–3 h |
| Prefill | 4 | 15–40 min |
| Install + smoke | — | 20–40 min (TRT engine build separate, can be 30–90+ min) |
| ×3 backends | 138 | plan **6–10 h** wall + buffer |

At ~$0.53/hr, budget ~$5–8 for a clean run if scripts are non-interactive. Interactive debugging doubles that.

## Identity check (every backend smoke)

`TTFT_p50 + (out - 1) × TPOT_p50 ≈ e2e_p50` within a few tens of ms.
