# Phase 7 predictions (write before measuring)

> **Errata:** the Phase 7 “cold” TTFT ≈ 108.8 ms anchor is **overturned**. See [`docs/phase_7_errata.md`](phase_7_errata.md). We have not measured a definitive cold prefill; Phase 8 flush arm was 129.5 ms @ 25.6% hit (`ok=False`).

Target: Vast.ai **A100 SXM4 80GB**, Qwen/Qwen2.5-7B-Instruct @ `a09a35458c702b33eeacc393d103063234e8bc28`, dtype **bfloat16**, CUDA graphs **on**, torch **cu12x** pin (avoid cu130 FA miss).

## Batch-1 decode floor

| Quantity | Value |
|----------|-------|
| Params | ~7.6B |
| Weight bytes (bf16/fp16) | ~15.2 GB |
| A100 SXM peak HBM | **Superseded for as-run SKU:** 40GB SXM is 1555 GB/s; **80GB SXM is 2039 GB/s HBM2e**. This table originally used 1555 / ~1300 achievable. |
| Floor TPOT ≈ weights / BW | Was 15.2e9 / 1.3e12 ≈ **11.7 ms** (40GB napkin). **As-run 80GB floor:** 15.2e9 / 2.039e12 ≈ **7.45 ms**. See `docs/phase_7_report.md`. |
| Phase 6 TinyLlama eager TPOT | 26–32 ms (eager tax; different model) |
| Prior 40GB smoke (same model) | TPOT ≈ 12.19 ms (graphs + FA) |

**Alarm:** batch-1 TPOT ≳ 40–50 ms on A100 with graphs on → likely still eager, wrong dtype path, or bandwidth problem. Re-check `enforce_eager` and `nvidia-smi` clocks before trusting the sweep.

## KV / concurrency knee (80GB)

Qwen2.5-7B uses GQA (28 Q / 4 KV). Rough KV footprint ~100–150 KB/token (bf16).

| Free HBM after weights (~15 GB of 80 @ mem_util=0.90) | ~55–57 GB usable for KV |
| Concurrent tokens at 125 KB/tok | ~55e9/125e3 ≈ **440k tokens** |
| At max_model_len 4096, concurrent seqs | 440k/4096 ≈ **~100+ seqs** (order of magnitude) |

Expect the **throughput knee past c=64**, possibly near **c=128**. Sweep includes **128** so the curve is not truncated in the linear region (Phase 6 failure mode).

Prior 40GB napkin (~20 GB KV) predicted c=16–64; that prediction is **superseded** for this 80GB session.

## Regime contrast

| Regime | Shape | Expect |
|--------|-------|--------|
| Decode-heavy | in≤512, out=128/512 | TTFT ≪ e2e; sys tok/s rises with c until knee |
| Prefill-heavy | in=2048, out=32 | TTFT large fraction of e2e; engines may separate on prefill |

## Cell / time budget

| Block | Cells / backend | Est. wall (napkin) |
|-------|----------------:|--------------------|
| Main grid (c incl. 128) | 48 | 2–3.5 h |
| Prefill | 4 | 15–40 min |
| Install + smoke + FA gate | — | 20–40 min (TRT engine build separate, can be 30–90+ min) |
| ×3 backends | 156 | plan **8–12 h** wall + buffer |

Start TRT NGC pull in the background during the FA gate / vLLM sweep.

## Identity check (every backend smoke)

`TTFT_p50 + (out - 1) × TPOT_p50 ≈ e2e_p50` within a few tens of ms.

## Attention gate

`VLLM_ATTENTION_BACKEND=FLASH_ATTN` must log `FLASH_ATTN` / FlashAttention v2. If `TRITON_ATTN`, stop and fix torch/cu12x pin before SGLang/TRT.
