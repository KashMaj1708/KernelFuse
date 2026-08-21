# Pending work (post–Phase 4)

Documented so Phase 5 can proceed without losing the Phase 4 follow-ups. None of these block the Phase 4 exit gate.

---

## P4-1 — Block-size sweep (optional, high value)

**Claim to test:** the occupancy cliff at width 8192 is partly a **launch-config artifact**, not a fundamental shared-memory limit.

At 33.8 KiB/block, smem caps residency at **1 block/SM** regardless of threads. Current launch uses **256 threads/block** → 256 threads/SM = **25%** of Turing’s 1024-thread limit. Launching **1024 threads/block** still fits 1 block (same 33.8 KiB) but restores **full thread occupancy** (4× the warps).

| width | current | proposed | expected threads/SM |
|------:|---------|----------|--------------------:|
| 8192 | 1 × 256 | 1 × **1024** | 256 → **1024** (full) |
| 4096 | 3 × 256 = 768 | 2 × **512** = 1024 | 768 → **1024** (full) |

This directly attacks the **MLP** limit diagnosed in Phase 4 (more resident warps → more outstanding requests). Should narrow the twin gap; scalar smem is hurt most by thin residency.

**Costs:** reduction tree grows (8 → 10 levels at 1024); barrier pressure rises (current barrier stall at 8192 is **0.65%** — headroom exists).

**Work:** parameterize `blockDim` (CLI or compile-time), Occupancy API check, CUDA-event BW + ncu twin pass at 256/512/1024. Prefer T4 for apples-to-apples.

**Better Phase 4 punchline if confirmed:** *“the occupancy cliff was a launch-config artifact, not a fundamental limit.”*

---

## P4-2 — Store path note (clarified; no code change needed)

`fixes_1` Step 6 required float4 on **stores** as well as loads. Code in `rmsnorm_fused_vec4.cu` **does** vectorize the bulk store (`out4[i] = o`) and weight loads (`w4`). Scalar remains only for the tail and the misaligned path.

The Phase 4 mechanism section previously blamed “reduction + weight + store still scalar” for the 2.65× `inst_executed` shortfall vs 4×. **Stores are not the shortfall.** Correct attribution: reduction tree + scalar tail + loop overhead. Report updated.

---

## P4-3 — Re-run smem_4096 events with longer warmup

T4 follow-up sampled smem_4096 while SM was still ramping (**690 MHz** post vs **1320 MHz** for later cases). Reported **170.7 GB/s** understates a fully boosted T4 and feeds the 4096→8192 time ratio.

**Work:** one CUDA-event re-run (warmup ≥50, or discard until clocks.sm ≥ 1200 MHz) for smem_4096 / smem_8192 / vec4_8192 on T4. Update `t4_event_bench.csv` and the Phase 4 table.

---

## Priority vs project timeline

Phase 4 is **closed** on stated criteria. P4-1–P4-3 are optional polish. Project effort is now ~⅓ done; the remaining ~⅔ is the **serving benchmark layer** (Phases 5–8). Do not let kernel polish delay the harness.

## Phase 5 status

**Closed** (exit gate). Post-close schema updates (do not rewrite archived CSVs):

- **TTFT / TPOT** on `GenerateResult` via streaming (`mock` busy-wait; `hf_local` `TextIteratorStreamer`)
- **System** vs **per-request** throughput both reported
- Concurrency axis labelled **placeholder** until Phase 6 engines
- Next matrix: [`bench/config_matrix_phase6_v1.yaml`](../bench/config_matrix_phase6_v1.yaml) (wider seq / concurrency) — do not edit v1

**Docker (before Phase 6):** run `scripts/install_docker_wsl_e.sh` inside Ubuntu WSL (sudo). Data-root → `/mnt/e/Docker/engine`. Skip Docker Desktop.
