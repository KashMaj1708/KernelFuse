# Phase 8 predictions (write before measuring)

**Headline:** land a **fused add+RMSNorm custom op** into **vLLM 0.8.5** on Qwen2.5-7B, prove correctness + CUDA graph capture, measure kernel-level speedup in isolation. **SGLang is follow-up**, not co-primary.

Phase 7 hygiene items (cache on/off pair, output equivalence, one variance repeat) fold into the **single A100 preamble** on the integration measurement night — not a separate remount.

---

## Pre-registered null (end-to-end serving)

At batch-1 decode, each token reads ~**15.2 GB** of bf16 weights. RMSNorm per layer touches roughly **3584 × 2 B × 2** (input + output) ≈ **14 KB/layer** → **~800 KB/token** across 28 layers.

| Quantity | Value |
|----------|-------|
| Norm bytes / weight bytes | ~800 KB / 15.2 GB ≈ **0.005%** |
| Predicted e2e TPOT shift (batch-1) | **≪ 1%** — inside run-to-run noise |
| Predicted sys tok/s shift at knee | **≪ 1%** |

**Alarm if:** you expect end-to-end TPOT or peak sys tok/s to move materially after the swap. That would mean the swap broke something (graph fallback, extra launch, dtype path), not that the kernel “won.”

**Deliverable that *should* move:**

- Isolated kernel bench vs `F.rms_norm` / vLLM baseline on `[rows, 3584]` bf16 decode shapes
- Correctness vs CPU golden + vs reference add+RMSNorm on residual path
- CUDA graph capture survives (no alloc/sync/device query in forward)

This is the MIGrate null: **integration + mechanism**, not a headline tok/s claim.

---

## Integration target (explicit)

| Choice | Decision |
|--------|----------|
| **First engine** | **vLLM 0.8.5** (documented CustomOp path; lower friction than SGLang) |
| **Follow-up** | SGLang 0.5.x after vLLM gate passes |
| **Model** | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28` |
| **Hardware for A/B** | A100-SXM4-80GB (same class as Phase 7) |
| **Integration dev** | **Rented A100** (M2–M5 in one session). M1 correctness on 1650 ($0). T4/Colab optional, not required. |

Do **not** parallelize vLLM + SGLang integration in Phase 8. Pick vLLM, finish the gate, then open SGLang.

---

## Kernel scope (must ship before A/B)

Phase 1–4 kernels are **pure RMSNorm**, fp32, 48 KiB smem mental model. Production swap requires:

### 1. Fused add + RMSNorm (not norm alone)

vLLM `RMSNorm.forward` optionally fuses **residual add + write-back + norm**. Replacing only the norm **splits one fused kernel into two** and measures a regression unrelated to kernel quality.

**Required:** `add_rmsnorm` variant — in-place residual mutation, single launch, matching vLLM numerics (bf16 in/out, fp32 accumulate).

### 2. bf16 + vec8 + smem on datacenter SKUs

| Phase 1–4 | Phase 8 production |
|-----------|-------------------|
| fp32 IO | bf16 IO, fp32 accumulate |
| float4 (4× fp32) | **8× bf16 per 128-bit load** |
| 48 KiB smem cap (1650) | up to **164 KiB opt-in** (A100) |
| Hidden 1024 benches | **3584** (Qwen2.5-7B), divisible by 8 ✓ |

Phase 4 occupancy numbers **do not transfer** without re-probe on target hidden + dtype.

### 3. CUDA graph capture contract

Custom op must:

- **Not** allocate in forward
- **Not** synchronize
- **Not** read device properties at call time
- Accept **static shapes** under graph replay

Violations → silent eager fallback or broken capture; A/B would lie.

---

## Development gates

| Gate | Where | Pass criterion |
|------|-------|----------------|
| G0 | 1650 (local, $0) | `add_rmsnorm_fused.exe` vs vLLM native golden @ 3584 bf16 |
| G1 | **Rented A100** | `pip install -e .` → `kernelfuse.fused_add_rms_norm` op smoke @ 3584 |
| G2 | **Rented A100** | vLLM 0.8.5 call-site swap; server starts; one generate |
| G3 | **Rented A100** | CUDA graph capture smoke with op in path |
| G4 | **Same session** | Preamble + baseline vs treatment A/B |

**Do not rent** until G0 is green (already done). **One A100 evening** can cover G1–G4 if you batch integration debug + measurement.

Tradeoff vs T4: you pay during extension iteration, but you get bf16, Phase 7 pins, and the final A/B box in one place — no porting surprises.

---

## A100 measurement block (one session)

Run once after integration gates. Order matters.

### Preamble (~30–45 min, piggyback Phase 7 leftovers)

| Cell | Purpose |
|------|---------|
| Greedy output-equivalence | same prompt/seed, diff text across vLLM baseline vs op path (or cross-backend if time) |
| One smoke × 3 backends repeated | between-run variance envelope on `in=128 out=128 c=1` |
| Cache **off** then **on**, `in=2048 out=128 c=1` | publish cold/warm TTFT pair (~109 ms vs ~21 ms class); log `prefix_cache_hit_pct` |
| Optional | NGC TRT peer with `max_model_len=4096` if time — still not Phase 8 headline |

Harness: `flush_cache_between_cells` **on** by default for pre-registered matrices; deliberate cache-on cells opt out via `--no-flush-cache-between-cells` on that run only.

### Integration A/B (~30 min)

Same pins as Phase 7 vLLM: 0.8.5, torch 2.6.0+cu124, FA, bf16, graphs on, mem 0.90.

| Run | Config |
|-----|--------|
| Baseline | stock vLLM RMSNorm path |
| Treatment | custom `add_rmsnorm` op at decoder call site |

Cells (minimal):

- identity smoke `in=128 out=128 c=1`
- `in=128 out=128 c=1` (batch-1 TPOT — **null lives here**)
- `in=128 out=512 c=64` (peak sys tok/s — **null lives here**)

Record: `run_metadata.json`, kernel microbench CSV, graph-capture pass/fail, `prompt_tokens`, `prefix_cache_hit_pct`.

---

## Identity check (same as Phase 7)

`TTFT_p50 + (out−1) × TPOT_p50 ≈ e2e_p50` within ~1 ms on smoke. Failure → fix harness/op before interpreting A/B.

---

## Exit gate

1. **Custom op in vLLM** serving Qwen2.5-7B on A100 with CUDA graphs captured.
2. **Correctness** documented (add+RMSNorm path, bf16, hidden 3584).
3. **Null confirmed:** e2e TPOT / sys tok/s unchanged within noise; **kernel isolation** shows speedup.
4. **Phase 8 report** states why end-to-end didn't move (bandwidth bound) — same discipline as Phase 7 retractions.
5. SGLang integration opened as Phase 8b / Phase 9 item, not blocking gate.

---

## Cost napkin

| Block | Hardware | Wall |
|-------|----------|------|
| add_rmsnorm + bf16 vec8 + op registration | 1650 / T4 | days, $0 |
| vLLM TinyLlama graph smoke | T4 Colab | hours, $0 |
| A100 preamble + A/B | 80GB box | **~1–2 h** meter |

One rental, not two.
