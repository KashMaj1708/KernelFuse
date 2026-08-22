# KernelFuse — Phased Build Plan

An inference-serving benchmark (vLLM / SGLang / TensorRT-LLM) plus a hand-written fused CUDA kernel, built so that every phase is gated on the previous one succeeding, and paid GPU time is pushed as late as possible. Develop on the GTX 1650 (or Colab/Kaggle free tier) for as long as the work allows; rent a 24 GB-class card only for the runs that genuinely need it.

**Hardware ladder used throughout**
- **Tier A — GTX 1650 (4 GB, local):** correctness, CPU/GPU parity, kernel dev and debug, harness dev against a tiny model.
- **Tier B — free cloud (Colab / Kaggle, ~16 GB):** anything that needs more than 4 GB but not a rented card; also a fallback Nsight profiling surface if the 1650 counters misbehave.
- **Tier C — rented 24 GB+ (A100 / 3090 / A10, by the hour):** final benchmark numbers and the clean Nsight Compute profiling pass. Only finished code runs here.

**Guiding rule:** nothing runs on Tier C until it has already run correctly on Tier A or B. Rented time is for measurement, never development.

---

## Phase 0 — Environment setup and sanity check
*Hardware: Tier A. Goal: prove the toolchain works before writing anything real.*

- Confirm the CUDA toolkit and driver are visible: `nvcc --version` and `nvidia-smi` both return sane output on the 1650.
- Compile and run a trivial "hello from thread" CUDA kernel end to end. This is the single check that the compile-launch-retrieve loop works on your box.
- Confirm the profiler runs at all: launch Nsight Compute (`ncu`) on the trivial kernel and confirm it produces a report, even a thin one. Note here whether the 1650 gives you full memory/occupancy counters or restricts them — this decides later whether final profiling must move to Tier B/C.
- Stand up a clean Python env (venv or conda) with a CPU-only PyTorch to start. Confirm `import torch` works and a tensor op runs on CPU.
- Create the repo (`KernelFuse`), add a README stub stating the two-layer goal, and commit the environment notes (versions, GPU, driver) so results are reproducible later.

**Exit gate:** trivial kernel compiles, launches, and profiles; Python env imports cleanly. If Nsight counters are restricted on the 1650, record that now — it is not a blocker, it just pins final profiling to Tier B/C.

---

## Phase 1 — Kernel correctness (CPU reference first)
*Hardware: Tier A. Goal: a correct RMSNorm, proven against a trusted reference, before any performance work.*

- Write a plain NumPy/PyTorch-CPU RMSNorm as the golden reference. This is the source of truth for correctness.
- Write the naive (unfused, un-optimized) CUDA RMSNorm kernel. Correctness only — one thread per element, no cleverness.
- Verify the CUDA output matches the CPU reference within floating-point tolerance across several input shapes (small, odd, and power-of-two sizes).
- Add a tiny test harness that fails loudly on mismatch. Commit it. Every later kernel version must pass this same test.

**Exit gate:** naive CUDA RMSNorm matches the CPU reference on all test shapes. Do not proceed to fusion or tuning until this is green.

---

## Phase 2 — The fused kernel (still correctness-gated)
*Hardware: Tier A. Goal: the actual fused kernel, still validated only for correctness.*

- Implement the fused RMSNorm: combine the normalization and the scale (and any elementwise follow-on you choose to fold in) into a single kernel launch, so intermediate results never round-trip to global memory.
- Run it against the Phase 1 test harness. It must match the CPU reference exactly as the naive version did.
- Keep both kernels (naive and fused) in the repo. The comparison between them is part of the story later.

**Exit gate:** fused kernel passes the same correctness tests as the naive one. Correctness is now locked; everything after this is performance and measurement.

---

## Phase 3 — Local performance signal (relative, not final)
*Hardware: Tier A. Goal: confirm the fused kernel is worth profiling, using cheap relative timing.*

- Add simple CUDA-event timing around each kernel. Measure naive vs fused vs the PyTorch built-in on the 1650, at input sizes that fit in 4 GB.
- You are looking for a *direction*, not a headline number: does fusion reduce time, and does it behave sanely as size grows? Small-card numbers are not reportable, but they tell you the kernel is doing what you think.
- If the fused version is not faster than naive at any size, stop and investigate here — cheaply, on local hardware — before spending anything on Tier C.

**Exit gate:** fused kernel shows a plausible relative speedup over naive on local timing. If it does not, debug now; do not rent a GPU to discover a logic problem.

---

## Phase 4 — Profiling and the "why" (the interview-critical phase)
*Hardware: Tier A if counters allow, else Tier B. Goal: the memory-bound analysis that makes the kernel mean something.*

- Profile fused / fused_smem / fused_vec4 with Nsight Compute. Capture: occupancy, memory throughput vs compute throughput, coalescing, warp stall reasons, and `__syncthreads` cost.
- **Primary chase — smem width 4096 vs 8192:** Napkin math (dynamic smem only) predicts ~**2×** (4 vs 2 blocks/SM). **Occupancy API on the 1650 (includes static `spartial`/`sinv`) shows 3 vs 1 blocks (768 vs 256 threads) = 3.0×.** Phase 3 measured BW **122.3 → 45.8 GB/s (~2.67×)**. So most of the “beyond 2×” gap is static shared eating a resident block — especially at 8192. Do **not** stop Nsight at “occupancy dropped”; still capture DRAM throughput and stall reasons for the residual vs 3.0×, and for vec4 vs scalar smem at the **same** 1-block occupancy.
- **Clocks on every run (mobile):** log `nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu` before and after each bench/profile. Nsight Compute replays kernels many times; if clocks sag, %peak shifts. Try `nvidia-smi -lgc` when admin is available — often refused on mobile.
- Tier A: `.\scripts\run_phase4_profile.ps1` (occupancy probe always; ncu if counters unlocked). Tier B: `notebooks/phase4_colab.ipynb` / `scripts/run_phase4_profile.sh` (prefer T4 = sm_75).
- Write the analysis: RMSNorm at inference sizes is memory-bandwidth-bound; fusion/staging cuts global round-trips. **Motivating frame:** torch 2.5.1 `F.rms_norm` composite = naive multi-pass tax in the wild; naive-vs-fused is the controlled version.
- Commit the profiling report and the written explanation into the repo (local `docs/phase_4_report.md` is gitignored — promote a short `docs/` summary when ready to publish).

**Exit gate:** you can state, with your own numbers, why the kernel is memory-bound, where fusion wins, and what closes napkin 2× vs measured ~2.7× (static smem → API 3×, plus any residual from Nsight stalls). Full stall/DRAM table may require Tier B if Tier A hits `ERR_NVGPUCTRPERM`.

---

## Phase 5 — Benchmark harness (built against a tiny model, locally)
*Hardware: Tier A. Goal: the full benchmarking harness, debugged on something that fits in 4 GB.*

- Build the benchmarking harness: config-matrix driver (batch sizes, sequence lengths, concurrency), latency-percentile collection (p50/p90/p99, not means), and throughput measurement.
- Debug the entire harness against the smallest possible model — a tiny/toy LLM or a heavily reduced config — so it fits on the 1650. The point is to prove the harness logic works, not to measure anything real.
- Pre-register the config matrix (write the sweep down before running it) — this mirrors your GordianBench methodology and is a named responsibility on the target roles.
- Make the harness backend-agnostic in structure now, even though only one backend will be reachable locally.

**Exit gate:** harness runs end to end against a toy model on the 1650, emitting a clean results table with percentiles. All logic is proven; only scale is missing.

---

## Phase 6 — Single-backend dry run on free cloud
*Hardware: Tier B. Goal: prove a real backend + real (small) model works before paying.*

- On Colab/Kaggle free tier, stand up one backend (vLLM is the easiest first) serving a small real model (e.g. a 1–3 B), and run the harness against it.
- This is the first time real serving software meets a real model in your pipeline. Shake out install issues, API mismatches, and OOM behavior here where it costs nothing.
- Confirm the harness's numbers look sane against a real backend, not just the toy.

**Exit gate:** one real backend serves a real model and the harness produces a coherent result set on free hardware. You now know the paid run will work before you start the meter.

---

## Phase 7 — Full multi-backend benchmark (paid, the meter is running)
*Hardware: Tier C. Goal: the actual reportable numbers, collected fast because everything is already debugged.*

- Rent a 24 GB+ card. Stand up all three backends (vLLM, SGLang, TensorRT-LLM) serving your chosen model(s) — one 7 B-class model is enough; add a larger/MoE only if the card and budget allow.
- Run the pre-registered sweep across all three backends. Collect throughput and latency percentiles across the config matrix.
- Also run the final clean Nsight Compute pass on the fused kernel here if you want headline profiling numbers on datacenter-class hardware (optional if Phase 4 on Tier B was already clean).
- **Torch RMSNorm honesty check:** on the rented card with *current* torch, profile `F.rms_norm` and count kernels per call. If it dispatches a fused CUDA kernel, that is the honest head-to-head and the number worth reporting. If it still decomposes, that is a finding in its own right (same multi-pass tax as Phase 3's 2.5.1 composite).
- Tear the instance down the moment the runs finish. You are paying only for finished code to execute.

**Exit gate:** complete, reportable benchmark tables across three backends, plus the kernel's final numbers (and an explicit fused-vs-composite torch status on that stack). Rented time stays in the low single-digit hours because no debugging happens here.

---

## Phase 8 — vLLM integration (custom add+RMSNorm op)
*Hardware: Tier A/B to develop; Tier C once for A/B. Goal: kernel inside the serving stack, with a pre-registered end-to-end null.*

- Write **add+RMSNorm** (not norm-only): vLLM fuses residual add + norm; splitting them measures a false regression.
- Port to **bf16 in/out, fp32 accumulate, vec8 loads**; re-probe smem/occupancy at hidden **3584** (Qwen2.5-7B).
- Package as torch custom op; register in **vLLM 0.8.5** first (**SGLang follow-up**, not co-primary).
- Dev on 1650 + Colab T4 until correctness and **CUDA graph capture** pass (no alloc/sync in forward).
- Pre-register null: e2e TPOT will not move (bandwidth-bound decode); deliverable is integration + isolated kernel speedup.
- One A100 session: Phase 7 preamble cells + baseline vs treatment A/B (see `docs/phase_8_predictions.md`).

**Exit gate:** op serving Qwen2.5-7B on A100 with graphs on; null confirmed; report explains why e2e didn't move.

---

## Phase 8b — SGLang integration (follow-up)
*Closed on A100.* Same kernelfuse op; rebuild extension against SGLang torch ABI. Null confirmed.

---

## Optional Phase 9 — Attention kernel (ceiling, not floor)
*Closed as stretch.* Naive decode-attn CUDA microbench on A100; slower than torch_ref — no serving claim.

---

### Cost summary
- Phases 0–5: **$0** (GTX 1650, local).
- Phase 6: **$0** (free cloud tier).
- Phase 7: **low single-digit hours of rented GPU** (~$10–30 depending on card and model size).
- Everything expensive is gated behind everything cheap having already passed.
