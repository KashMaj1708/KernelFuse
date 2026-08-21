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

- Profile the fused kernel with Nsight Compute. Capture: occupancy, memory throughput vs compute throughput, and whether global-memory access is coalesced.
- Write the analysis: RMSNorm at inference sizes is memory-bandwidth-bound, not compute-bound, which is *why* fusion (fewer global-memory round-trips) helps and why simply adding arithmetic would not. Back this with your own captured numbers.
- If the 1650's counters were restricted (flagged in Phase 0), run this pass on a free Colab/Kaggle GPU instead — still no paid time.
- Commit the profiling report and the written explanation into the repo.

**Exit gate:** you can state, with your own profiler numbers behind it, why the kernel is memory-bound and where fusion wins. This is the artifact the top roles actually read.

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
- Tear the instance down the moment the runs finish. You are paying only for finished code to execute.

**Exit gate:** complete, reportable benchmark tables across three backends, plus the kernel's final numbers. Rented time stays in the low single-digit hours because no debugging happens here.

---

## Phase 8 — Analysis, write-up, and the contribution on-ramp
*Hardware: none. Goal: turn runs into the artifact, and surface the open-source PR.*

- Write the results analysis: where each backend wins and why (batching strategy, KV-cache handling, the prefill-vs-decode split), stated in mechanism terms, not just a number table.
- Fold the kernel story in: the benchmark located the hot path, the kernel drilled into it. Present naive vs fused vs built-in with the memory-bound reasoning.
- Be explicit about scope: single-node, model sizes named, honest about where your kernel loses to library code and why.
- Capture anything you found broken or undocumented in a backend during Phases 6–7 (a wrong throughput calc, a missing config path, a metrics bug). That is the seed of an upstream PR — file the issue / open the fix in *their* repo. The benchmark is the on-ramp; the merged change is the separate contribution.
- Finalize the README as a proper project front page: the two-layer story, the methodology, the headline findings, reproduction steps.

**Exit gate:** repo reads as one coherent inference-performance body of work; resume bullet writes itself; at least one concrete upstream-PR candidate identified.

---

## Optional Phase 9 — Attention kernel (ceiling, not floor)
*Hardware: Tier A to develop, Tier B/C to profile. Goal: upgrade the kernel story, only after everything above is done.*

- Attempt a fused attention kernel (decode-phase). Same discipline: CPU reference → correctness → fusion → profiling.
- This is unbounded in difficulty; treat it strictly as an upgrade. RMSNorm already carries the kernel signal, so nothing above depends on this landing.
- If it works even naively-but-correctly-and-profiled, it is a strong addition. If it stalls, drop it without cost.

**Exit gate:** none required — this phase is pure upside.

---

### Cost summary
- Phases 0–5: **$0** (GTX 1650, local).
- Phase 6: **$0** (free cloud tier).
- Phase 7: **low single-digit hours of rented GPU** (~$10–30 depending on card and model size).
- Everything expensive is gated behind everything cheap having already passed.
