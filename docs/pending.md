## Methodology cleanup (post 8 / 8b / 9)

| Item | Status |
|------|--------|
| Phase 7 cold-anchor overturn | **done** — `docs/phase_7_errata.md` |
| Flush fail-closed (`ok=False`) | **done** — `bench/cache_obs.py` + runner |
| CUDA-event microbench | **done** — `scripts/phase8/kernel_microbench.py` (re-run on GPU when convenient) |
| Phase 8b regression rewrite | **done** — report |
| Phase 9 correctness CSV | **done** — `scripts/phase9/decode_attn_correctness.py` |

## Phase 8 — vLLM integration

**Closed** (serving null + integration). Kernel isolation claim retracted pending event/graph/ncu.

## Phase 8b — SGLang integration

**Closed** as **~1.4% consistent regression**, not null. CUDA version-check bypass caveat on all treatment numbers.

## Phase 9 — decode attention (ceiling)

**Closed as stretch.** Correctness script added; timing interpretation corrected (torch_ref overhead floor).

## Phase 7

**Closed** with errata: no definitive cold prefill measured.
