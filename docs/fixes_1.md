# fixes_1.md — Pre-Phase-3 fixes

Everything here lands before Phase 3 timing begins. Ordered so each step is verifiable on its own and nothing later depends on an unverified earlier step. Re-run the correctness harness after every step; it stays the gate throughout.

Steps 1 to 4 are correctness and test-coverage work. Steps 5 to 7 are kernel variants. Step 8 is the final regression sweep.

---

## Step 1 — Confirm `weight` is non-trivial in the harness

**Why:** if `weight` was all ones during Phase 1 and 2, any bug in the scale path is invisible and has been invisible for two phases.

**Do:**
- Open `tests/test_rmsnorm_correctness.py` and check how `weight` is generated.
- If it is ones (or not generated at all), switch it to a random vector per shape, seeded for reproducibility. Use values away from 1.0 so a dropped multiply is obvious, e.g. uniform in [0.5, 2.0].
- Make sure the CPU golden reference applies the same `weight` vector.

**Verify:** re-run `--backend both`. Still 20/20. If something now fails, you have found a real bug that the ones-weight was hiding.

---

## Step 2 — Add value-space edge cases

**Why:** Phase 1 and 2 covered shape space only. All inputs were well-conditioned random data. Reductions break on the value extremes, not the shape extremes.

**Do:** add these cases to the harness, each at a couple of widths (e.g. 128 and 1024):
- An all-zeros row. `mean(x^2)` is 0, so `inv_rms = rsqrt(eps)`. Well-defined, easy to mishandle.
- A row of very large magnitudes (e.g. 1e18) where `x^2` approaches float32 range.
- A row of very small magnitudes (e.g. 1e-20) where `eps` dominates the sum.
- Mixed signs, including a row that sums to near zero before squaring.
- A single-nonzero row (one large value, rest zeros).

**Verify:** re-run both backends. Expect passes. If the large-magnitude case fails, note it: that is an fp32 range limit, not necessarily a kernel bug, and it informs the accumulator decision in Step 4.

---

## Step 3 — Extend the shape list to realistic inference sizes

**Why:** current max width is 1024 and max row count is 64. Real RMSNorm runs at hidden dims 3584 to 8192 with thousands of rows. Phase 3 timing at current sizes measures launch overhead, not bandwidth. Large widths are also where the fused design's shared-memory sizing actually gets exercised.

**Do:** add to the shape list, keeping all 20 existing shapes:
- Realistic widths at small row counts (cheap correctness checks): `(4, 3584)`, `(4, 4096)`, `(4, 8192)`
- Non-multiple-of-4 and non-power-of-2 widths at realistic scale, to protect the vectorized path later: `(4, 4095)`, `(4, 4097)`, `(4, 3584)`
- Large row counts for timing later: `(4096, 4096)`, `(2048, 8192)`
- One awkward large case: `(1000, 3584)`

**Memory check:** `(4096, 4096)` fp32 is 64 MB per tensor. With input, output, and weight you are well under 4 GB. `(2048, 8192)` is also 64 MB. Both fit.

**Verify:** re-run both backends across the full extended list. The naive kernel may get slow at the large shapes since its reduce is one thread per row; that is expected and is correctness-only for now. If the fused kernel fails at 8192, that is a shared-memory or block-sizing limit and needs fixing before Step 5.

---

## Step 4 — Switch the accumulator from `double` to `float`

**Why:** consumer Turing runs FP64 at 1/32 of FP32 and a 3090 at 1/64. On the 1650 that puts FP64 throughput close enough to memory throughput that the memory-bound conclusion in Phase 4 gets murky. Production RMSNorm kernels (PyTorch, vLLM) accumulate in fp32, so benchmarking a double version against them measures a difference you introduced rather than one that matters.

**Do:**
- In `rmsnorm_fused.cu`, change the per-thread accumulator and the shared-memory tree reduce from `double` to `float`. Use `fmaf` for the accumulation.
- The shared-memory reduce buffer halves in size as a side effect. Keep that in mind for Step 5's budget.
- Keep the double path behind a compile-time flag (e.g. `-DACC_DOUBLE`) if you want it as a numerical reference. Do not time it.

**Verify:** re-run both backends on the full extended shape list. Expect passes, possibly with slightly larger deviation at width 8192 since fp32 accumulation over 8192 terms carries more error than double did. If the existing `rtol=1e-4` starts failing at the widest shapes, that is expected numerics, not a bug: widen tolerance for those shapes specifically and document why, rather than widening globally.

**Note:** your reference is NumPy, which uses pairwise summation in `mean`. Your shared-memory tree reduce is also effectively pairwise, so the two should agree well. The naive kernel's serial per-row loop is the least accurate of the three and is the one most likely to drift at large widths.

---

## Step 5 — Add the shared-memory staged variant

**Why:** this is the actual design fix. The current fused kernel still reads `x` twice from global memory, so the only traffic it saves is the `inv_rms` round-trip, which is one float per row against N floats per row. That is roughly a 0.1% reduction at width 1024. The fusion needs to stage the row on-chip so the second access is a shared-memory read, not a global one.

**Do:** create `kernels/rmsnorm/rmsnorm_fused_smem.cu` as a new variant. Do not modify the Phase 2 kernel; it stays as a comparison point.

- Declare dynamic shared memory sized `N * sizeof(float)` at launch.
- Pass 1: each thread strides over columns, loads `x` once, writes the value into shared memory, and accumulates the square into its register accumulator.
- Tree reduce to `inv_rms` as before.
- Pass 2: each thread strides over columns again, reading from shared memory instead of global, and writes `out = srow[c] * inv_rms * weight[c]`.

**Width ceiling:** at 4096 the row is 16 KB, at 8192 it is 32 KB. sm_75 allows up to 64 KB shared per SM with a per-block cap that requires opt-in above 48 KB via `cudaFuncSetAttribute` with `cudaFuncAttributeMaxDynamicSharedMemorySize`. Add that opt-in call. Above roughly 16384 elements this variant cannot work; add an explicit guard that falls back to the Phase 2 kernel rather than silently producing wrong results.

**Occupancy note:** 16 KB per block means about 3 resident blocks per SM; 32 KB means about 2. This is a genuine tradeoff, not a free win, and quantifying it is a Phase 4 result.

**Verify:** the new variant passes the full extended shape list, including the guard path at widths that exceed the shared-memory budget.

---

## Step 6 — Add the vectorized (`float4`) variant

**Why:** on a bandwidth-bound kernel, moving 16 bytes per load instruction instead of 4 usually matters more than the second-read question. This is likely your largest single bandwidth win and it gives Phase 4 a second axis to analyze.

**Do:** create `kernels/rmsnorm/rmsnorm_fused_smem_vec4.cu`, built on the Step 5 variant.

- Reinterpret the row pointers as `float4` for the bulk of the row.
- Handle the tail (`N % 4`) with a scalar loop.
- Guard on pointer alignment: if the base pointers are not 16-byte aligned, fall back to the scalar path. `cudaMalloc` returns suitably aligned memory, but row offsets `row * N` are only aligned when `N % 4 == 0`, so the guard is a real requirement, not a formality.
- Apply the same treatment to the store path, not just the loads.

**Verify:** passes the full extended shape list, specifically including `(4, 4095)` and `(4, 4097)` from Step 3, which are the cases that exercise the tail and the misalignment fallback.

---

## Step 7 — Register the variants in the harness

**Why:** Phase 3 needs to time four implementations against each other, and each must be selectable and independently verified.

**Do:**
- Extend `--backend` to accept `naive | fused | fused_smem | fused_vec4 | all`.
- Update `scripts/run_rmsnorm_tests.ps1` to build all four binaries and run `--backend all`.
- Keep the `.bin` I/O contract identical across all four so Phase 3 can swap implementations without changing the driver.

**Verify:** `--backend all` runs green across the full extended shape list.

---

## Step 8 — Full regression sweep and commit

**Do:**
- Run `--backend all` across every shape, including the new realistic and edge-value cases, with the randomized `weight`.
- Record which shapes, if any, needed widened tolerance after the fp32 accumulator change, and why.
- Update `docs/ENVIRONMENT.md` if anything about the build changed (new flags, the shared-memory opt-in call).
- Commit all four kernels, the extended harness, and the notes as the pre-Phase-3 baseline.

**Exit gate for fixes_1:** four kernel variants, all passing the same extended harness with non-trivial weight and edge-value coverage, fp32 accumulation throughout, and realistic shapes available for timing. Phase 3 can then measure naive vs fused vs fused_smem vs fused_vec4 and have a genuine experiment rather than a single before-and-after.

---

## Deferred, not forgotten

Not required before Phase 3, but track these:

- **fp16 / bf16 path.** Real serving runs half precision, which halves the bytes moved and is central to the memory-bound argument. Either add it in Phase 4 or state explicitly in the writeup that the analysis is fp32 and reason about how it scales.
- **Register-staged variant.** The alternative to Step 5, holding the row in registers via a templated per-thread count with full unrolling. Faster when it fits, but spills to local memory if indexing is not compile-time resolvable. Worth building only if Phase 4 profiling shows the shared-memory path is the constraint.
- **L2 residency expectation.** At small widths the staged and re-read variants should converge, because the row stays in L2 and the second read is nearly free. The gap should open as the working set outgrows L2. Showing that divergence is a stronger Phase 4 result than a flat speedup number, so keep both variants and sweep width deliberately.
