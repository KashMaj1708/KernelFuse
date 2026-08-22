#!/usr/bin/env bash
# Static SASS compare: kernelfuse vs vLLM fused_add_rms_norm (no ncu needed).
set -euo pipefail
cd /workspace/KernelFuse
source .venv-vllm/bin/activate
source scripts/phase8/env_torch_lib.sh
export PYTHONPATH=/workspace/KernelFuse
OUT=/workspace/kernelfuse_results/phase8
mkdir -p "$OUT"

KF_SO=$(python - <<'PY'
import glob, kernelfuse, os
d = os.path.dirname(kernelfuse.__file__)
cands = glob.glob(os.path.join(d, "_C*.so"))
assert cands, d
print(cands[0])
PY
)
VLLM_SO=$(python - <<'PY'
import glob, os, vllm
d = os.path.dirname(vllm.__file__)
cands = sorted(glob.glob(os.path.join(d, "_C*.so")))
# Fallback: some builds ship as vllm/_C*.abi3.so
if not cands:
    cands = sorted(glob.glob(os.path.join(d, "**", "_C*.so"), recursive=True))
assert cands, d
print(cands[0])
PY
)

echo "KF_SO=$KF_SO"
echo "VLLM_SO=$VLLM_SO"
printf '%s\n' "$KF_SO" > "$OUT/kf_so_path.txt"
printf '%s\n' "$VLLM_SO" > "$OUT/vllm_so_path.txt"

echo "=== kernelfuse LDG/STG width histogram ==="
cuobjdump -sass "$KF_SO" 2>/dev/null | grep -oE '\b(LDG|STG)(\.[A-Z0-9]+)+' | sort | uniq -c | sort -rn | tee "$OUT/sass_kf_ldg_stg.txt"

echo "=== vLLM symbols (rms / fused_add) ==="
cuobjdump -symbols "$VLLM_SO" 2>/dev/null | grep -iE 'rms|fused_add|RMS' | tee "$OUT/sass_vllm_symbols.txt" || true

echo "=== kernelfuse symbols ==="
cuobjdump -symbols "$KF_SO" 2>/dev/null | tee "$OUT/sass_kf_symbols.txt" || true

python - <<'PY'
import pathlib, re, subprocess
from collections import Counter

out = pathlib.Path("/workspace/kernelfuse_results/phase8")
kf_so = (out / "kf_so_path.txt").read_text().strip()
vllm_so = (out / "vllm_so_path.txt").read_text().strip()


def dump_and_split(so: str, label: str, needle: str):
    raw = subprocess.check_output(
        ["cuobjdump", "-sass", so], stderr=subprocess.DEVNULL, text=True, errors="replace"
    )
    (out / f"sass_{label}_full.txt").write_text(raw)
    parts = re.split(r"(?=^\s*Function\s*:)", raw, flags=re.M)
    hits = [p for p in parts if re.search(needle, p[:1200], re.I)]
    print(f"{label}: {len(parts)} functions, {len(hits)} matching /{needle}/")
    summary = []
    for h in hits[:20]:
        name_m = re.search(r"Function\s*:\s*(\S+)", h)
        n = name_m.group(1) if name_m else "unknown"
        safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", n)[:140]
        path = out / f"sass_{label}_{safe}.txt"
        path.write_text(h)
        widths = Counter()
        for line in h.splitlines():
            m = re.search(r"\b((?:LDG|STG)(?:\.[A-Z0-9]+)+)", line)
            if m:
                widths[m.group(1)] += 1
        shfl = len(re.findall(r"\bSHFL\b", h))
        bar = len(re.findall(r"\bBAR\.SYNC\b", h))
        summary.append((safe, shfl, bar, widths))
        print(f"  {safe}")
        print(f"    SHFL={shfl} BAR.SYNC={bar}")
        for k, v in widths.most_common(20):
            print(f"    {v:5d}  {k}")
    # Also whole-binary LDG/STG for vLLM filtered to rms function files only
    return summary


dump_and_split(kf_so, "kf", r"fused_add_rms|rms_norm|kernelfuse")
dump_and_split(vllm_so, "vllm", r"fused_add_rms|rms_norm|RMSNorm|rms_norm_kernel")

# Four-array bandwidth from measured times (rows=16384, hidden=3584, bf16)
bytes4 = 4 * 16384 * 3584 * 2
print(f"four_array_bytes={bytes4} ({bytes4/1e6:.1f} MB)")
for name, us in [("kf", 547.0), ("vllm", 284.0)]:
    tbs = (bytes4 / (us * 1e-6)) / 1e12
    print(f"  {name}: {us:.1f} us -> {tbs:.3f} TB/s under 4-array model")
print("DONE extract")
PY

echo DONE
