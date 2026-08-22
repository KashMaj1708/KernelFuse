#!/usr/bin/env bash
set -euo pipefail
cd /workspace/KernelFuse
source .venv-vllm/bin/activate
source scripts/phase8/env_torch_lib.sh
OUT=/workspace/kernelfuse_results/phase8
mkdir -p "$OUT"

python - <<'PY'
import glob, os, sys, vllm
sys.stderr = open(os.devnull, "w")
d = os.path.dirname(vllm.__file__)
cands = sorted(glob.glob(os.path.join(d, "_C*.so")))
assert cands, d
open("/workspace/kernelfuse_results/phase8/vllm_so_path.txt", "w").write(cands[0] + "\n")
print(cands[0])
PY

VLLM_SO=$(cat "$OUT/vllm_so_path.txt")
echo "VLLM_SO=$VLLM_SO"

cuobjdump -symbols "$VLLM_SO" 2>/dev/null | grep -iE 'rms|norm|fused' | tee "$OUT/sass_vllm_symbols.txt" || true

python - <<'PY'
import pathlib, re, subprocess
from collections import Counter

out = pathlib.Path("/workspace/kernelfuse_results/phase8")
so = (out / "vllm_so_path.txt").read_text().strip()
print("dumping", so)
raw = subprocess.check_output(
    ["cuobjdump", "-sass", so], stderr=subprocess.DEVNULL, text=True, errors="replace"
)
(out / "sass_vllm_full.txt").write_text(raw)
parts = re.split(r"(?=^\s*Function\s*:)", raw, flags=re.M)
needles = [r"fused_add_rms", r"rms_norm", r"RMSNorm", r"AddRms", r"rmsNorm"]
hits = [p for p in parts if any(re.search(n, p[:2000], re.I) for n in needles)]
print(f"functions={len(parts)} hits={len(hits)}")
comb = Counter()
for h in hits[:30]:
    name_m = re.search(r"Function\s*:\s*(\S+)", h)
    n = name_m.group(1) if name_m else "unknown"
    safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", n)[:140]
    (out / f"sass_vllm_{safe}.txt").write_text(h)
    widths = Counter()
    for line in h.splitlines():
        m = re.search(r"\b((?:LDG|STG)(?:\.[A-Z0-9]+)+)", line)
        if m:
            widths[m.group(1)] += 1
            comb[m.group(1)] += 1
    shfl = len(re.findall(r"\bSHFL\b", h))
    bar = len(re.findall(r"\bBAR\.SYNC\b", h))
    print(f"{safe}\n  SHFL={shfl} BAR.SYNC={bar}")
    for k, v in widths.most_common(15):
        print(f"  {v:5d}  {k}")
print("--- combined ---")
for k, v in comb.most_common(20):
    print(f"{v:5d}  {k}")
# If no hits, show any function with LDG.E.128 near 'norm' in full symbol list via strings
if not hits:
    print("NO SYMBOL HITS — searching body for rms_norm strings")
    for p in parts:
        if "rms" in p[:800].lower() or "RMS" in p[:800]:
            print(p[:200].replace("\n", " "))
PY
