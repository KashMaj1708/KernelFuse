#!/usr/bin/env bash
# Mid-regime hump + all fused_add_rms_norm vLLM instantiations (one-symbol each).
set -euo pipefail
cd /workspace/KernelFuse
source .venv-vllm/bin/activate
source scripts/phase8/env_torch_lib.sh
export PYTHONPATH=/workspace/KernelFuse
OUT=/workspace/kernelfuse_results/phase8
mkdir -p "$OUT" /workspace/kernelfuse_results/phase9

echo "=== mid-regime rows sweep ==="
python scripts/phase8/kernel_microbench.py --graph \
  --rows 128 256 512 1024 2048 \
  --out "$OUT/kernel_rows_sweep_mid.csv" | tee "$OUT/kernel_rows_sweep_mid.log"

SO=$(python -c 'import glob,logging,os,vllm; logging.disable(logging.CRITICAL); print(sorted(glob.glob(os.path.dirname(vllm.__file__)+"/_C*.so"))[0])' 2>/dev/null | grep '\.so$' | tail -1)
echo "VLLM_SO=$SO"
printf '%s\n' "$SO" > "$OUT/vllm_so_path.txt"

echo "=== all fused_add_rms_norm symbols ==="
cuobjdump -symbols "$SO" 2>/dev/null | grep -i fused_add_rms | tee "$OUT/sass_vllm_all_rms_syms.txt" || true

# Also locate dispatch in installed vLLM sources
python - <<'PY'
import pathlib, vllm
root = pathlib.Path(vllm.__file__).parent
hits = []
for p in root.rglob("*"):
    if p.suffix not in {".py", ".cu", ".cuh", ".h", ".hpp"}:
        continue
    try:
        t = p.read_text(errors="ignore")
    except Exception:
        continue
    if "fused_add_rms_norm" in t and ("vector" in t.lower() or "VEC" in t or "align" in t.lower()):
        hits.append(str(p))
print("source hits:")
for h in hits[:20]:
    print(" ", h)
PY

python - <<'PY'
import pathlib, re, subprocess
from collections import Counter

out = pathlib.Path("/workspace/kernelfuse_results/phase8")
so = (out / "vllm_so_path.txt").read_text().strip()
syms_raw = subprocess.check_output(["cuobjdump", "-symbols", so], stderr=subprocess.DEVNULL, text=True, errors="replace")
names = sorted({line.split()[-1] for line in syms_raw.splitlines() if "fused_add_rms_norm" in line and "STT_FUNC" in line})
print(f"STT_FUNC fused_add_rms_norm count={len(names)}")
(out / "sass_vllm_fused_add_names.txt").write_text("\n".join(names) + "\n")
summary = []
for n in names:
    try:
        sass = subprocess.check_output(
            ["cuobjdump", "-fun", n, "-sass", so],
            stderr=subprocess.DEVNULL, text=True, errors="replace", timeout=90,
        )
    except Exception as e:
        print(f"skip {n}: {e}")
        continue
    widths = Counter()
    for line in sass.splitlines():
        m = re.search(r"\b((?:LDG|STG)(?:\.[A-Z0-9]+)+)", line)
        if m:
            widths[m.group(1)] += 1
    has128 = any("128" in k for k in widths)
    tag = "VEC128" if has128 else ("SCALAR16" if any("U16" in k for k in widths) else "OTHER")
    safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", n)[:100]
    (out / f"sass_vllm_{tag}_{safe}.txt").write_text(sass)
    top = ", ".join(f"{k}:{v}" for k, v in widths.most_common(8))
    print(f"[{tag}] {n}\n  {top}")
    summary.append(f"{tag}\t{n}\t{top}")
(out / "sass_vllm_dispatch_summary.txt").write_text("\n".join(summary) + "\n")
print("DONE symbol dumps")
PY

echo "=== phase9 fp32 metrics ==="
python scripts/phase9/decode_attn_correctness.py \
  --out /workspace/kernelfuse_results/phase9/decode_attn_correctness_f32.csv

echo ALL_DONE
