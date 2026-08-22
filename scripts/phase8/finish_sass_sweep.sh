#!/usr/bin/env bash
# Finish after uint4 fix: targeted vLLM SASS + sweep + phase9 (no full _C dump).
set -euo pipefail
cd /workspace/KernelFuse
source .venv-vllm/bin/activate
source scripts/phase8/env_torch_lib.sh
export PYTHONPATH=/workspace/KernelFuse
OUT=/workspace/kernelfuse_results/phase8
mkdir -p "$OUT" /workspace/kernelfuse_results/phase9

VLLM_SO=$(python - <<'PY'
import glob, os, sys, vllm
sys.stderr = open(os.devnull, "w")
d = os.path.dirname(vllm.__file__)
print(sorted(glob.glob(d + "/_C*.so"))[0])
PY
)
echo "VLLM_SO=$VLLM_SO"

# Prefer cuobjdump -fun on fused_add_rms_norm symbols only
SYMS=$(cuobjdump -symbols "$VLLM_SO" 2>/dev/null | grep -iE 'fused_add_rms|rms_norm_kernel' | awk '{print $NF}' | head -20 || true)
echo "candidate symbols:"
echo "$SYMS" | tee "$OUT/sass_vllm_rms_syms.txt"

# Dump only matching functions via -fun if available; else grep ELF with nvdisasm limited
python - <<'PY'
import pathlib, re, subprocess
from collections import Counter
out = pathlib.Path("/workspace/kernelfuse_results/phase8")
so = open("/dev/stdin").read().strip() if False else pathlib.Path("/workspace/kernelfuse_results/phase8/vllm_so_path.txt")
# write path
import glob, os, sys, vllm
sys.stderr = open(os.devnull, "w")
d = os.path.dirname(vllm.__file__)
so = sorted(glob.glob(d + "/_C*.so"))[0]
pathlib.Path("/workspace/kernelfuse_results/phase8/vllm_so_path.txt").write_text(so + "\n")

# Use cuobjdump -fun for each interesting mangled name
syms_txt = (out / "sass_vllm_rms_syms.txt").read_text()
syms = [s.strip() for s in syms_txt.splitlines() if s.strip() and not s.startswith("candidate")]
# Also pull from symbols file more carefully
raw_syms = subprocess.check_output(["cuobjdump", "-symbols", so], stderr=subprocess.DEVNULL, text=True, errors="replace")
names = []
for line in raw_syms.splitlines():
    if re.search(r"fused_add_rms|rms_norm_kernel", line, re.I) and "STT_FUNC" in line:
        names.append(line.split()[-1])
# Dedupe, prefer bf16 fused_add
pref = [n for n in names if "fused_add_rms" in n]
if not pref:
    pref = names[:8]
print(f"dumping {len(pref)} functions")
comb = Counter()
for n in pref[:12]:
    try:
        sass = subprocess.check_output(
            ["cuobjdump", "-fun", n, "-sass", so],
            stderr=subprocess.DEVNULL, text=True, errors="replace", timeout=120,
        )
    except Exception as e:
        print(f"  skip {n}: {e}")
        continue
    safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", n)[:120]
    (out / f"sass_vllm_{safe}.txt").write_text(sass)
    widths = Counter()
    for line in sass.splitlines():
        m = re.search(r"\b((?:LDG|STG)(?:\.[A-Z0-9]+)+)", line)
        if m:
            widths[m.group(1)] += 1
            comb[m.group(1)] += 1
    shfl = len(re.findall(r"\bSHFL\b", sass))
    bar = len(re.findall(r"\bBAR\.SYNC\b", sass))
    print(f"{safe}\n  SHFL={shfl} BAR.SYNC={bar}")
    for k, v in widths.most_common(10):
        print(f"  {v:5d}  {k}")
print("--- combined LDG/STG ---")
for k, v in comb.most_common(15):
    print(f"{v:5d}  {k}")
(out / "sass_vllm_ldg_stg_targeted.txt").write_text(
    "\n".join(f"{v:5d}  {k}" for k, v in comb.most_common())
)
print("vllm targeted sass done")
PY

echo "=== correctness smoke ==="
python - <<'PY'
import torch, kernelfuse
from vllm import _custom_ops as ops
torch.manual_seed(0)
for rows in [1, 512, 4096]:
    x0 = torch.randn(rows, 3584, device="cuda", dtype=torch.bfloat16)
    r0 = torch.randn(rows, 3584, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(3584, device="cuda", dtype=torch.bfloat16)
    xk, rk = x0.clone(), r0.clone(); xv, rv = x0.clone(), r0.clone()
    kernelfuse.fused_add_rms_norm(xk, rk, w, 1e-6)
    ops.fused_add_rms_norm(xv, rv, w, 1e-6)
    print(f"rows={rows} dx={(xk.float()-xv.float()).abs().max().item():.4e} dr={(rk.float()-rv.float()).abs().max().item():.4e}")
PY

echo "=== rows sweep ==="
python scripts/phase8/kernel_microbench.py --graph \
  --out "$OUT/kernel_rows_sweep_graph_v4.csv" | tee "$OUT/kernel_rows_sweep_graph_v4.log"

echo "=== phase9 fp32 ==="
python scripts/phase9/decode_attn_correctness.py \
  --out /workspace/kernelfuse_results/phase9/decode_attn_correctness_f32.csv

python - <<'PY'
import csv
from pathlib import Path
p = Path("/workspace/kernelfuse_results/phase8/kernel_rows_sweep_graph_v4.csv")
rows = list(csv.DictReader(p.open()))
r = next(x for x in rows if int(x["rows"]) == 16384)
bytes4 = 4 * 16384 * 3584 * 2
kf = float(r["kernelfuse_min_us"]); vl = float(r["vllm_min_us"])
print(f"rows=16384 kf={kf:.3f}us vllm={vl:.3f}us ratio={vl/kf:.3f}")
print(f"four_array_MB={bytes4/1e6:.1f}")
print(f"TB/s_kf={bytes4/(kf*1e-6)/1e12:.3f}  TB/s_vllm={bytes4/(vl*1e-6)/1e12:.3f}")
PY
echo ALL_DONE
