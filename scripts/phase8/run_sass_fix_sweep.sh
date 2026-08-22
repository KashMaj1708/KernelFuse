#!/usr/bin/env bash
# Rebuild uint4 kernel, verify SASS widths, re-sweep, Phase 9 fp32 correctness.
set -euo pipefail
cd /workspace/KernelFuse
source .venv-vllm/bin/activate
source scripts/phase8/env_torch_lib.sh
export PYTHONPATH=/workspace/KernelFuse
export TORCH_CUDA_ARCH_LIST=8.0
OUT=/workspace/kernelfuse_results/phase8
mkdir -p "$OUT" /workspace/kernelfuse_results/phase9

python - <<'PY'
import sys
import torch.utils.cpp_extension as cpp
cpp._check_cuda_version = lambda *a, **k: None
sys.argv = ["setup.py", "build_ext", "--inplace"]
import runpy
runpy.run_path("setup.py", run_name="__main__")
print("rebuild ok")
PY

KF_SO=$(python - <<'PY'
import glob, kernelfuse, os, sys
sys.stderr = open(os.devnull, "w")
d = os.path.dirname(kernelfuse.__file__)
print(glob.glob(os.path.join(d, "_C*.so"))[0])
PY
)
echo "KF_SO=$KF_SO"
echo "=== kernelfuse LDG/STG after uint4 fix ==="
cuobjdump -sass "$KF_SO" 2>/dev/null | grep -oE '\b(LDG|STG)(\.[A-Z0-9]+)+' | sort | uniq -c | sort -rn | tee "$OUT/sass_kf_ldg_stg_v4.txt"

python - <<'PY'
import pathlib, re, subprocess
from collections import Counter
out = pathlib.Path("/workspace/kernelfuse_results/phase8")
so = open(out/"kf_so_path.txt","w")
# path from env
import glob, kernelfuse, os
p = glob.glob(os.path.join(os.path.dirname(kernelfuse.__file__), "_C*.so"))[0]
pathlib.Path("/workspace/kernelfuse_results/phase8/kf_so_path.txt").write_text(p+"\n")
raw = subprocess.check_output(["cuobjdump","-sass",p], stderr=subprocess.DEVNULL, text=True, errors="replace")
parts = re.split(r"(?=^\s*Function\s*:)", raw, flags=re.M)
for part in parts:
    if "fused_add_rms_norm_bf16_vec8" in part[:500]:
        widths=Counter()
        for line in part.splitlines():
            m=re.search(r"\b((?:LDG|STG)(?:\.[A-Z0-9]+)+)", line)
            if m: widths[m.group(1)]+=1
        print("vec8 widths:")
        for k,v in widths.most_common():
            print(f"  {v:5d}  {k}")
        pathlib.Path("/workspace/kernelfuse_results/phase8/sass_kf_vec8_v4.txt").write_text(part)
PY

# vLLM SASS
bash scripts/phase8/sass_vllm_only.sh || true

# correctness + sweep
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

python scripts/phase8/kernel_microbench.py --graph \
  --out "$OUT/kernel_rows_sweep_graph_v4.csv" | tee "$OUT/kernel_rows_sweep_graph_v4.log"

python scripts/phase9/decode_attn_correctness.py \
  --out /workspace/kernelfuse_results/phase9/decode_attn_correctness_f32.csv

# four-array arithmetic from v4 log
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
