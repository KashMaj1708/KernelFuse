#!/usr/bin/env bash
# One-symbol vLLM SASS + rows sweep + Phase 9. Never dump whole _C.so.
set -euo pipefail
cd /workspace/KernelFuse
source .venv-vllm/bin/activate
source scripts/phase8/env_torch_lib.sh
export PYTHONPATH=/workspace/KernelFuse
OUT=/workspace/kernelfuse_results/phase8
mkdir -p "$OUT" /workspace/kernelfuse_results/phase9

# Resolve .so without vLLM INFO polluting the path (logging often hits stdout).
SO=$(python -c 'import glob,logging,os,sys,vllm; logging.disable(logging.CRITICAL); print(sorted(glob.glob(os.path.dirname(vllm.__file__)+"/_C*.so"))[0])' 2>/dev/null | grep '\.so$' | tail -1)
echo "VLLM_SO=$SO"
test -f "$SO"

SYM=$(cuobjdump -symbols "$SO" 2>/dev/null | grep fused_add_rms_norm_kernel | grep BFloat16 | awk '{print $NF}' | head -1 || true)
if [[ -z "${SYM}" ]]; then
  echo "fallback symbol search..."
  cuobjdump -symbols "$SO" 2>/dev/null | grep -i fused_add_rms | head -30 || true
  SYM=$(cuobjdump -symbols "$SO" 2>/dev/null | grep -i fused_add_rms_norm | awk '{print $NF}' | head -1)
fi
echo "SYM=$SYM"
test -n "$SYM"

echo "=== vLLM one-symbol LDG/STG ==="
cuobjdump -fun "$SYM" -sass "$SO" 2>/dev/null | tee "$OUT/sass_vllm_fused_add_rms_bf16.txt" \
  | grep -oE '\b(LDG|STG)(\.[A-Z0-9]+)+' | sort | uniq -c | sort -rn \
  | tee "$OUT/sass_vllm_fused_add_rms_bf16_ldg_stg.txt"

echo "=== kf LDG/STG ==="
KF=$(python -c 'import glob,kernelfuse,os; print(glob.glob(os.path.dirname(kernelfuse.__file__)+"/_C*.so")[0])')
echo "KF_SO=$KF"
cuobjdump -sass "$KF" 2>/dev/null | grep -oE '\b(LDG|STG)(\.[A-Z0-9]+)+' | sort | uniq -c | sort -rn \
  | tee "$OUT/sass_kf_ldg_stg_v4.txt"

echo "=== correctness smoke ==="
python - <<'PY'
import torch, kernelfuse
from vllm import _custom_ops as ops
torch.manual_seed(0)
for rows in [1, 512, 4096]:
    x0 = torch.randn(rows, 3584, device="cuda", dtype=torch.bfloat16)
    r0 = torch.randn(rows, 3584, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(3584, device="cuda", dtype=torch.bfloat16)
    xk, rk = x0.clone(), r0.clone()
    xv, rv = x0.clone(), r0.clone()
    kernelfuse.fused_add_rms_norm(xk, rk, w, 1e-6)
    ops.fused_add_rms_norm(xv, rv, w, 1e-6)
    print(f"rows={rows} dx={(xk.float()-xv.float()).abs().max().item():.4e} dr={(rk.float()-rv.float()).abs().max().item():.4e}")
PY

echo "=== rows sweep ==="
python scripts/phase8/kernel_microbench.py --graph \
  --out "$OUT/kernel_rows_sweep_graph_v4.csv" | tee "$OUT/kernel_rows_sweep_graph_v4.log"

echo "=== phase9 ==="
python scripts/phase9/decode_attn_correctness.py \
  --out /workspace/kernelfuse_results/phase9/decode_attn_correctness_f32.csv

python - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(Path("/workspace/kernelfuse_results/phase8/kernel_rows_sweep_graph_v4.csv").open()))
print("keys", list(rows[0].keys()))
r = next(x for x in rows if int(x["rows"]) == 16384)
keys = list(r.keys())
kf_k = next(k for k in keys if "kernelfuse" in k and "min" in k)
vl_k = next(k for k in keys if "vllm" in k and "min" in k)
kf, vl = float(r[kf_k]), float(r[vl_k])
bytes4 = 4 * 16384 * 3584 * 2
print(f"rows=16384 kf={kf:.3f}us vllm={vl:.3f}us ratio={vl/kf:.3f}")
print(f"four_array_MB={bytes4/1e6:.1f}")
print(f"TB/s_kf={bytes4/(kf*1e-6)/1e12:.3f}  TB/s_vllm={bytes4/(vl*1e-6)/1e12:.3f}")
PY
echo ALL_DONE
