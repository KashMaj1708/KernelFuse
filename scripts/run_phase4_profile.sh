#!/usr/bin/env bash
# Phase 4 Nsight Compute sweep (Tier B/C — Linux Colab/Kaggle/rental).
# Prefer a Turing GPU (T4 = sm_75) so shared-memory occupancy matches the 1650 story.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/reports/phase4"
mkdir -p "$OUT"

log_clocks() {
  echo "=== GPU clocks ($1) ==="
  nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,temperature.gpu,power.draw --format=csv || true
}

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "missing $1"; exit 1; }; }
need_cmd nvcc
need_cmd nvidia-smi
if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not on PATH — install Nsight Compute or use the Colab notebook cell that locates it."
  exit 1
fi

log_clocks suite-start

ARCH="${CUDA_ARCH:-sm_75}"
NVCC_FLAGS=(-O3 -std=c++17 "-arch=${ARCH}")
build_one() {
  local src="$1" out="$2"
  echo "nvcc $src -> $out"
  nvcc "${NVCC_FLAGS[@]}" "$src" -o "$out"
}

build_one "$ROOT/kernels/rmsnorm/occupancy_probe.cu" "$OUT/occupancy_probe"
build_one "$ROOT/kernels/rmsnorm/rmsnorm_fused.cu" "$OUT/rmsnorm_fused"
build_one "$ROOT/kernels/rmsnorm/rmsnorm_fused_smem.cu" "$OUT/rmsnorm_fused_smem"
build_one "$ROOT/kernels/rmsnorm/rmsnorm_fused_vec4.cu" "$OUT/rmsnorm_fused_vec4"

echo "=== occupancy probe ==="
"$OUT/occupancy_probe" | tee "$OUT/occupancy_probe.txt"

METRICS=(
  sm__warps_active.avg.pct_of_peak_sustained_active
  launch__occupancy_limit_shared_mem
  launch__occupancy_limit_blocks
  launch__occupancy_limit_registers
  dram__bytes_read.sum
  dram__bytes_write.sum
  dram__bytes.sum.per_second
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
  smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active
  smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active
  smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
  gpu__time_duration.avg
)
METRICS_CSV=$(IFS=,; echo "${METRICS[*]}")

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

python3 - <<'PY' "$TMP"
import sys, numpy as np
from pathlib import Path
td = Path(sys.argv[1])
cases = [(2048, 4096), (2048, 8192)]
rng = np.random.default_rng(0)
for rows, cols in cases:
    rng.standard_normal((rows, cols), dtype=np.float32).tofile(td / f"x_{rows}_{cols}.bin")
    rng.uniform(0.5, 2.0, size=(cols,)).astype(np.float32).tofile(td / f"w_{cols}.bin")
    print("wrote", rows, cols)
PY

profile_case() {
  local name="$1" bin="$2" rows="$3" cols="$4"
  log_clocks "$name"
  local x="$TMP/x_${rows}_${cols}.bin"
  local w="$TMP/w_${cols}.bin"
  local rep="$OUT/${name}"
  echo "=== ncu $name ==="
  ncu --target-processes all --launch-count 1 \
    --metrics "$METRICS_CSV" \
    -o "$rep" --force-overwrite \
    "$bin" bench "$rows" "$cols" 1e-6 0 1 "$x" "$w" \
    | tee "$OUT/${name}_ncu.log"
  # CSV export for the report (works even headless)
  ncu --import "${rep}.ncu-rep" --csv \
    --page raw \
    > "$OUT/${name}.csv" 2>/dev/null || \
  ncu --import "${rep}.ncu-rep" --csv > "$OUT/${name}.csv" || true
}

profile_case smem_4096  "$OUT/rmsnorm_fused_smem" 2048 4096
profile_case smem_8192  "$OUT/rmsnorm_fused_smem" 2048 8192
profile_case vec4_8192  "$OUT/rmsnorm_fused_vec4" 2048 8192
profile_case fused_8192 "$OUT/rmsnorm_fused"      2048 8192

log_clocks suite-end
echo "Reports in $OUT"
echo "Primary chase: compare smem_4096 vs smem_8192 occupancy, dram__bytes.sum.per_second,"
echo "  and barrier vs long_scoreboard stall % — the 2.0x vs 2.7x gap lives there."
