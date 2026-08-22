"""
Phase 8 add+RMSNorm correctness — vLLM 0.8.5 fused_add_rms_norm semantics.

Compares CUDA binary against add_rmsnorm_torch golden (bf16 IO, fp32 reduce).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels" / "rmsnorm"))

from rmsnorm_ref import add_rmsnorm_torch  # noqa: E402

EXE = ROOT / "kernels" / "rmsnorm" / "add_rmsnorm_fused.exe"

PHASE8_SHAPES = [
    (1, 3584),
    (4, 3584),
    (8, 3584),
    (32, 3584),
    (1, 128),
    (4, 4096),
    (1, 8192),
]

EPS = 1e-6
RTOL = 2e-2
ATOL = 2e-2


def make_weight(rng: np.random.Generator, cols: int) -> torch.Tensor:
    w = rng.uniform(0.5, 2.0, size=(cols,)).astype(np.float32)
    return torch.from_numpy(w).to(dtype=torch.bfloat16)


def make_inputs(rng: np.random.Generator, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(rng.standard_normal((rows, cols)).astype(np.float32)).to(
        dtype=torch.bfloat16
    )
    r = torch.from_numpy(rng.standard_normal((rows, cols)).astype(np.float32)).to(
        dtype=torch.bfloat16
    )
    return x, r


def run_cuda(
    exe: Path,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, str | None]:
    rows, cols = x.shape
    with tempfile.TemporaryDirectory(prefix="kf_add_rmsnorm_") as td:
        td_path = Path(td)
        x_path = td_path / "x.bin"
        r_path = td_path / "r.bin"
        w_path = td_path / "w.bin"
        ox_path = td_path / "ox.bin"
        or_path = td_path / "or.bin"
        x.contiguous().cpu().view(torch.int16).numpy().tofile(x_path)
        residual.contiguous().cpu().view(torch.int16).numpy().tofile(r_path)
        weight.contiguous().cpu().view(torch.int16).numpy().tofile(w_path)

        cmd = [
            str(exe),
            str(rows),
            str(cols),
            repr(float(eps)),
            str(x_path),
            str(r_path),
            str(w_path),
            str(ox_path),
            str(or_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"CUDA runner failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        path = None
        for line in proc.stdout.splitlines():
            m = re.match(r"PATH\s+(\S+)", line.strip())
            if m:
                path = m.group(1)
        ox = torch.frombuffer(ox_path.read_bytes(), dtype=torch.int16).view(torch.bfloat16)
        or_ = torch.frombuffer(or_path.read_bytes(), dtype=torch.int16).view(torch.bfloat16)
        return ox.reshape(rows, cols), or_.reshape(rows, cols), path


def assert_close(
    got: torch.Tensor,
    ref: torch.Tensor,
    label: str,
    shape: tuple[int, int],
) -> None:
    g = got.float().cpu()
    r = ref.float().cpu()
    if not torch.allclose(g, r, rtol=RTOL, atol=ATOL):
        diff = (g - r).abs()
        i = int(diff.argmax())
        raise AssertionError(
            f"{label} mismatch shape={shape} max_abs={diff.max():.4e} "
            f"at flat {i} got={g.view(-1)[i]:.6g} ref={r.view(-1)[i]:.6g}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 8 add+RMSNorm correctness")
    parser.add_argument("--exe", type=Path, default=EXE)
    args = parser.parse_args()
    exe = args.exe.resolve()
    if not exe.is_file():
        print(f"Missing {exe} — run .\\scripts\\build_add_rmsnorm_fused.ps1", file=sys.stderr)
        return 1

    print(f"Testing {exe} vs vLLM 0.8.5 native golden (bf16, eps={EPS})")
    failures = 0
    for i, (rows, cols) in enumerate(PHASE8_SHAPES):
        rng = np.random.default_rng(2000 + i)
        x, residual = make_inputs(rng, rows, cols)
        weight = make_weight(rng, cols)
        x_ref = x.clone()
        r_ref = residual.clone()
        out_x, out_r = add_rmsnorm_torch(x_ref, r_ref, weight, eps=EPS)
        try:
            cuda_x, cuda_r, path = run_cuda(exe, x, residual, weight, EPS)
            assert_close(cuda_x, out_x, "output x", (rows, cols))
            assert_close(cuda_r, out_r, "residual", (rows, cols))
            print(f"  PASS shape=({rows}, {cols}) PATH={path}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL shape=({rows}, {cols}): {exc}", file=sys.stderr)

    if failures:
        print(f"\nFAILED: {failures} shape(s)", file=sys.stderr)
        return 1
    print(f"\nOK: all {len(PHASE8_SHAPES)} shapes match vLLM native golden")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
