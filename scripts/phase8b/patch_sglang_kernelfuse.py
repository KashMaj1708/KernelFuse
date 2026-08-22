#!/usr/bin/env python3
"""Patch SGLang 0.5.x layernorm.py to route fused_add_rmsnorm through kernelfuse."""

from __future__ import annotations

import os
import sys
from pathlib import Path

MARKER = "# KERNELFUSE_PATCH"


def find_layernorm() -> Path:
    import sglang  # noqa: WPS433

    p = Path(sglang.__file__).resolve().parent / "srt" / "layers" / "layernorm.py"
    if not p.is_file():
        raise SystemExit(f"layernorm.py not found at {p}")
    return p


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"already patched: {path}")
        return

    needle = """    from sgl_kernel import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )"""

    replacement = f"""    from sgl_kernel import (
        fused_add_rmsnorm as _sgl_fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )
    {MARKER}
    import os as _os_kf

    def fused_add_rmsnorm(x, residual, weight, eps):  # noqa: F811
        if _os_kf.environ.get("KERNELFUSE_FUSED_ADD_RMSNORM", "") == "1":
            from kernelfuse._C import fused_add_rms_norm as _kf_op
            _kf_op(x, residual, weight, eps)
            return
        return _sgl_fused_add_rmsnorm(x, residual, weight, eps)"""

    if needle not in text:
        raise SystemExit("sgl_kernel import block not found — SGLang version mismatch?")

    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    print(f"patched: {path}")


def main() -> int:
    if os.environ.get("KERNELFUSE_UNPATCH") == "1":
        print("KERNELFUSE_UNPATCH=1 — skip")
        return 0
    patch(find_layernorm())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
