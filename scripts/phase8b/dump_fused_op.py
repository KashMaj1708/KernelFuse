#!/usr/bin/env python3
import sglang
from pathlib import Path

p = Path(sglang.__file__).parent / "kernels" / "fused_op.py"
text = p.read_text(encoding="utf-8")
for needle in ("_RMSNormOp", "fused_add", "add_rms", "residual"):
    i = text.find(needle)
    while i >= 0:
        print(f"=== {needle} @ {i} ===")
        print(text[max(0, i - 200) : i + 600])
        print()
        i = text.find(needle, i + 1)
