#!/usr/bin/env python3
"""Find SGLang RMSNorm / fused-add call sites."""
from __future__ import annotations

import sys
from pathlib import Path

import sglang

root = Path(sglang.__file__).resolve().parent
print("sglang", root)
needles = ("fused_add_rms", "fused_add_rms_norm", "rmsnorm", "RMSNorm")
for p in sorted(root.rglob("*.py")):
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    low = text.lower()
    if any(n.lower() in low for n in needles):
        if "def " in text and ("rms" in low or "norm" in low):
            print("---", p.relative_to(root))
            for i, line in enumerate(text.splitlines(), 1):
                if any(n.lower() in line.lower() for n in needles):
                    print(f"{i:5d}: {line[:120]}")
