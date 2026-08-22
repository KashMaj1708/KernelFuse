#!/usr/bin/env python3
import sglang
from pathlib import Path

p = Path(sglang.__file__).parent / "srt" / "layers" / "layernorm.py"
text = p.read_text(encoding="utf-8")
for needle in ("class RMSNorm", "fused_add_rmsnorm", "def forward"):
    idx = 0
    while True:
        i = text.find(needle, idx)
        if i < 0:
            break
        print(f"=== {needle} @ {i} ===")
        print(text[i : i + 1200])
        print()
        idx = i + len(needle)
