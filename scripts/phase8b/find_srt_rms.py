#!/usr/bin/env python3
import sglang
from pathlib import Path

root = Path(sglang.__file__).parent
for sub in ("srt", "models"):
    base = root / sub
    if not base.is_dir():
        continue
    for p in base.rglob("*.py"):
        t = p.read_text(errors="ignore")
        if "RMSNorm" in t or "rms_norm" in t:
            if "class " in t or "def " in t:
                print(p)
