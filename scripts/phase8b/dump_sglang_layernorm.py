#!/usr/bin/env python3
import sglang
from pathlib import Path

p = Path(sglang.__file__).parent / "srt" / "layers" / "layernorm.py"
print(p)
print(p.read_text(encoding="utf-8")[:8000])
