#!/usr/bin/env python3
import vllm
from pathlib import Path

p = Path(vllm.__file__).resolve().parent / "model_executor" / "layers" / "layernorm.py"
print(p)
text = p.read_text(encoding="utf-8")
needle = "def fused_add_rms_norm"
idx = text.find(needle)
print("--- snippet ---")
print(repr(text[idx : idx + 600]) if idx >= 0 else "NOT FOUND")
