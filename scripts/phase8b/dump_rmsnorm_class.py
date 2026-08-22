#!/usr/bin/env python3
import sglang,re
from pathlib import Path
p = Path(sglang.__file__).parent / "srt" / "layers" / "layernorm.py"
text = p.read_text(encoding="utf-8")
# extract RMSNorm class body until next top-level class
m = re.search(r"class RMSNorm\(BaseFusedOp\):.*?(?=\nclass |\Z)", text, re.S)
if m:
    body = m.group(0)
    for needle in ("forward_cuda", "fused_add_rmsnorm", "forward_native"):
        for match in re.finditer(re.escape(needle), body):
            i = match.start()
            print(f"--- {needle} ---")
            print(body[i:i+900])
            print()
