#!/usr/bin/env python3
import sys
try:
    from kernelfuse import _C
    print("_C ok", dir(_C))
    from kernelfuse import fused_add_rms_norm
    print("fused", fused_add_rms_norm)
except Exception as e:
    print("FAIL", type(e).__name__, e)
    sys.exit(1)
