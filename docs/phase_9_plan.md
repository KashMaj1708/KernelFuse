# Phase 9 — decode attention kernel (ceiling)

Optional stretch: single-query decode attention vs KV cache (bf16, fp32 softmax).

Not wired into serving — microbench + reference correctness only.

Artifacts: `kernels/attention/decode_attn_bench.cu`, `decode_attn_ref.py`

Script: `scripts/phase9/session_decode_attn_bench.sh`
