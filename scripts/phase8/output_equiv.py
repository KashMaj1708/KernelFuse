#!/usr/bin/env python3
"""Greedy output equivalence: baseline vs kernelfuse treatment on same prompt."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.backends.vllm_http import VllmHttpBackend
from bench.prompts import make_prompt
from bench.schema import GenerateRequest


def restart_server(use_kf: bool, results_dir: Path) -> None:
    tag = "treatment" if use_kf else "baseline"
    log = results_dir / f"vllm_{tag}_server.log"
    env = os.environ.copy()
    if use_kf:
        env["KERNELFUSE_FUSED_ADD_RMSNORM"] = "1"
    else:
        env.pop("KERNELFUSE_FUSED_ADD_RMSNORM", None)
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
    time.sleep(3)
    model = env.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
    revision = env.get("REVISION", "a09a35458c702b33eeacc393d103063234e8bc28")
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--revision",
        revision,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        env.get("DTYPE", "bfloat16"),
        "--gpu-memory-utilization",
        env.get("GPU_MEM_UTIL", "0.90"),
        "--max-model-len",
        env.get("MAX_MODEL_LEN", "4096"),
    ]
    with log.open("w", encoding="utf-8") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    import urllib.request

    for _ in range(180):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2)
            return
        except Exception:
            time.sleep(5)
    raise SystemExit(f"server failed to start for tag={tag}")


def greedy_text(model: str, prompt: str, max_tokens: int) -> str:
    os.environ.setdefault("KERNELFUSE_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    be = VllmHttpBackend(model)
    be.start()
    # Direct completion for full text capture
    import json as _json
    from urllib.request import Request, urlopen

    payload = {
        "model": be.model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    req = Request(
        f"{be.base_url}/completions",
        data=_json.dumps(payload).encode(),
        headers=be._headers(),
        method="POST",
    )
    with urlopen(req, timeout=600) as resp:
        data = _json.loads(resp.read().decode())
    choices = data.get("choices") or []
    return (choices[0].get("text") if choices else "") or ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    p.add_argument("--input-tokens", type=int, default=128)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=Path, default=Path("/workspace/kernelfuse_results/phase8"))
    p.add_argument("--no-restart", action="store_true", help="Only compare if server already on path")
    args = p.parse_args()

    prompt = make_prompt(args.input_tokens, args.seed, 0, model_id=args.model, max_prompt_tokens=4096 - args.output_tokens - 8)
    out_path = args.results_dir / "output_equiv.json"

    texts: dict[str, str] = {}
    for use_kf, label in ((False, "baseline"), (True, "treatment")):
        if not args.no_restart:
            restart_server(use_kf, args.results_dir)
        texts[label] = greedy_text(args.model, prompt, args.output_tokens)

    match = texts["baseline"] == texts["treatment"]
    result = {
        "match": match,
        "prompt_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "seed": args.seed,
        "baseline_len": len(texts["baseline"]),
        "treatment_len": len(texts["treatment"]),
        "baseline_preview": texts["baseline"][:200],
        "treatment_preview": texts["treatment"][:200],
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("OUTPUT_EQUIV", "PASS" if match else "FAIL")
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
