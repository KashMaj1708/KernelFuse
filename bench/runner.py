"""Phase 5+ harness runner — expand matrix, drive backends, emit percentiles.

Reports:
  - e2e latency p50/p90/p99
  - TTFT / TPOT p50/p90/p99 (streaming)
  - system throughput (total out tokens / wall clock)
  - mean per-request tok/s

Concurrency note (Phase 5): under hf_local's CUDA lock, c>1 is serialized;
under mock, c>1 is sleep overlap. Labelled placeholder until vLLM/SGLang.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.backends import make_backend
from bench.metrics import summarize_latencies
from bench.prompts import make_prompt
from bench.schema import CellResult, CellSpec, GenerateRequest


def load_matrix(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_cells(
    cfg: dict,
    backends: list[str] | None,
    *,
    smoke: bool = False,
    prefill_heavy: bool = False,
) -> list[CellSpec]:
    wanted = backends or list(cfg["backends"])
    run = cfg["run"]
    models = cfg["models"]
    sweep = dict(cfg["sweep"])
    if smoke:
        override = (
            cfg.get("instance_smoke")
            or cfg.get("colab_smoke")
            or cfg.get("tier_a_smoke_subset")
        )
        if not override:
            raise KeyError(
                "smoke requested but matrix has no instance_smoke/colab_smoke/tier_a_smoke_subset"
            )
        for key in ("batch_size", "input_tokens", "output_tokens", "concurrency"):
            if key in override:
                sweep[key] = override[key]
    if prefill_heavy:
        override = cfg.get("prefill_heavy")
        if not override:
            raise KeyError("prefill_heavy requested but matrix has no prefill_heavy block")
        for key in ("batch_size", "input_tokens", "output_tokens", "concurrency"):
            if key in override:
                sweep[key] = override[key]
    cells: list[CellSpec] = []
    for backend in wanted:
        if backend not in models:
            raise KeyError(f"no model mapping for backend {backend}")
        model = models[backend]
        for bs, itok, otok, conc in itertools.product(
            sweep["batch_size"],
            sweep["input_tokens"],
            sweep["output_tokens"],
            sweep["concurrency"],
        ):
            cells.append(
                CellSpec(
                    matrix_version=cfg["matrix_version"],
                    backend=backend,
                    model=model,
                    batch_size=bs,
                    input_tokens=itok,
                    output_tokens=otok,
                    concurrency=conc,
                    warmup_requests=int(run["warmup_requests"]),
                    num_requests=int(run["num_requests"]),
                    seed=int(run["seed"]),
                )
            )
    return cells


def _prompt(
    input_tokens: int,
    seed: int,
    idx: int,
    *,
    model_id: str,
    max_model_len: int | None,
    output_tokens: int,
) -> str:
    # Leave room for generation + small margin inside the engine context window.
    max_prompt = None
    if max_model_len is not None:
        max_prompt = max(8, int(max_model_len) - int(output_tokens) - 8)
    return make_prompt(
        input_tokens,
        seed,
        idx,
        model_id=model_id,
        max_prompt_tokens=max_prompt,
    )


def run_cell(
    cell: CellSpec,
    backend,
    *,
    max_model_len: int | None = None,
) -> CellResult:
    result = CellResult(cell=cell)

    def one(i: int, warmup: bool) -> None:
        req = GenerateRequest(
            prompt=_prompt(
                cell.input_tokens,
                cell.seed,
                i,
                model_id=cell.model,
                max_model_len=max_model_len,
                output_tokens=cell.output_tokens,
            ),
            max_new_tokens=cell.output_tokens,
            request_id=f"{cell.backend}-{i}-{'w' if warmup else 'm'}",
        )
        # batch_size > 1: sequential generates per request slot (real batching = vLLM).
        lat_sum = 0.0
        tok_sum = 0
        ttft_sum = 0.0
        tpot_vals: list[float] = []
        n_ttft = 0
        ok = True
        first_err: str | None = None
        for b in range(cell.batch_size):
            r = backend.generate(
                GenerateRequest(
                    prompt=req.prompt + f" b{b}",
                    max_new_tokens=req.max_new_tokens,
                    request_id=f"{req.request_id}-b{b}",
                )
            )
            lat_sum += r.latency_ms
            tok_sum += r.output_tokens
            if r.ttft_ms is not None:
                ttft_sum += r.ttft_ms
                n_ttft += 1
            if r.tpot_ms is not None:
                tpot_vals.append(r.tpot_ms)
            if not r.ok:
                ok = False
                if first_err is None and r.error:
                    first_err = r.error
        if warmup:
            return
        if not ok:
            result.errors += 1
            if result.errors == 1 and first_err:
                print(f"    first error: {first_err[:240]}", flush=True)
        result.latencies_ms.append(lat_sum)
        result.output_tokens_total += tok_sum
        if n_ttft > 0:
            result.ttft_ms.append(ttft_sum / n_ttft)
        if tpot_vals:
            result.tpot_ms.append(sum(tpot_vals) / len(tpot_vals))
        if lat_sum > 0 and tok_sum > 0:
            result.per_request_tok_s.append(tok_sum / (lat_sum / 1000.0))

    for i in range(cell.warmup_requests):
        one(i, warmup=True)

    t0 = time.perf_counter()
    if cell.concurrency <= 1:
        for i in range(cell.num_requests):
            one(i, warmup=False)
    else:
        with ThreadPoolExecutor(max_workers=cell.concurrency) as pool:
            futs = [pool.submit(one, i, False) for i in range(cell.num_requests)]
            for f in as_completed(futs):
                f.result()
    result.wall_s = time.perf_counter() - t0
    return result


def print_table(results: list[CellResult], percentiles: list[int]) -> None:
    if results and results[0].cell.matrix_version.startswith("phase5"):
        print(
            "note: concurrency is a PLACEHOLDER on Phase 5 backends "
            "(hf_local CUDA lock = serial; mock = sleep overlap). "
            "Real scheduling results require vLLM/SGLang/TRT-LLM.",
            flush=True,
        )
    elif results and any(r.cell.backend == "vllm" for r in results):
        print(
            "note: T4/sm_75 uses XFormers (not FlashAttention). "
            "Record dtype=float16 + attention_backend in run_metadata.json. "
            "Phase 6 vs Phase 7 are not cross-tier comparable.",
            flush=True,
        )
    hdr = (
        f"{'backend':<10} {'bs':>3} {'in':>4} {'out':>4} {'c':>2}  "
        f"{'p50':>7} {'p90':>7}  "
        f"{'ttft50':>7} {'tpot50':>7}  "
        f"{'sys_t/s':>8} {'req_t/s':>8}  {'err':>3}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in results:
        e2e = summarize_latencies(r.latencies_ms, percentiles) if r.latencies_ms else {}
        ttft = summarize_latencies(r.ttft_ms, percentiles) if r.ttft_ms else {}
        tpot = summarize_latencies(r.tpot_ms, percentiles) if r.tpot_ms else {}
        print(
            f"{r.cell.backend:<10} {r.cell.batch_size:3d} "
            f"{r.cell.input_tokens:4d} {r.cell.output_tokens:4d} {r.cell.concurrency:2d}  "
            f"{e2e.get('p50', float('nan')):7.1f} {e2e.get('p90', float('nan')):7.1f}  "
            f"{ttft.get('p50', float('nan')):7.1f} {tpot.get('p50', float('nan')):7.1f}  "
            f"{r.system_throughput_tok_s:8.1f} {r.mean_per_request_tok_s:8.1f}  "
            f"{r.errors:3d}",
            flush=True,
        )


def _concurrency_note(matrix_version: str, backend: str) -> str:
    if matrix_version.startswith("phase5") or backend in ("mock", "hf_local"):
        return "placeholder_phase5"
    return "real_scheduling"


def _row_from_result(r: CellResult, percentiles: list[int]) -> dict:
    e2e = summarize_latencies(r.latencies_ms, percentiles) if r.latencies_ms else {}
    ttft = summarize_latencies(r.ttft_ms, percentiles) if r.ttft_ms else {}
    tpot = summarize_latencies(r.tpot_ms, percentiles) if r.tpot_ms else {}
    row = {
        "matrix_version": r.cell.matrix_version,
        "backend": r.cell.backend,
        "model": r.cell.model,
        "batch_size": r.cell.batch_size,
        "input_tokens": r.cell.input_tokens,
        "output_tokens": r.cell.output_tokens,
        "concurrency": r.cell.concurrency,
        "concurrency_note": _concurrency_note(r.cell.matrix_version, r.cell.backend),
        "system_throughput_tok_s": f"{r.system_throughput_tok_s:.4f}",
        "mean_per_request_tok_s": f"{r.mean_per_request_tok_s:.4f}",
        "errors": r.errors,
        "n_samples": len(r.latencies_ms),
        "wall_s": f"{r.wall_s:.4f}",
    }
    for p in percentiles:
        row[f"p{p}_ms"] = f"{e2e.get('p'+str(p), float('nan')):.4f}"
        row[f"ttft_p{p}_ms"] = f"{ttft.get('p'+str(p), float('nan')):.4f}"
        row[f"tpot_p{p}_ms"] = f"{tpot.get('p'+str(p), float('nan')):.4f}"
    return row


def csv_fields(percentiles: list[int]) -> list[str]:
    return [
        "matrix_version",
        "backend",
        "model",
        "batch_size",
        "input_tokens",
        "output_tokens",
        "concurrency",
        "concurrency_note",
        *[f"p{p}_ms" for p in percentiles],
        *[f"ttft_p{p}_ms" for p in percentiles],
        *[f"tpot_p{p}_ms" for p in percentiles],
        "system_throughput_tok_s",
        "mean_per_request_tok_s",
        "errors",
        "n_samples",
        "wall_s",
    ]


def write_csv(path: Path, results: list[CellResult], percentiles: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = csv_fields(percentiles)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(_row_from_result(r, percentiles))


def append_csv_row(path: Path, result: CellResult, percentiles: list[int]) -> None:
    """Incremental write — survive mid-session interruption (Phase 7)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = csv_fields(percentiles)
    new_file = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow(_row_from_result(result, percentiles))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5+ serving harness")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "bench" / "config_matrix_phase5_v1.yaml",
    )
    parser.add_argument("--backends", nargs="+", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "phase5" / "results_phase5_v1.csv",
    )
    parser.add_argument("--limit-cells", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use matrix colab_smoke (or tier_a_smoke_subset) axes instead of full sweep.",
    )
    parser.add_argument(
        "--prefill-heavy",
        action="store_true",
        help="Use matrix prefill_heavy axes (long in, short out) instead of full sweep.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="Write run_metadata.json (default: <out_dir>/run_metadata.json).",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        help="Recorded attention backend (e.g. xformers). Defaults from matrix/env.",
    )
    args = parser.parse_args()

    cfg = load_matrix(args.matrix)
    launch = cfg.get("launch") or {}
    max_model_len = launch.get("max_model_len")
    if max_model_len is not None:
        max_model_len = int(max_model_len)
    percentiles = [int(p) for p in cfg["run"]["report_percentiles"]]
    cells = expand_cells(
        cfg, args.backends, smoke=args.smoke, prefill_heavy=args.prefill_heavy
    )
    if args.limit_cells > 0:
        cells = cells[: args.limit_cells]

    print(
        f"matrix={cfg['matrix_version']}  cells={len(cells)}  smoke={args.smoke}  "
        f"prefill_heavy={args.prefill_heavy}  "
        f"backends={sorted({c.backend for c in cells})}  max_model_len={max_model_len}",
        flush=True,
    )
    results: list[CellResult] = []
    by_backend: dict[str, list[CellSpec]] = {}
    for c in cells:
        by_backend.setdefault(c.backend, []).append(c)

    # Incremental CSV: truncate once at start, then append each cell.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.is_file():
        args.out.unlink()

    for bname, bcells in by_backend.items():
        model = bcells[0].model
        backend = make_backend(bname, model)
        print(f"\n=== backend={bname} model={model} ===", flush=True)
        backend.start()
        try:
            for cell in bcells:
                print(
                    f"  cell bs={cell.batch_size} in={cell.input_tokens} "
                    f"out={cell.output_tokens} c={cell.concurrency} ...",
                    flush=True,
                )
                cell_result = run_cell(cell, backend, max_model_len=max_model_len)
                results.append(cell_result)
                append_csv_row(args.out, cell_result, percentiles)
                print(f"    appended -> {args.out}", flush=True)
        finally:
            backend.stop()

    print(flush=True)
    print_table(results, percentiles)
    print(f"\nwrote {args.out} ({len(results)} cells)", flush=True)

    # Provenance sidecar (required for Phase 6+ real backends).
    meta_path = args.metadata_out or (args.out.parent / "run_metadata.json")
    from bench.run_metadata import collect_run_metadata, write_run_metadata

    model_id = cells[0].model if cells else ""
    attn = (
        args.attention_backend
        or launch.get("attention_backend")
        or ("xformers" if str(cfg.get("hardware_tier", "")).upper() in ("B", "A") else None)
    )
    meta = collect_run_metadata(
        matrix_version=str(cfg["matrix_version"]),
        model_id=model_id,
        dtype=str(launch.get("dtype", "float16")),
        enforce_eager=bool(launch.get("enforce_eager", True)),
        gpu_memory_utilization=float(launch.get("gpu_memory_utilization", 0.75)),
        attention_backend=attn,
        extra={
            "smoke": args.smoke,
            "prefill_heavy": args.prefill_heavy,
            "out_csv": str(args.out),
            "max_model_len": max_model_len,
            "prompt_tokenization": "hf_tokenizer",
        },
    )
    write_run_metadata(meta_path, meta)
    print(f"wrote {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
