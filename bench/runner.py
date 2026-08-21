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
from bench.schema import CellResult, CellSpec, GenerateRequest


def load_matrix(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_cells(cfg: dict, backends: list[str] | None) -> list[CellSpec]:
    wanted = backends or list(cfg["backends"])
    run = cfg["run"]
    models = cfg["models"]
    sweep = cfg["sweep"]
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


def _prompt(input_tokens: int, seed: int, idx: int) -> str:
    word = f"w{seed}_{idx % 97}"
    return " ".join([word] * max(1, input_tokens))


def run_cell(cell: CellSpec, backend) -> CellResult:
    result = CellResult(cell=cell)

    def one(i: int, warmup: bool) -> None:
        req = GenerateRequest(
            prompt=_prompt(cell.input_tokens, cell.seed, i),
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
    print(
        "note: concurrency is a PLACEHOLDER on Phase 5 backends "
        "(hf_local CUDA lock = serial; mock = sleep overlap). "
        "Real scheduling results require vLLM/SGLang/TRT-LLM.",
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


def write_csv(path: Path, results: list[CellResult], percentiles: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
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
                "concurrency_note": "placeholder_phase5",
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
            w.writerow(row)


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
    args = parser.parse_args()

    cfg = load_matrix(args.matrix)
    percentiles = [int(p) for p in cfg["run"]["report_percentiles"]]
    cells = expand_cells(cfg, args.backends)
    if args.limit_cells > 0:
        cells = cells[: args.limit_cells]

    print(
        f"matrix={cfg['matrix_version']}  cells={len(cells)}  "
        f"backends={sorted({c.backend for c in cells})}",
        flush=True,
    )
    results: list[CellResult] = []
    by_backend: dict[str, list[CellSpec]] = {}
    for c in cells:
        by_backend.setdefault(c.backend, []).append(c)

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
                results.append(run_cell(cell, backend))
        finally:
            backend.stop()

    print(flush=True)
    print_table(results, percentiles)
    write_csv(args.out, results, percentiles)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
