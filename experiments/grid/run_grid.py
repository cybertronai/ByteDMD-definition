#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Grid of cache-energy heuristics × algorithms.

Columns: algorithm (both a Python impl traced by bytedmd_ir and a hand-placed
         manual allocator impl).

Rows (heuristics — all approximate ceil(sqrt(addr))-priced energy):
  n_loads           — raw load count (energy if every access cost 1)
  mwis_lower_bound  — interval-LP lower bound on any allocator's cost
  bytedmd_classic   — Mattson LRU stack depth estimate (no liveness)
  bytedmd_live      — LRU with liveness compaction
  min_heap          — greedy live-bytes allocator (realistic automatic)
  belady            — offline-optimal allocator (best automatic)
  manual            — hand-placed schedule cost (gold-standard reference)
"""
from __future__ import annotations

import csv
import os
import sys
import time
from typing import Callable, Dict, List, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from bytedmd_ir import (
    L2Event,
    bytedmd_classic,
    bytedmd_live,
    compile_belady,
    compile_min_heap,
    cost,
    matmul_naive,
    matmul_rmm,
    matmul_tiled,
    mwis_lower_bound,
    n_loads,
    trace,
)
import algorithms as alg
import manual as man


# ============================================================================
# Grid definition
# ============================================================================

def mat(n: int, val: float = 1.0) -> List[List[float]]:
    return [[val] * n for _ in range(n)]

def rect(rows: int, cols: int, val: float = 1.0) -> List[List[float]]:
    return [[val] * cols for _ in range(rows)]

def vec(n: int, val: float = 1.0) -> List[float]:
    return [val] * n


# Sizes
N_MM = 16            # matrix size for matmul family
N_TR = 32            # size for transpose
N_MV = 64            # size for matvec
N_ATT, D_ATT, BK = 32, 2, 8  # attention

# Each algorithm is (display_name, traced_fn, traced_args, manual_cost_fn)
ALGOS: List[Tuple[str, Callable, Tuple, Callable[[], int]]] = [
    (f"naive_matmul(n={N_MM})",
        matmul_naive,              (mat(N_MM), mat(N_MM)),
        lambda: man.manual_naive_matmul(N_MM)),
    (f"tiled_matmul(n={N_MM})",
        matmul_tiled,              (mat(N_MM), mat(N_MM)),
        lambda: man.manual_tiled_matmul(N_MM)),
    (f"rmm(n={N_MM})",
        matmul_rmm,                (mat(N_MM), mat(N_MM)),
        lambda: man.manual_rmm(N_MM, T=4)),
    (f"strassen(n={N_MM})",
        alg.matmul_strassen,       (mat(N_MM), mat(N_MM)),
        lambda: man.manual_strassen(N_MM, T=4)),
    (f"naive_attn(N={N_ATT},d={D_ATT})",
        alg.naive_attention,       (rect(N_ATT, D_ATT), rect(N_ATT, D_ATT), rect(N_ATT, D_ATT)),
        lambda: man.manual_naive_attention(N_ATT, D_ATT)),
    (f"flash_attn(N={N_ATT},d={D_ATT},Bk={BK})",
        lambda Q, K, V: alg.flash_attention(Q, K, V, Bk=BK),
        (rect(N_ATT, D_ATT), rect(N_ATT, D_ATT), rect(N_ATT, D_ATT)),
        lambda: man.manual_flash_attention(N_ATT, D_ATT, BK)),
    (f"transpose_naive(n={N_TR})",
        alg.transpose_naive,       (mat(N_TR),),
        lambda: man.manual_transpose_naive(N_TR)),
    (f"transpose_blocked(n={N_TR})",
        alg.transpose_blocked,     (mat(N_TR),),
        lambda: man.manual_transpose_blocked(N_TR)),
    (f"transpose_recursive(n={N_TR})",
        alg.transpose_recursive,   (mat(N_TR),),
        lambda: man.manual_transpose_recursive(N_TR)),
    (f"matvec_row(n={N_MV})",
        alg.matvec_row,            (mat(N_MV), vec(N_MV)),
        lambda: man.manual_matvec_row(N_MV)),
    (f"matvec_col(n={N_MV})",
        alg.matvec_col,            (mat(N_MV), vec(N_MV)),
        lambda: man.manual_matvec_col(N_MV)),
]


def min_heap_cost(events: Sequence[L2Event]) -> int:
    return cost(compile_min_heap(events))

def belady_cost(events: Sequence[L2Event]) -> int:
    return cost(compile_belady(events))


HEURISTICS: List[Tuple[str, Callable[[Sequence[L2Event]], int]]] = [
    ("n_loads",          n_loads),
    ("mwis_lower_bound", mwis_lower_bound),
    ("bytedmd_classic",  bytedmd_classic),
    ("bytedmd_live",     bytedmd_live),
    ("min_heap",         min_heap_cost),
    ("belady",           belady_cost),
]


# ============================================================================
# Driver
# ============================================================================

CELL_BUDGET_S = 10.0


def main() -> None:
    col_names = [a[0] for a in ALGOS]

    # Pre-trace each algorithm once.
    traces: Dict[str, List[L2Event]] = {}
    trace_times: Dict[str, float] = {}
    for name, fn, args, _ in ALGOS:
        t0 = time.perf_counter()
        events, _iv = trace(fn, args)
        trace_times[name] = time.perf_counter() - t0
        traces[name] = events

    # Fill the grid (heuristic rows + manual row).
    row_names: List[str] = [h[0] for h in HEURISTICS] + ["manual"]
    grid: List[List[int]] = [[0] * len(ALGOS) for _ in row_names]
    cell_time: List[List[float]] = [[0.0] * len(ALGOS) for _ in row_names]

    for ci, name in enumerate(col_names):
        events = traces[name]
        for ri, (hname, hfn) in enumerate(HEURISTICS):
            t0 = time.perf_counter()
            val = hfn(events)
            dt = time.perf_counter() - t0
            grid[ri][ci] = int(val)
            cell_time[ri][ci] = dt
            if dt > CELL_BUDGET_S:
                print(f"WARN cell ({hname},{name}) {dt:.2f}s > {CELL_BUDGET_S}s")

    # Manual row
    manual_ri = len(HEURISTICS)
    for ci, (_, _, _, manual_fn) in enumerate(ALGOS):
        t0 = time.perf_counter()
        val = manual_fn()
        dt = time.perf_counter() - t0
        grid[manual_ri][ci] = int(val)
        cell_time[manual_ri][ci] = dt
        if dt > CELL_BUDGET_S:
            print(f"WARN manual cell ({col_names[ci]}) {dt:.2f}s > {CELL_BUDGET_S}s")

    # --- Trace stats ---
    print("\nTrace sizes")
    print(f"{'algorithm':<36} {'events':>8} {'trace_s':>8}")
    print("-" * 54)
    for name in col_names:
        print(f"{name:<36} {len(traces[name]):>8} {trace_times[name]:>8.3f}")

    # --- Stdout table ---
    col_w = max(18, max(len(c) for c in col_names))
    first_w = max(len("heuristic"), max(len(r) for r in row_names))
    print("\nGrid (raw)")
    header = f"{'heuristic':<{first_w}}" + "".join(f"{c:>{col_w+2}}" for c in col_names)
    print(header)
    print("-" * len(header))
    for ri, hname in enumerate(row_names):
        row = f"{hname:<{first_w}}" + "".join(f"{grid[ri][ci]:>{col_w+2}}" for ci in range(len(col_names)))
        print(row)

    print("\nCell time (s)")
    print(f"{'heuristic':<{first_w}}" + "".join(f"{c:>{col_w+2}}" for c in col_names))
    for ri, hname in enumerate(row_names):
        print(f"{hname:<{first_w}}" + "".join(f"{cell_time[ri][ci]:>{col_w+2}.3f}" for ci in range(len(col_names))))

    total_cells = sum(sum(row) for row in cell_time)
    total = total_cells + sum(trace_times.values())
    print(f"\nTotal wall time: {total:.2f}s (trace: {sum(trace_times.values()):.2f}s, cells: {total_cells:.2f}s)")

    # --- CSV ---
    csv_path = os.path.join(HERE, "grid.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["heuristic"] + col_names)
        for ri, hname in enumerate(row_names):
            w.writerow([hname] + [grid[ri][ci] for ci in range(len(col_names))])
    print(f"Saved {csv_path}")

    # --- Markdown table ---
    md_path = os.path.join(HERE, "grid.md")
    col_widths = [max(len(c), max(len(str(grid[ri][ci])) for ri in range(len(row_names))))
                  for ci, c in enumerate(col_names)]
    fw = max(len("heuristic"), max(len(r) for r in row_names))
    with open(md_path, "w") as f:
        f.write("| " + "heuristic".ljust(fw) + " | "
                + " | ".join(c.ljust(col_widths[ci]) for ci, c in enumerate(col_names))
                + " |\n")
        f.write("|" + "-" * (fw + 2)
                + "|" + "|".join("-" * (w + 2) for w in col_widths)
                + "|\n")
        for ri, hname in enumerate(row_names):
            f.write("| " + hname.ljust(fw) + " | "
                    + " | ".join(str(grid[ri][ci]).rjust(col_widths[ci])
                                 for ci in range(len(col_names)))
                    + " |\n")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
