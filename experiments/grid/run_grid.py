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
    matmul_rmm,
    matmul_tiled,
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

def cube(d0: int, d1: int, d2: int, val: float = 1.0):
    return [[[val] * d2 for _ in range(d1)] for _ in range(d0)]

def tensor4(d0: int, d1: int, d2: int, d3: int, val: float = 1.0):
    return [[[[val] * d3 for _ in range(d2)] for _ in range(d1)] for _ in range(d0)]


# Sizes
N_MM = 16            # matrix size for matmul family
N_TR = 32            # size for transpose
N_MV = 64            # size for matvec
N_ATT, D_ATT, BK = 32, 2, 8  # attention
N_FFT = 256          # FFT input length (power of 2)
N_STENCIL = 32       # stencil grid side
LEAF_STENCIL = 8     # tile-recursion base size
H_SP, W_SP, K_SP = 32, 32, 5           # spatial (single-channel) convolution
H_CV, W_CV, K_CV, CIN, COUT = 16, 16, 3, 4, 4   # regular (multi-channel) conv
N_FFTC = 256                             # FFT-accelerated 1D convolution
N_SORT = 64                              # mergesort input length
M_LCS, N_LCS = 32, 32                    # LCS DP table (m+1) x (n+1)
N_LU = 32                                # LU / Cholesky square size
NB_LU = 8                                # block size for blocked_lu / blocked_qr
M_QR, N_QR = 32, 32                      # Householder / blocked QR
M_TSQR, N_TSQR, BR_TSQR = 64, 16, 8      # tall-skinny QR: 64x16, block=8

# Each algorithm is (display_name, traced_fn, traced_args, manual_cost_fn)
ALGOS: List[Tuple[str, Callable, Tuple, Callable[[], int]]] = [
    (f"naive_matmul(n={N_MM})",
        alg.matmul_naive_abt,      (mat(N_MM), mat(N_MM)),
        lambda: man.manual_naive_matmul(N_MM)),
    (f"tiled_matmul(n={N_MM})",
        matmul_tiled,              (mat(N_MM), mat(N_MM)),
        lambda: man.manual_tiled_matmul(N_MM)),
    (f"rmm(n={N_MM})",
        matmul_rmm,                (mat(N_MM), mat(N_MM)),
        lambda: man.manual_rmm(N_MM, T=4)),
    (f"naive_strassen(n={N_MM})",
        alg.matmul_strassen,       (mat(N_MM), mat(N_MM)),
        lambda: man.manual_strassen(N_MM, T=4)),
    (f"fused_strassen(n={N_MM})",
        alg.matmul_strassen,       (mat(N_MM), mat(N_MM)),
        lambda: man.manual_fused_strassen(N_MM, T=4)),
    (f"naive_attn(N={N_ATT},d={D_ATT})",
        alg.naive_attention,       (rect(N_ATT, D_ATT), rect(N_ATT, D_ATT), rect(N_ATT, D_ATT)),
        lambda: man.manual_naive_attention(N_ATT, D_ATT)),
    (f"flash_attn(N={N_ATT},d={D_ATT},Bk={BK})",
        lambda Q, K, V: alg.flash_attention(Q, K, V, Bk=BK),
        (rect(N_ATT, D_ATT), rect(N_ATT, D_ATT), rect(N_ATT, D_ATT)),
        lambda: man.manual_flash_attention(N_ATT, D_ATT, BK)),
    (f"matvec_row(n={N_MV})",
        alg.matvec_row,            (mat(N_MV), vec(N_MV)),
        lambda: man.manual_matvec_row(N_MV)),
    (f"matvec_col(n={N_MV})",
        alg.matvec_col,            (mat(N_MV), vec(N_MV)),
        lambda: man.manual_matvec_col(N_MV)),
    (f"fft_iterative(N={N_FFT})",
        alg.fft_iterative,         (vec(N_FFT),),
        lambda: man.manual_fft_iterative(N_FFT)),
    (f"fft_recursive(N={N_FFT})",
        alg.fft_recursive,         (vec(N_FFT),),
        lambda: man.manual_fft_recursive(N_FFT)),
    (f"stencil_naive({N_STENCIL}x{N_STENCIL})",
        alg.stencil_naive,         (mat(N_STENCIL),),
        lambda: man.manual_stencil_naive(N_STENCIL)),
    (f"stencil_recursive({N_STENCIL}x{N_STENCIL},leaf={LEAF_STENCIL})",
        lambda A: alg.stencil_recursive(A, leaf=LEAF_STENCIL),
        (mat(N_STENCIL),),
        lambda: man.manual_stencil_recursive(N_STENCIL, leaf=LEAF_STENCIL)),
    (f"spatial_conv({H_SP}x{W_SP},K={K_SP})",
        alg.spatial_convolution,
        (rect(H_SP, W_SP), rect(K_SP, K_SP)),
        lambda: man.manual_spatial_convolution(H_SP, W_SP, K_SP)),
    (f"regular_conv({H_CV}x{W_CV},K={K_CV},Cin={CIN},Cout={COUT})",
        alg.regular_convolution,
        (cube(H_CV, W_CV, CIN), tensor4(K_CV, K_CV, CIN, COUT)),
        lambda: man.manual_regular_convolution(H_CV, W_CV, K_CV, CIN, COUT)),
    (f"fft_conv(N={N_FFTC})",
        alg.fft_conv,
        (vec(N_FFTC), vec(N_FFTC)),
        lambda: man.manual_fft_conv(N_FFTC)),
    (f"quicksort(N={N_SORT})",
        alg.quicksort,
        (vec(N_SORT),),
        lambda: man.manual_quicksort(N_SORT)),
    (f"heapsort(N={N_SORT})",
        alg.heapsort,
        (vec(N_SORT),),
        lambda: man.manual_heapsort(N_SORT)),
    (f"mergesort(N={N_SORT})",
        alg.mergesort,
        (vec(N_SORT),),
        lambda: man.manual_mergesort(N_SORT)),
    (f"lcs_dp({M_LCS}x{N_LCS})",
        alg.lcs_dp,
        (vec(M_LCS), vec(N_LCS)),
        lambda: man.manual_lcs_dp(M_LCS, N_LCS)),
    (f"lu_no_pivot(n={N_LU})",
        alg.lu_no_pivot,             (mat(N_LU),),
        lambda: man.manual_lu_no_pivot(N_LU)),
    (f"blocked_lu(n={N_LU},NB={NB_LU})",
        lambda A: alg.blocked_lu(A, NB=NB_LU),
        (mat(N_LU),),
        lambda: man.manual_blocked_lu(N_LU, NB=NB_LU)),
    (f"recursive_lu(n={N_LU})",
        alg.recursive_lu,            (mat(N_LU),),
        lambda: man.manual_recursive_lu(N_LU)),
    (f"lu_partial_pivot(n={N_LU})",
        alg.lu_partial_pivot,        (mat(N_LU),),
        lambda: man.manual_lu_partial_pivot(N_LU)),
    (f"cholesky(n={N_LU})",
        alg.cholesky,                (mat(N_LU),),
        lambda: man.manual_cholesky(N_LU)),
    (f"householder_qr({M_QR}x{N_QR})",
        alg.householder_qr,          (rect(M_QR, N_QR),),
        lambda: man.manual_householder_qr(M_QR, N_QR)),
    (f"blocked_qr({M_QR}x{N_QR},NB={NB_LU})",
        lambda A: alg.blocked_qr(A, NB=NB_LU),
        (rect(M_QR, N_QR),),
        lambda: man.manual_blocked_qr(M_QR, N_QR, NB=NB_LU)),
    (f"tsqr({M_TSQR}x{N_TSQR},br={BR_TSQR})",
        lambda A: alg.tsqr(A, block_rows=BR_TSQR),
        (rect(M_TSQR, N_TSQR),),
        lambda: man.manual_tsqr(M_TSQR, N_TSQR, block_rows=BR_TSQR)),
]


# Column order for the output table. "manual" is a sentinel — value comes
# from ALGOS[i][3]() rather than from a trace-based heuristic.
METRICS: List[Tuple[str, Callable[[Sequence[L2Event]], int] | None]] = [
    ("bytedmd_live",    bytedmd_live),
    ("manual",          None),
    ("bytedmd_classic", bytedmd_classic),
]


# ============================================================================
# Driver
# ============================================================================

CELL_BUDGET_S = 10.0


def main() -> None:
    # Rows = algorithms, Cols = metrics (heuristics + manual, in METRICS order).
    algo_names = [a[0] for a in ALGOS]
    metric_names = [m[0] for m in METRICS]

    # Pre-trace each algorithm once.
    traces: Dict[str, List[L2Event]] = {}
    trace_times: Dict[str, float] = {}
    for name, fn, args, _ in ALGOS:
        t0 = time.perf_counter()
        events, _iv = trace(fn, args)
        trace_times[name] = time.perf_counter() - t0
        traces[name] = events

    # Fill the grid: grid[algo_index][metric_index]
    grid: List[List[int]] = [[0] * len(METRICS) for _ in ALGOS]
    cell_time: List[List[float]] = [[0.0] * len(METRICS) for _ in ALGOS]

    for ri, (name, _, _, manual_fn) in enumerate(ALGOS):
        events = traces[name]
        for ci, (mname, mfn) in enumerate(METRICS):
            t0 = time.perf_counter()
            val = manual_fn() if mfn is None else mfn(events)
            dt = time.perf_counter() - t0
            grid[ri][ci] = int(val)
            cell_time[ri][ci] = dt
            if dt > CELL_BUDGET_S:
                print(f"WARN cell ({name},{mname}) {dt:.2f}s > {CELL_BUDGET_S}s")

    # --- Trace stats ---
    print("\nTrace sizes")
    print(f"{'algorithm':<36} {'events':>8} {'trace_s':>8}")
    print("-" * 54)
    for name in algo_names:
        print(f"{name:<36} {len(traces[name]):>8} {trace_times[name]:>8.3f}")

    # --- Stdout table (algorithms as rows) ---
    algo_w = max(len("algorithm"), max(len(a) for a in algo_names))
    col_w = max(16, max(len(m) for m in metric_names))
    print("\nGrid (raw)")
    header = f"{'algorithm':<{algo_w}}" + "".join(f"{m:>{col_w+2}}" for m in metric_names)
    print(header)
    print("-" * len(header))
    for ri, aname in enumerate(algo_names):
        row = f"{aname:<{algo_w}}" + "".join(f"{grid[ri][ci]:>{col_w+2},}" for ci in range(len(metric_names)))
        print(row)

    print("\nCell time (s)")
    print(f"{'algorithm':<{algo_w}}" + "".join(f"{m:>{col_w+2}}" for m in metric_names))
    for ri, aname in enumerate(algo_names):
        print(f"{aname:<{algo_w}}" + "".join(f"{cell_time[ri][ci]:>{col_w+2}.3f}" for ci in range(len(metric_names))))

    total_cells = sum(sum(row) for row in cell_time)
    total = total_cells + sum(trace_times.values())
    print(f"\nTotal wall time: {total:.2f}s (trace: {sum(trace_times.values()):.2f}s, cells: {total_cells:.2f}s)")

    # --- CSV ---
    csv_path = os.path.join(HERE, "grid.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm"] + metric_names)
        for ri, aname in enumerate(algo_names):
            w.writerow([aname] + [grid[ri][ci] for ci in range(len(metric_names))])
    print(f"Saved {csv_path}")

    # --- Markdown table ---
    md_path = os.path.join(HERE, "grid.md")
    def fmt(v: int) -> str:
        return f"{v:,}"
    col_widths = [max(len(m), max(len(fmt(grid[ri][ci])) for ri in range(len(ALGOS))))
                  for ci, m in enumerate(metric_names)]
    fw = max(len("algorithm"), max(len(a) for a in algo_names))
    with open(md_path, "w") as f:
        f.write("| " + "algorithm".ljust(fw) + " | "
                + " | ".join(m.ljust(col_widths[ci]) for ci, m in enumerate(metric_names))
                + " |\n")
        f.write("|" + "-" * (fw + 2)
                + "|" + "|".join("-" * (w + 2) for w in col_widths)
                + "|\n")
        for ri, aname in enumerate(algo_names):
            f.write("| " + aname.ljust(fw) + " | "
                    + " | ".join(fmt(grid[ri][ci]).rjust(col_widths[ci])
                                 for ci in range(len(metric_names)))
                    + " |\n")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
