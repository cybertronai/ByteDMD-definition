#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Grid of cache-friendliness heuristics × algorithms.

Columns: naive_matmul, tiled_matmul, rmm, strassen, naive_attention,
         flash_attention.

Rows (heuristics, cheapest-to-compute first):
  n_loads           — raw load count (FLOP-like baseline)
  footprint         — distinct variables ever stored (cold-start bytes)
  peak_live         — max concurrently-live variables
  mwis_lower_bound  — interval-LP lower bound on any allocator
  bytedmd_classic   — Mattson LRU, no liveness compaction
  bytedmd_live      — LRU with liveness compaction
  min_heap          — greedy live-bytes allocator cost  (manual-placement ref)
  belady            — offline-optimal allocator cost    (manual-placement ref)

All heuristics operate on a common L2 trace produced by bytedmd_ir.trace().
The 2D Manhattan-distance cache model is baked into cost via
ceil(sqrt(addr)) = min disc radius containing addr cells.
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from bytedmd_ir import (
    L2Event,
    L2Load,
    L2Store,
    bytedmd_classic,
    bytedmd_live,
    compile_belady,
    compile_min_heap,
    cost,
    matmul_naive,
    matmul_rmm,
    matmul_tiled,
    mwis_lower_bound,
    trace,
)


# ============================================================================
# Extra algorithms
# ============================================================================

def _split(M):
    n = len(M); h = n // 2
    return ([[M[i][j] for j in range(h)] for i in range(h)],
            [[M[i][j] for j in range(h, n)] for i in range(h)],
            [[M[i][j] for j in range(h)] for i in range(h, n)],
            [[M[i][j] for j in range(h, n)] for i in range(h, n)])

def _join(C11, C12, C21, C22):
    h = len(C11); n = 2 * h
    return [[C11[i][j] if j < h else C12[i][j-h] for j in range(n)] for i in range(h)] + \
           [[C21[i][j] if j < h else C22[i][j-h] for j in range(n)] for i in range(h)]

def _addm(A, B):
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def _subm(A, B):
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


def matmul_strassen(A, B):
    """Strassen's algorithm: 7 recursive multiplies per 2x2 block."""
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)
    M1 = matmul_strassen(_addm(A11, A22), _addm(B11, B22))
    M2 = matmul_strassen(_addm(A21, A22), B11)
    M3 = matmul_strassen(A11, _subm(B12, B22))
    M4 = matmul_strassen(A22, _subm(B21, B11))
    M5 = matmul_strassen(_addm(A11, A12), B22)
    M6 = matmul_strassen(_subm(A21, A11), _addm(B11, B12))
    M7 = matmul_strassen(_subm(A12, A22), _addm(B21, B22))
    C11 = _addm(_subm(_addm(M1, M4), M5), M7)
    C12 = _addm(M3, M5)
    C21 = _addm(M2, M4)
    C22 = _addm(_subm(_addm(M1, M3), M2), M6)
    return _join(C11, C12, C21, C22)


# --- Attention helpers (trace-safe stand-ins for max/exp/inv) ---

def _max2(a, b):
    return a + b  # same read pattern as max(a, b)

def _exp(x):
    return x * x  # same read pattern as exp(x)

def _inv(x):
    return x * x  # same read pattern as 1/x


def naive_attention(Q, K, V):
    """softmax(Q @ K^T / sqrt(d)) @ V, one-shot."""
    N = len(Q); d = len(Q[0]); scale = d ** -0.5
    S = [[None] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            acc = Q[i][0] * K[j][0]
            for dd in range(1, d):
                acc = acc + Q[i][dd] * K[j][dd]
            S[i][j] = acc * scale
    P = [[None] * N for _ in range(N)]
    for i in range(N):
        mx = S[i][0]
        for j in range(1, N):
            mx = _max2(mx, S[i][j])
        row_sum = None
        for j in range(N):
            P[i][j] = _exp(S[i][j] - mx)
            row_sum = P[i][j] if row_sum is None else row_sum + P[i][j]
        inv_sum = _inv(row_sum)
        for j in range(N):
            P[i][j] = P[i][j] * inv_sum
    O = [[None] * d for _ in range(N)]
    for i in range(N):
        for dd in range(d):
            acc = P[i][0] * V[0][dd]
            for j in range(1, N):
                acc = acc + P[i][j] * V[j][dd]
            O[i][dd] = acc
    return O


def flash_attention(Q, K, V, Bk=2):
    """Flash attention with online softmax over K/V blocks of size Bk."""
    N = len(Q); d = len(Q[0]); scale = d ** -0.5
    num_blocks = (N + Bk - 1) // Bk
    O = [[None] * d for _ in range(N)]
    for i in range(N):
        m_prev = None; l_prev = None
        o_acc = [None] * d
        for kb in range(num_blocks):
            k0 = kb * Bk
            k1 = min(k0 + Bk, N)
            bs = k1 - k0
            s_block = [None] * bs
            for j in range(bs):
                kj = k0 + j
                acc = Q[i][0] * K[kj][0]
                for dd in range(1, d):
                    acc = acc + Q[i][dd] * K[kj][dd]
                s_block[j] = acc * scale
            m_block = s_block[0]
            for j in range(1, bs):
                m_block = _max2(m_block, s_block[j])
            p_block = [None] * bs
            l_block = None
            for j in range(bs):
                p_block[j] = _exp(s_block[j] - m_block)
                l_block = p_block[j] if l_block is None else l_block + p_block[j]
            o_block = [None] * d
            for dd in range(d):
                acc = p_block[0] * V[k0][dd]
                for j in range(1, bs):
                    acc = acc + p_block[j] * V[k0 + j][dd]
                o_block[dd] = acc
            if m_prev is None:
                m_prev = m_block; l_prev = l_block
                for dd in range(d):
                    o_acc[dd] = o_block[dd]
            else:
                m_new = _max2(m_prev, m_block)
                alpha = _exp(m_prev - m_new)
                beta = _exp(m_block - m_new)
                l_prev = alpha * l_prev + beta * l_block
                for dd in range(d):
                    o_acc[dd] = alpha * o_acc[dd] + beta * o_block[dd]
                m_prev = m_new
        inv_l = _inv(l_prev)
        for dd in range(d):
            O[i][dd] = o_acc[dd] * inv_l
    return O


# ============================================================================
# Extra heuristics (the ones not already in bytedmd_ir)
# ============================================================================

def n_loads(events: Sequence[L2Event]) -> int:
    return sum(1 for ev in events if isinstance(ev, L2Load))


def footprint(events: Sequence[L2Event]) -> int:
    """Count of distinct variables ever stored — total unique bytes."""
    return sum(1 for ev in events if isinstance(ev, L2Store))


def peak_live(events: Sequence[L2Event]) -> int:
    """Max number of variables that are concurrently live (between store and last load)."""
    last_load: Dict[int, int] = {}
    for i, ev in enumerate(events):
        if isinstance(ev, L2Load):
            last_load[ev.var] = i
    live = 0
    peak = 0
    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            if ev.var in last_load:
                live += 1
                if live > peak:
                    peak = live
        elif isinstance(ev, L2Load):
            if last_load.get(ev.var) == i:
                live -= 1
    return peak


def min_heap_cost(events: Sequence[L2Event]) -> int:
    return cost(compile_min_heap(events))


def belady_cost(events: Sequence[L2Event]) -> int:
    return cost(compile_belady(events))


# ============================================================================
# Grid definition
# ============================================================================

def make_mat(n: int, val: float = 1.0) -> List[List[float]]:
    return [[val] * n for _ in range(n)]


# (name, thunk) — each thunk returns (callable, args) ready for trace()
def build_algorithms(n_mm: int = 8, N_att: int = 8, d_att: int = 2, Bk: int = 2):
    A = make_mat(n_mm); B = make_mat(n_mm)
    Q = make_mat(N_att, 1.0); K = make_mat(N_att, 1.0); V = make_mat(N_att, 1.0)
    # Trim Q/K/V to N_att rows by d_att cols
    Q = [[1.0] * d_att for _ in range(N_att)]
    K = [[1.0] * d_att for _ in range(N_att)]
    V = [[1.0] * d_att for _ in range(N_att)]
    return [
        (f"naive_matmul(n={n_mm})",     matmul_naive,                       (A, B)),
        (f"tiled_matmul(n={n_mm})",     matmul_tiled,                       (A, B)),
        (f"rmm(n={n_mm})",              matmul_rmm,                         (A, B)),
        (f"strassen(n={n_mm})",         matmul_strassen,                    (A, B)),
        (f"naive_attn(N={N_att},d={d_att})",
            naive_attention,            (Q, K, V)),
        (f"flash_attn(N={N_att},d={d_att},Bk={Bk})",
            lambda Q, K, V: flash_attention(Q, K, V, Bk=Bk), (Q, K, V)),
    ]


HEURISTICS: List[Tuple[str, Callable[[Sequence[L2Event]], int]]] = [
    ("n_loads",          n_loads),
    ("footprint",        footprint),
    ("peak_live",        peak_live),
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
    algos = build_algorithms(n_mm=16, N_att=32, d_att=2, Bk=8)

    # Pre-trace each algorithm once (shared across all heuristics).
    traces: Dict[str, List[L2Event]] = {}
    trace_times: Dict[str, float] = {}
    for name, fn, args in algos:
        t0 = time.perf_counter()
        events, _ = trace(fn, args)
        dt = time.perf_counter() - t0
        traces[name] = events
        trace_times[name] = dt

    # Fill the grid.
    col_names = [name for name, _, _ in algos]
    row_names = [h[0] for h in HEURISTICS]
    grid = np.zeros((len(HEURISTICS), len(algos)), dtype=float)
    cell_time = np.zeros_like(grid)

    for ci, name in enumerate(col_names):
        events = traces[name]
        for ri, (hname, hfn) in enumerate(HEURISTICS):
            t0 = time.perf_counter()
            val = hfn(events)
            dt = time.perf_counter() - t0
            grid[ri, ci] = val
            cell_time[ri, ci] = dt
            if dt > CELL_BUDGET_S:
                print(f"WARN: cell ({hname}, {name}) took {dt:.2f}s > {CELL_BUDGET_S}s")

    # --- Print trace stats ---
    print("\nTrace sizes")
    print(f"{'algorithm':<34} {'events':>8} {'trace_s':>8}")
    print("-" * 52)
    for name in col_names:
        print(f"{name:<34} {len(traces[name]):>8} {trace_times[name]:>8.3f}")

    # --- Print grid ---
    print("\nGrid (raw)")
    header = f"{'heuristic':<18}" + "".join(f"{c:>20}" for c in col_names)
    print(header)
    print("-" * len(header))
    for ri, hname in enumerate(row_names):
        row = f"{hname:<18}" + "".join(f"{int(grid[ri, ci]):>20}" for ci in range(len(col_names)))
        print(row)

    print("\nCell time (s)")
    print(f"{'heuristic':<18}" + "".join(f"{c:>20}" for c in col_names))
    for ri, hname in enumerate(row_names):
        print(f"{hname:<18}" + "".join(f"{cell_time[ri, ci]:>20.3f}" for ci in range(len(col_names))))

    total = cell_time.sum() + sum(trace_times.values())
    print(f"\nTotal wall time: {total:.2f}s (trace: {sum(trace_times.values()):.2f}s, cells: {cell_time.sum():.2f}s)")

    # --- Save CSV ---
    csv_path = os.path.join(HERE, "grid.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["heuristic"] + col_names)
        for ri, hname in enumerate(row_names):
            w.writerow([hname] + [int(grid[ri, ci]) for ci in range(len(col_names))])
    print(f"Saved {csv_path}")

    # --- Save heatmap (column-normalized so cell = ratio to that column's min) ---
    col_min = grid.min(axis=0, keepdims=True)
    col_min = np.where(col_min == 0, 1, col_min)
    normalized = grid / col_min

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(normalized, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=30, ha="right")
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names)
    for ri in range(len(row_names)):
        for ci in range(len(col_names)):
            ax.text(ci, ri, f"{int(grid[ri, ci])}", ha="center", va="center",
                    color="white" if normalized[ri, ci] > normalized.max() * 0.5 else "black",
                    fontsize=7)
    ax.set_title("heuristic cost per algorithm (cell label = raw value, color = ratio to column min)")
    fig.colorbar(im, ax=ax, label="ratio to column min")
    fig.tight_layout()
    png_path = os.path.join(HERE, "grid.png")
    fig.savefig(png_path, dpi=140)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
