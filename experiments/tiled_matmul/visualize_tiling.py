#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Visualize naive vs tiled matrix multiplication memory access patterns.

Plots Time (operation sequence) on X-axis and 1D Physical Memory Address on
Y-axis, making cache locality visually obvious:

  - Naive: Matrix B forms massive diagonal sweeps across its entire memory
    space (0 to N^2). The time gap between reusing the first row of B is
    O(N^2) operations — the cache will have completely overwritten it.
    Almost every read of B is a slow RAM fetch.

  - Tiled: Accesses form tight, localized blocks ("clouds"). For hundreds
    of operations, the CPU is locked into a tiny vertical band of memory.
    Because the block easily fits in the cache, the CPU executes the math
    at lightspeed before moving to the next block.

Usage:
    uv run --script visualize_tiling.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def trace_naive_matmul(N):
    """Trace memory reads for Naive C = A @ B.T.

    Computes C[i, j] += A[i, k] * B[j, k]. Both A and B are accessed
    row-major, but B is swept entirely for every (i, j) pair — the
    worst-case LRU pattern.
    """
    addr_A, addr_B = [], []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                addr_A.append(i * N + k)
                addr_B.append(j * N + k)
    return np.array(addr_A), np.array(addr_B)


def trace_tiled_matmul(N, T):
    """Trace memory reads for 3D-tiled C = A @ B.T with T×T×T blocks.

    Blocks all three loop indices (i, j, k) by T. Each (bi, bj, bk)
    block touches a T×T tile of A and B (2T² addresses) and performs
    T³ FMAs inside. The full T×T tile of C is assembled by iterating
    over N/T values of bk.
    """
    addr_A, addr_B = [], []
    for bi in range(0, N, T):
        for bj in range(0, N, T):
            for bk in range(0, N, T):
                for i in range(bi, min(bi + T, N)):
                    for j in range(bj, min(bj + T, N)):
                        for k in range(bk, min(bk + T, N)):
                            addr_A.append(i * N + k)
                            addr_B.append(j * N + k)
    return np.array(addr_A), np.array(addr_B)


def trace_naive_tiled_matmul(N, T):
    """Trace memory reads for naive matmul within output-partitioned tiles.

    Partitions output matrix C into T×T tiles. Within each tile, uses
    the naive inner order i → j → k: every C[i, j] is fully accumulated
    over the entire k = 0..N-1 range before moving to the next output
    element in the tile. Tiling is applied only to the output — each
    individual output element is still computed with a full naive inner
    product, so the full column of B is swept once per (i, j) pair.

    Loop nest: bi → bj → i → j → k. This is "partitioned matrix tiling"
    with no loop-interchange inside the tile.
    """
    addr_A, addr_B = [], []
    for bi in range(0, N, T):
        for bj in range(0, N, T):
            for i in range(bi, min(bi + T, N)):
                for j in range(bj, min(bj + T, N)):
                    for k in range(N):
                        addr_A.append(i * N + k)
                        addr_B.append(j * N + k)
    return np.array(addr_A), np.array(addr_B)


def trace_tiled_2d_matmul(N, T):
    """Trace memory reads for 2D-tiled (output-stationary) C = A @ B.T.

    Blocks only (i, j) by T; the contraction index k runs over the full
    N dimension per (bi, bj) tile. The T×T tile of C stays pinned as
    an accumulator while one row of A and one row of B stream in per k.
    Each (bi, bj) iteration is a full rank-N update to its output tile.

    Footprint inside a tile: T² C accumulators + 2T panel values per k
    step → 2NT distinct input addresses touched before moving on.
    """
    addr_A, addr_B = [], []
    for bi in range(0, N, T):
        for bj in range(0, N, T):
            for k in range(N):
                for i in range(bi, min(bi + T, N)):
                    for j in range(bj, min(bj + T, N)):
                        addr_A.append(i * N + k)
                        addr_B.append(j * N + k)
    return np.array(addr_A), np.array(addr_B)


def main():
    N = 8
    T = 4

    print(f"Tracing Naive Matmul (N={N})...")
    A_n, B_n = trace_naive_matmul(N)
    print(f"Tracing Naive-Tiled / output-partitioned Matmul (N={N}, T={T})...")
    A_nt, B_nt = trace_naive_tiled_matmul(N, T)
    print(f"Tracing 2D-Tiled / output-stationary Matmul (N={N}, T={T})...")
    A_2d, B_2d = trace_tiled_2d_matmul(N, T)
    print(f"Tracing 3D-Tiled Matmul (N={N}, T={T})...")
    A_3d, B_3d = trace_tiled_matmul(N, T)

    offset_B = N * N
    scatter_kws = {'s': 4, 'alpha': 0.9, 'rasterized': False, 'linewidths': 0}

    print("Rendering plots...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True, sharey=True)

    panels = [
        (axes[0], A_n, B_n,
         rf"NAIVE Matrix Multiplication ($A \times B^T$), N={N}" "\n"
         "B is swept top-to-bottom for every output element (cache thrashing)"),
        (axes[1], A_nt, B_nt,
         rf"NAIVE-TILED / output-partitioned (N={N}, T={T}) — blocks $i, j$, inner order $i{{\to}}j{{\to}}k$" "\n"
         f"each C[i,j] is still fully accumulated over all N values of k before moving on — B sweeps persist inside each tile"),
        (axes[2], A_2d, B_2d,
         rf"2D-TILED / output-stationary (N={N}, T={T}) — blocks $i, j$ with $k$ hoisted outside" "\n"
         f"T$\\times$T=$\\mathbf{{{T*T}}}$ accumulators pinned; k sweeps full N per tile, A and B panels stream"),
        (axes[3], A_3d, B_3d,
         rf"3D-TILED (N={N}, T={T}) — blocks $i, j, k$" "\n"
         f"each inner (bi, bj, bk) block reuses a {T}$\\times${T} tile of both A and B"),
    ]

    for ax, A, B, title in panels:
        t = np.arange(len(A))
        ax.scatter(t, A, color='tab:blue', label='Matrix A reads', **scatter_kws)
        ax.scatter(t, B + offset_B, color='tab:red', label='Matrix B reads',
                   **scatter_kws)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.set_ylabel("1D Physical Address", fontsize=10)
        ax.legend(loc='upper right', markerscale=3, framealpha=0.95, fontsize=9)
        ax.axhline(offset_B, color='black', lw=1, ls='--', alpha=0.5)
        ax.set_yticks([0, N*N, 2*N*N])
        ax.set_yticklabels(['0', f'{N*N}\nA|B boundary', f'{2*N*N}'],
                           fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (Access Index)", fontsize=11)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'matmul_access_pattern.svg')
    plt.savefig(out, bbox_inches='tight', dpi=720)
    print(f"Saved: {out}")
    print(f"N={N}, T={T}: {len(A_n):,} accesses per matrix per variant")


if __name__ == "__main__":
    main()
