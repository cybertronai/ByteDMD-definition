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
    """Trace memory reads for Tiled C = A @ B.T with T×T blocks.

    Outer loops iterate over blocks; inner loops compute the math strictly
    inside the block. Both A and B tiles fit in cache simultaneously,
    so reads hit L1 instead of RAM.
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


def main():
    N = 16
    T = 4

    print(f"Tracing Naive Matmul (N={N})...")
    A_n, B_n = trace_naive_matmul(N)
    time_n = np.arange(len(A_n))

    print(f"Tracing Tiled Matmul (N={N}, T={T})...")
    A_t, B_t = trace_tiled_matmul(N, T)
    time_t = np.arange(len(A_t))

    print("Rendering plots...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True)

    offset_B = N * N
    scatter_kws = {'s': 8, 'alpha': 0.7, 'rasterized': True, 'linewidths': 0}

    # --- TOP PANEL: NAIVE ---
    axes[0].scatter(time_n, A_n, color='tab:blue', label='Matrix A reads', **scatter_kws)
    axes[0].scatter(time_n, B_n + offset_B, color='tab:red', label='Matrix B reads', **scatter_kws)
    axes[0].set_title(
        r"NAIVE Matrix Multiplication ($A \times B^T$), N=8" "\n"
        "B is swept top-to-bottom for every output element (cache thrashing)",
        fontsize=13, fontweight='bold', pad=10)
    axes[0].set_ylabel("1D Physical Address", fontsize=11)
    leg = axes[0].legend(loc='upper right', markerscale=3, framealpha=0.95, fontsize=10)

    # --- BOTTOM PANEL: TILED ---
    axes[1].scatter(time_t, A_t, color='tab:blue', label='Matrix A reads', **scatter_kws)
    axes[1].scatter(time_t, B_t + offset_B, color='tab:red', label='Matrix B reads', **scatter_kws)
    axes[1].set_title(
        f"TILED Matrix Multiplication (Tile = {T}x{T})\n"
        f"Accesses locked into small blocks — each tile fits in cache",
        fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xlabel("Time (Access Index)", fontsize=11)
    axes[1].set_ylabel("1D Physical Address", fontsize=11)
    axes[1].legend(loc='upper right', markerscale=3, framealpha=0.95, fontsize=10)

    for ax in axes:
        ax.axhline(offset_B, color='black', lw=1, ls='--', alpha=0.5)
        ax.set_yticks([0, N*N//2, N*N, offset_B + N*N//2, 2*N*N])
        ax.set_yticklabels(['0', f'{N*N//2}', f'{N*N}\nA|B boundary',
                            f'{offset_B + N*N//2}', f'{2*N*N}'], fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'matmul_access_pattern.svg')
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    print(f"N={N}, T={T}: {len(A_n):,} accesses per matrix")


if __name__ == "__main__":
    main()
