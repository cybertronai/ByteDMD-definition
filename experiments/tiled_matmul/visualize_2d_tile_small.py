#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Pedagogical 2D-tiling (output-stationary) at N=4, T=2.

4x4 matrices, 2 blocks per dimension -> 4 (bi, bj) phases, each doing
a full k-sweep of N=4 on a 2x2 output tile. 64 compute iterations,
128 interleaved A/B reads.
"""

import os
import sys
sys.path.insert(
    0,
    '/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/ByteDMD/experiments/tiled_matmul')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from visualize_tiling import trace_tiled_2d_matmul


def main():
    N = 4
    T = 2
    offset_B = N * N           # 16
    n_ops = N ** 3             # 64 compute ops
    phase_len = N * T * T      # 16 ops per (bi, bj) phase

    A_addr, B_addr = trace_tiled_2d_matmul(N, T)
    # Interleaved timeline: step 2*idx reads A, step 2*idx+1 reads B.
    t_A = 2 * np.arange(n_ops)
    t_B = 2 * np.arange(n_ops) + 1

    coords = []
    for bi in range(0, N, T):
        for bj in range(0, N, T):
            for k in range(N):
                for i in range(bi, bi + T):
                    for j in range(bj, bj + T):
                        coords.append((bi, bj, k, i, j))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(t_A, A_addr, color='tab:blue', s=90, alpha=0.9,
               linewidths=0, label='Matrix A read')
    ax.scatter(t_B, B_addr + offset_B, color='tab:red', s=90, alpha=0.9,
               linewidths=0, label='Matrix B read')

    # Label every dot with its (i, k) / (j, k) coordinate.
    for idx, (bi, bj, k, i, j) in enumerate(coords):
        ax.annotate(f'A[{i},{k}]', (2 * idx, i * N + k),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=6.5, color='tab:blue')
        ax.annotate(f'B[{j},{k}]', (2 * idx + 1, j * N + k + offset_B),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=6.5, color='tab:red')

    # Phase separators.
    for p in range(1, (N // T) ** 2):
        ax.axvline(2 * p * phase_len - 0.5, color='black', ls=':',
                   lw=1, alpha=0.55)

    # Phase labels under the plot.
    phase_idx = 0
    for bi in range(0, N, T):
        for bj in range(0, N, T):
            x_mid = 2 * (phase_idx * phase_len + phase_len / 2)
            ax.text(x_mid, -3.5,
                    f'phase {phase_idx + 1}: (bi={bi}, bj={bj})\n'
                    f'$C[{bi}:{bi+T}, {bj}:{bj+T}]$ accumulator pinned',
                    ha='center', va='top', fontsize=9, fontweight='bold')
            phase_idx += 1

    # A / B regions.
    ax.axhline(offset_B - 0.5, color='black', lw=1.2, ls='--', alpha=0.6)
    ax.text(-3, offset_B / 2 - 0.5,
            'A\n(rows of A,\naddrs 0..15)',
            ha='right', va='center', fontsize=9)
    ax.text(-3, offset_B + offset_B / 2 - 0.5,
            'B\n(rows of B,\naddrs 16..31)',
            ha='right', va='center', fontsize=9)

    ax.set_xlim(-6, 2 * n_ops + 2)
    ax.set_ylim(-6, 2 * offset_B + 1)
    ax.set_xlabel('Time (read index — A and B interleaved per compute step)')
    ax.set_ylabel('1D Physical Address')
    ax.set_title(
        rf'2D-TILED (output-stationary) $A \times B^T$ with $N={N}$, $T={T}$' '\n'
        f'{(N // T) ** 2} (bi, bj) phases × {N} k-steps × {T * T} (i,j) = '
        f'{n_ops} FMAs, {2 * n_ops} reads',
        fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='upper right', markerscale=0.7, framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yticks(list(range(0, 2 * offset_B + 1, 2)))
    ax.set_xticks(range(0, 2 * n_ops + 1, 8))

    fig.tight_layout()
    out = os.path.join(
        '/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/ByteDMD/'
        'experiments/tiled_matmul',
        'matmul_2d_tile_n4_t2.svg')
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
