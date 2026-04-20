#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Naive vs Tiled matmul on the Manhattan Diamond memory model.

Uses the two-arena layout from docs/manhattan-diamond.md:
  - Arguments (A, B) at negative addresses (read-only, above the core)
  - Scratch (scratchpad tiles, accumulator, C output) at positive addresses
  - Reading address d costs ceil(sqrt(|d|)), writes are free

Uses small matrices (N=4, T=2) so every access is individually visible.
Both reads (costed) and writes (free) are traced.

Usage:
    uv run --script visualize_manual.py
"""

import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


REGION_COLORS = {
    'scratch': 'tab:cyan',
    'arg_A': 'tab:blue',
    'arg_B': 'tab:red',
    'out_C': 'tab:purple',
    'accum': 'tab:green',
}


def read_cost(addr):
    """Manhattan diamond cost: ceil(sqrt(|addr|))."""
    return math.isqrt(abs(addr) - 1) + 1 if addr != 0 else 0


def trace_naive(N):
    """Naive triple-loop C += A @ B^T on Manhattan Diamond.

    Arguments layout (negative addresses, read-only):
      -1..-N^2     : matrix A (row-major)
      -(N^2+1)..-2N^2 : matrix B (row-major)

    Scratch layout (positive addresses):
      +1           : accumulator s
      +2..+N^2+1   : output matrix C

    Returns (addrs, is_write, annotations, regions, cost).
    """
    addrs = []
    writes = []
    annotations = []
    cost = 0

    # Argument addresses (negative, read-only)
    A = -1          # A[i][j] at -(1 + i*N + j)
    B = -(N * N + 1)  # B[i][j] at -(N^2 + 1 + i*N + j)

    def addr_A(i, j): return -(1 + i * N + j)
    def addr_B(i, j): return -(N * N + 1 + i * N + j)

    # Scratch addresses (positive)
    s = 1                 # accumulator
    C = 2                 # C[i][j] at 2 + i*N + j

    def addr_C(i, j): return 2 + i * N + j

    def read(addr, desc):
        nonlocal cost
        c = read_cost(addr)
        addrs.append(addr)
        writes.append(False)
        annotations.append((addr, c, 'R', desc))
        cost += c

    def write(addr, desc):
        addrs.append(addr)
        writes.append(True)
        annotations.append((addr, 0, 'W', desc))

    for i in range(N):
        for j in range(N):
            read(addr_C(i, j), f"C[{i}][{j}] (init)")
            read(addr_A(i, 0), f"A[{i}][0]")
            read(addr_B(j, 0), f"B[{j}][0]")
            write(s, f"s = A[{i}][0]*B[{j}][0]")
            for k in range(1, N):
                read(addr_A(i, k), f"A[{i}][{k}]")
                read(addr_B(j, k), f"B[{j}][{k}]")
                read(s, f"s (acc for C[{i}][{j}])")
                write(s, f"s += A[{i}][{k}]*B[{j}][{k}]")
            read(s, f"s final")
            write(addr_C(i, j), f"C[{i}][{j}] = s")

    regions = {
        'accum': (s, s),
        'out_C': (C, C + N*N - 1),
        'arg_A': (addr_A(N-1, N-1), addr_A(0, 0)),
        'arg_B': (addr_B(N-1, N-1), addr_B(0, 0)),
    }
    return addrs, writes, annotations, regions, cost


def trace_tiled(N, T):
    """Tiled matmul with scratchpad on Manhattan Diamond.

    Arguments layout (negative addresses, read-only):
      -1..-N^2     : matrix A
      -(N^2+1)..-2N^2 : matrix B

    Scratch layout (positive addresses, fast):
      +1..+T^2     : scratchpad tile sA
      +T^2+1..+2T^2 : scratchpad tile sB
      +2T^2+1..+3T^2 : scratchpad tile sC
      +3T^2+1..+3T^2+N^2 : output matrix C

    Returns (addrs, is_write, annotations, regions, cost).
    """
    addrs = []
    writes = []
    annotations = []
    cost = 0

    def addr_A(i, j): return -(1 + i * N + j)
    def addr_B(i, j): return -(N * N + 1 + i * N + j)

    # Scratch addresses (positive, low = fast)
    sA = 1
    sB = 1 + T * T
    sC = 1 + 2 * T * T
    C = 1 + 3 * T * T  # C[i][j] at C + i*N + j

    def addr_sA(ii, kk): return sA + ii * T + kk
    # AB^T: sB holds a T-by-T tile of B indexed row-major by [jj][kk].
    def addr_sB(jj, kk): return sB + jj * T + kk
    def addr_sC(ii, jj): return sC + ii * T + jj
    def addr_C(i, j): return C + i * N + j

    def read(addr, desc):
        nonlocal cost
        c = read_cost(addr)
        addrs.append(addr)
        writes.append(False)
        annotations.append((addr, c, 'R', desc))
        cost += c

    def write(addr, desc):
        addrs.append(addr)
        writes.append(True)
        annotations.append((addr, 0, 'W', desc))

    for bi in range(0, N, T):
        for bj in range(0, N, T):
            # DMA: load C tile into scratchpad sC
            for ii in range(T):
                for jj in range(T):
                    read(addr_C(bi+ii, bj+jj), f"DMA C[{bi+ii}][{bj+jj}]->sC")
                    write(addr_sC(ii, jj), f"sC[{ii}][{jj}] = C[{bi+ii}][{bj+jj}]")

            for bk in range(0, N, T):
                # DMA: load A tile from args into sA
                for ii in range(T):
                    for kk in range(T):
                        read(addr_A(bi+ii, bk+kk), f"DMA A[{bi+ii}][{bk+kk}]->sA")
                        write(addr_sA(ii, kk), f"sA[{ii}][{kk}]")
                # DMA: load B tile from args into sB (AB^T layout:
                # rows of B stream contiguously, indexed by [jj][kk]).
                for jj in range(T):
                    for kk in range(T):
                        read(addr_B(bj+jj, bk+kk), f"DMA B[{bj+jj}][{bk+kk}]->sB")
                        write(addr_sB(jj, kk), f"sB[{jj}][{kk}]")
                # MAC loop: all reads from scratchpad (low positive addresses)
                for ii in range(T):
                    for jj in range(T):
                        read(addr_sC(ii, jj), f"sC[{ii}][{jj}] (acc)")
                        for kk in range(T):
                            read(addr_sA(ii, kk), f"sA[{ii}][{kk}]")
                            read(addr_sB(jj, kk), f"sB[{jj}][{kk}]")
                        write(addr_sC(ii, jj), f"sC[{ii}][{jj}] += ...")

            # Flush: read sC, write back to C
            for ii in range(T):
                for jj in range(T):
                    read(addr_sC(ii, jj), f"flush sC[{ii}][{jj}]")
                    write(addr_C(bi+ii, bj+jj), f"C[{bi+ii}][{bj+jj}]")

    regions = {
        'scratch': (sA, sC + T*T - 1),
        'out_C': (C, C + N*N - 1),
        'arg_A': (addr_A(N-1, N-1), addr_A(0, 0)),
        'arg_B': (addr_B(N-1, N-1), addr_B(0, 0)),
    }
    return addrs, writes, annotations, regions, cost


def print_trace(name, annotations, max_lines=None):
    """Print a human-readable trace of every memory access."""
    print(f"\n{'='*76}")
    print(f" {name} — Detailed Access Trace")
    print(f"{'='*76}")
    print(f"{'#':>4}  {'R/W':>3}  {'addr':>5}  {'cost':>4}  description")
    print(f"{'-'*4}  {'-'*3}  {'-'*5}  {'-'*4}  {'-'*40}")
    total = 0
    n = len(annotations) if max_lines is None else min(max_lines, len(annotations))
    for i in range(n):
        addr, c, rw, desc = annotations[i]
        total += c
        print(f"{i:>4}    {rw}  {addr:>5}  {c:>4}  {desc}")
    if max_lines is not None and max_lines < len(annotations):
        remaining_cost = sum(a[1] for a in annotations[max_lines:])
        total += remaining_cost
        print(f" ...  ({len(annotations) - max_lines} more accesses, "
              f"cost={remaining_cost:,})")
    print(f"\nTotal: {len(annotations)} accesses "
          f"({sum(1 for a in annotations if a[2] == 'R')} reads, "
          f"{sum(1 for a in annotations if a[2] == 'W')} writes), "
          f"read cost = {total:,}")


def plot_panel(ax, addrs_list, is_write_list, regions, title, cost, y_min, y_max):
    ys = np.array(addrs_list)
    ws = np.array(is_write_list)
    xs = np.arange(len(ys))

    for name, color in REGION_COLORS.items():
        if name not in regions:
            continue
        lo, hi = regions[name]
        mask_region = (ys >= lo) & (ys <= hi)
        mask_r = mask_region & ~ws
        mask_w = mask_region & ws
        if mask_r.any():
            ax.scatter(xs[mask_r], ys[mask_r], s=8, alpha=0.7, c=color,
                       label=f"{name} R ({lo}..{hi})", rasterized=True,
                       linewidths=0, marker='o')
        if mask_w.any():
            ax.scatter(xs[mask_w], ys[mask_w], s=20, alpha=0.7,
                       facecolors='none', edgecolors=color,
                       label=f"{name} W", rasterized=True,
                       linewidths=0.8, marker='v')

    n_reads = int((~ws).sum())
    n_writes = int(ws.sum())
    ax.axhline(0, color='black', lw=1, ls='-', alpha=0.4)
    ax.set_ylabel('Address (neg=args, pos=scratch)', fontsize=10)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f'{title}\n{n_reads} reads + {n_writes} writes, '
                 f'read cost = {cost:,}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.01, 0.5),
              framealpha=0.95)


def main():
    N = 16
    T = 4

    naive_addrs, naive_ws, naive_ann, naive_reg, naive_cost = trace_naive(N)
    tiled_addrs, tiled_ws, tiled_ann, tiled_reg, tiled_cost = trace_tiled(N, T)

    # Print detailed traces
    print_trace(f"Naive Matmul (N={N})", naive_ann)
    print_trace(f"Tiled Matmul with Scratchpad (N={N}, T={T})", tiled_ann)

    # Summary
    print(f'\n{"="*76}')
    print(f' Manhattan Diamond Model: N={N}, T={T}')
    print(f' Arguments (A, B) at negative addresses (read-only)')
    print(f' Scratch (sA, sB, sC, C) at positive addresses')
    print(f' Read cost = ceil(sqrt(|addr|)), writes free')
    print(f'{"="*76}')
    nr = sum(1 for w in naive_ws if not w)
    nw = sum(1 for w in naive_ws if w)
    tr = sum(1 for w in tiled_ws if not w)
    tw = sum(1 for w in tiled_ws if w)
    print(f'  Naive:  {nr} reads + {nw} writes   read cost = {naive_cost:>6,}')
    print(f'  Tiled:  {tr} reads + {tw} writes   read cost = {tiled_cost:>6,}')
    if tiled_cost > 0:
        print(f'  Speedup: {naive_cost/tiled_cost:.2f}x')

    # Plot
    all_addrs = naive_addrs + tiled_addrs
    y_min = min(all_addrs) - 2
    y_max = max(all_addrs) + 2

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    plot_panel(axes[0], naive_addrs, naive_ws, naive_reg,
              f'Naive Matmul (N={N}) — args above, scratch below',
              naive_cost, y_min, y_max)
    plot_panel(axes[1], tiled_addrs, tiled_ws, tiled_reg,
              f'Tiled with Scratchpad (N={N}, T={T}) — MAC loop in low scratch',
              tiled_cost, y_min, y_max)
    axes[1].set_xlabel('Access index', fontsize=11)

    fig.suptitle(
        f'Manhattan Diamond: Naive vs Tiled (N={N})  '
        f'[neg = arguments, pos = scratch]\n'
        f'Naive cost={naive_cost:,}  |  Tiled cost={tiled_cost:,}  |  '
        f'Ratio={naive_cost/tiled_cost:.2f}x',
        fontsize=12, y=0.99)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'manual_access_pattern.svg')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
