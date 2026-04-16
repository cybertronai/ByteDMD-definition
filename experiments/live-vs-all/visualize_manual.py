#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Visualize the read-access trace of the Manual RMM implementation.

Scatter plot of (read_index, physical_address) with points colored by
which buffer they belong to: scratchpad tiles (fast_A, fast_B, fast_C)
in the lowest addresses, and main-memory buffers (A, B, C) stacked above.
Region boundaries are marked, and per-point cost = ceil(sqrt(addr)).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

import manual_matmul as mm


def trace_reads(N: int, tile_size: int = 4):
    """Run matmul_rmm_manual while capturing every read address.

    Returns:
        log         : list of addresses read, in order
        region_info : dict mapping region label -> (start_addr, end_addr)
    """
    addrs = []
    orig_read = mm.ManualAllocator.read

    def logged_read(self, addr):
        addrs.append(addr)
        return orig_read(self, addr)

    # Monkey-patch the allocator, then instantiate via the public entry point.
    mm.ManualAllocator.read = logged_read
    try:
        # We need the allocator to expose its final region layout; rerun the
        # matmul manually to learn the base addresses.
        alloc = mm.ManualAllocator()
        sp = mm.ScratchpadRMM(alloc, N, tile_size)
        regions = {
            'fast_A':    (sp.fast_A,  sp.fast_A  + tile_size * tile_size - 1),
            'fast_B':    (sp.fast_B,  sp.fast_B  + tile_size * tile_size - 1),
            'fast_C':    (sp.fast_C,  sp.fast_C  + tile_size * tile_size - 1),
        }
        A = [[1] * N for _ in range(N)]
        B = [[1] * N for _ in range(N)]
        ptrA = mm._load_matrix(alloc, A)
        ptrB = mm._load_matrix(alloc, B)
        C = [[0.0] * N for _ in range(N)]
        ptrC = mm._load_matrix(alloc, C)
        regions['main_A'] = (ptrA, ptrA + N * N - 1)
        regions['main_B'] = (ptrB, ptrB + N * N - 1)
        regions['main_C'] = (ptrC, ptrC + N * N - 1)

        # Now run the real matmul; the monkey-patched read will record all accesses.
        addrs.clear()
        cost = mm.matmul_rmm_manual(A, B, tile_size=tile_size)
    finally:
        mm.ManualAllocator.read = orig_read

    return addrs, regions, cost


def classify(addr: int, regions: dict) -> str:
    for label, (lo, hi) in regions.items():
        if lo <= addr <= hi:
            return label
    return 'other'


def plot_trace(N: int, tile_size: int = 4, out_path: str = 'manual_trace.png'):
    addrs, regions, cost = trace_reads(N, tile_size)
    xs = np.arange(len(addrs))
    ys = np.array(addrs)

    region_order = ['fast_A', 'fast_B', 'fast_C', 'main_A', 'main_B', 'main_C']
    region_colors = {
        'fast_A': 'tab:green',
        'fast_B': 'tab:olive',
        'fast_C': 'tab:cyan',
        'main_A': 'tab:red',
        'main_B': 'tab:orange',
        'main_C': 'tab:purple',
    }
    labels = np.array([classify(int(a), regions) for a in ys])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
    )

    # Top panel: address trace
    for region in region_order:
        mask = labels == region
        if mask.any():
            ax_top.scatter(xs[mask], ys[mask], s=6, alpha=0.55,
                            c=region_colors[region],
                            label=f"{region}  ({regions[region][0]}..{regions[region][1]})",
                            rasterized=True, linewidths=0)

    # Mark region boundaries with horizontal dotted lines
    for region in region_order:
        lo, hi = regions[region]
        ax_top.axhline(lo, color=region_colors[region], linestyle=':',
                       alpha=0.2, linewidth=0.8)
    ax_top.set_ylabel('Physical address', fontsize=12)
    ax_top.set_title(
        f'Manual RMM read-access trace — N={N}, tile={tile_size}\n'
        f'Total reads = {len(addrs):,},   total cost '
        f'∑⌈√addr⌉ = {cost:,}',
        fontsize=13,
    )
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5),
                   framealpha=0.95)

    # Bottom panel: per-read cost
    per_cost = np.array([math.isqrt(int(a) - 1) + 1 for a in ys])
    ax_bot.plot(xs, per_cost, color='black', linewidth=0.4, alpha=0.5)
    ax_bot.set_xlabel('Read operation index', fontsize=12)
    ax_bot.set_ylabel('⌈√addr⌉  (cost)', fontsize=12)
    ax_bot.set_title('Per-read cost', fontsize=11)
    ax_bot.grid(True, alpha=0.3)

    # Running total on a twin axis so readers can see the cost accumulate.
    ax_cum = ax_bot.twinx()
    ax_cum.plot(xs, np.cumsum(per_cost), color='tab:red', linewidth=1.2,
                 alpha=0.8, label='cumulative cost')
    ax_cum.set_ylabel('Cumulative cost', color='tab:red', fontsize=11)
    ax_cum.tick_params(axis='y', labelcolor='tab:red')

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'Saved: {out_path}   ({len(addrs):,} reads, cost {cost:,})')
    plt.close()

    # Brief text summary: how many reads per region.
    total = len(addrs)
    print('Per-region read counts:')
    for region in region_order:
        n = int((labels == region).sum())
        if n:
            print(f'  {region:>8s} : {n:>7,}  ({100*n/total:5.1f} %)  addrs {regions[region][0]}..{regions[region][1]}')


def plot_reads_per_addr(N: int, tile_size: int = 4,
                         out_path: str = 'manual_reads_per_addr.png'):
    """Bar chart of read-count vs physical address."""
    addrs, regions, cost = trace_reads(N, tile_size)
    max_addr = max(addrs)
    counts = np.zeros(max_addr + 1, dtype=np.int64)
    for a in addrs:
        counts[a] += 1

    region_colors = {
        'fast_A': 'tab:green',
        'fast_B': 'tab:olive',
        'fast_C': 'tab:cyan',
        'main_A': 'tab:red',
        'main_B': 'tab:orange',
        'main_C': 'tab:purple',
    }
    region_order = ['fast_A', 'fast_B', 'fast_C', 'main_A', 'main_B', 'main_C']

    # Per-addr cost contribution = count * ceil(sqrt(addr))
    xs = np.arange(1, max_addr + 1)
    per_addr_cost = counts[1:] * np.array([math.isqrt(a - 1) + 1 for a in xs])

    fig, (ax_cnt, ax_cost) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={'height_ratios': [1, 1]},
    )

    # Top panel: read count per address, colored by region
    # Draw each region as its own bar series so the legend is clean.
    for region in region_order:
        lo, hi = regions[region]
        reg_xs = np.arange(lo, hi + 1)
        reg_ys = counts[lo:hi + 1]
        ax_cnt.bar(reg_xs, reg_ys, width=1.0, color=region_colors[region],
                    label=f'{region}  (addrs {lo}..{hi})',
                    edgecolor='none', zorder=3)
    ax_cnt.set_ylabel('# reads', fontsize=12)
    ax_cnt.set_title(
        f'Manual RMM read count per physical address — '
        f'N={N}, tile={tile_size}\n'
        f'total reads = {len(addrs):,},   '
        f'total cost ∑⌈√addr⌉ = {cost:,}',
        fontsize=13,
    )
    ax_cnt.grid(True, axis='y', alpha=0.3)
    ax_cnt.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5),
                   framealpha=0.95)

    # Bottom panel: cost contribution per address
    for region in region_order:
        lo, hi = regions[region]
        reg_xs = np.arange(lo, hi + 1)
        reg_cost = per_addr_cost[lo - 1:hi]
        ax_cost.bar(reg_xs, reg_cost, width=1.0, color=region_colors[region],
                     edgecolor='none', zorder=3)
    ax_cost.set_xlabel('Physical address', fontsize=12)
    ax_cost.set_ylabel('Cost contribution  (count · ⌈√addr⌉)', fontsize=12)
    ax_cost.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()

    print('Per-region summary:')
    print(f"  {'region':>8s}  {'addrs':>12s}  {'reads':>8s}  {'mean':>6s}  "
          f"{'cost':>10s}  {'%cost':>6s}")
    for region in region_order:
        lo, hi = regions[region]
        reg_reads = int(counts[lo:hi + 1].sum())
        reg_cost = int(per_addr_cost[lo - 1:hi].sum())
        mean_reads = reg_reads / (hi - lo + 1)
        print(f"  {region:>8s}  {lo:>5}..{hi:<5}  {reg_reads:>8,}  "
              f"{mean_reads:>6.1f}  {reg_cost:>10,}  {100 * reg_cost / cost:>5.1f}%")


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    tile = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    out_dir = os.path.dirname(__file__)
    plot_trace(N=N, tile_size=tile,
                out_path=os.path.join(out_dir, f'manual_trace_n{N}.png'))
    plot_reads_per_addr(N=N, tile_size=tile,
                        out_path=os.path.join(out_dir,
                                              f'manual_reads_per_addr_n{N}.png'))


if __name__ == '__main__':
    main()
