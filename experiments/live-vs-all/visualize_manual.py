#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Visualize the read-access trace of the Manual matmul implementations.

Runs both `matmul_naive_manual` and `matmul_rmm_manual` with instrumented
ManualAllocators that log every read address, then produces two side-by-side
comparison plots:

  manual_trace_n{N}.png         — address vs read-index scatter, with a
                                    per-read-cost panel underneath; naive
                                    on top, RMM on bottom.
  manual_reads_per_addr_n{N}.png — #reads-per-address histogram and
                                    per-address cost contribution; naive
                                    on top, RMM on bottom.

The total cost and per-region cost shares are annotated in each panel so
the "energy" difference between naive and RMM is immediate.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import manual_matmul as mm


# Shared colour scheme so the two algorithms are comparable at a glance.
REGION_COLORS = {
    'fast_A': 'tab:green',
    'fast_B': 'tab:olive',
    'fast_C': 'tab:cyan',
    'main_A': 'tab:red',
    'main_B': 'tab:orange',
    'main_C': 'tab:purple',
}


def _trace(run_fn, *, region_setup) -> tuple[list, dict, int]:
    """Run `run_fn(alloc)` with a logging allocator and return (addrs, regions, cost)."""
    log: list[int] = []
    orig_read = mm.ManualAllocator.read

    def logged_read(self, addr: int) -> float:
        log.append(addr)
        return orig_read(self, addr)

    mm.ManualAllocator.read = logged_read
    try:
        alloc = mm.ManualAllocator()
        regions = region_setup(alloc)
        run_fn(alloc, regions)
    finally:
        mm.ManualAllocator.read = orig_read
    return list(log), regions, alloc.cost


def trace_naive(N: int) -> tuple[list, dict, int]:
    A_in = [[1] * N for _ in range(N)]
    B_in = [[1] * N for _ in range(N)]

    def setup(alloc):
        ptrA = mm._load_matrix(alloc, A_in)
        ptrB = mm._load_matrix(alloc, B_in)
        C_in = [[0.0] * N for _ in range(N)]
        ptrC = mm._load_matrix(alloc, C_in)
        return {
            'main_A': (ptrA, ptrA + N * N - 1),
            'main_B': (ptrB, ptrB + N * N - 1),
            'main_C': (ptrC, ptrC + N * N - 1),
            '_ptrA': ptrA, '_ptrB': ptrB, '_ptrC': ptrC,
        }

    def run(alloc, regions):
        ptrA, ptrB, ptrC = regions['_ptrA'], regions['_ptrB'], regions['_ptrC']
        for i in range(N):
            for j in range(N):
                c_val = alloc.read(ptrC + i * N + j)
                for k in range(N):
                    c_val += alloc.read(ptrA + i * N + k) * alloc.read(ptrB + k * N + j)
                alloc.write(ptrC + i * N + j, c_val)

    addrs, regions, cost = _trace(run, region_setup=setup)
    for internal in ('_ptrA', '_ptrB', '_ptrC'):
        regions.pop(internal, None)
    return addrs, regions, cost


def trace_rmm(N: int, tile_size: int = 4) -> tuple[list, dict, int]:
    A_in = [[1] * N for _ in range(N)]
    B_in = [[1] * N for _ in range(N)]

    def setup(alloc):
        sp = mm.ScratchpadRMM(alloc, N, tile_size)
        ptrA = mm._load_matrix(alloc, A_in)
        ptrB = mm._load_matrix(alloc, B_in)
        C_in = [[0.0] * N for _ in range(N)]
        ptrC = mm._load_matrix(alloc, C_in)
        return {
            'fast_A': (sp.fast_A, sp.fast_A + tile_size * tile_size - 1),
            'fast_B': (sp.fast_B, sp.fast_B + tile_size * tile_size - 1),
            'fast_C': (sp.fast_C, sp.fast_C + tile_size * tile_size - 1),
            'main_A': (ptrA, ptrA + N * N - 1),
            'main_B': (ptrB, ptrB + N * N - 1),
            'main_C': (ptrC, ptrC + N * N - 1),
            '_sp': sp, '_ptrA': ptrA, '_ptrB': ptrB, '_ptrC': ptrC,
        }

    def run(alloc, regions):
        sp = regions['_sp']
        ptrA, ptrB, ptrC = regions['_ptrA'], regions['_ptrB'], regions['_ptrC']

        def recurse(rA, cA, rB, cB, rC, cC, size):
            if size == tile_size:
                sp.compute_tile(ptrA, ptrB, ptrC, rA, cA, rB, cB, rC, cC)
                return
            h = size // 2
            recurse(rA,     cA,     rB,     cB,     rC,     cC,     h)
            recurse(rA,     cA,     rB,     cB + h, rC,     cC + h, h)
            recurse(rA + h, cA,     rB,     cB + h, rC + h, cC + h, h)
            recurse(rA + h, cA,     rB,     cB,     rC + h, cC,     h)
            recurse(rA + h, cA + h, rB + h, cB,     rC + h, cC,     h)
            recurse(rA + h, cA + h, rB + h, cB + h, rC + h, cC + h, h)
            recurse(rA,     cA + h, rB + h, cB + h, rC,     cC + h, h)
            recurse(rA,     cA + h, rB + h, cB,     rC,     cC,     h)

        recurse(0, 0, 0, 0, 0, 0, N)
        sp.flush_C(ptrC)

    addrs, regions, cost = _trace(run, region_setup=setup)
    for internal in ('_sp', '_ptrA', '_ptrB', '_ptrC'):
        regions.pop(internal, None)
    return addrs, regions, cost


def classify(addr: int, regions: dict) -> str:
    for label, (lo, hi) in regions.items():
        if lo <= addr <= hi:
            return label
    return 'other'


def _plot_trace_panel(ax_scatter, ax_cost, addrs, regions, algo_label, cost, y_max):
    xs = np.arange(len(addrs))
    ys = np.array(addrs)
    labels = np.array([classify(int(a), regions) for a in ys])
    for region, color in REGION_COLORS.items():
        if region not in regions:
            continue
        mask = labels == region
        if mask.any():
            ax_scatter.scatter(xs[mask], ys[mask], s=6, alpha=0.55, c=color,
                                label=f"{region} ({regions[region][0]}..{regions[region][1]})",
                                rasterized=True, linewidths=0)
    ax_scatter.set_ylabel('Physical address', fontsize=11)
    ax_scatter.set_ylim(0, y_max)
    ax_scatter.set_title(f'{algo_label}  —  {len(addrs):,} reads,  cost ∑⌈√addr⌉ = {cost:,}',
                          fontsize=12)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5),
                       framealpha=0.95)

    per_cost = np.array([math.isqrt(int(a) - 1) + 1 for a in ys])
    ax_cost.plot(xs, per_cost, color='black', linewidth=0.3, alpha=0.5)
    ax_cost.set_ylabel('⌈√addr⌉', fontsize=10)
    ax_cost.grid(True, alpha=0.3)
    ax_cum = ax_cost.twinx()
    ax_cum.plot(xs, np.cumsum(per_cost), color='tab:red', linewidth=1.2, alpha=0.8)
    ax_cum.set_ylabel('cum cost', color='tab:red', fontsize=10)
    ax_cum.tick_params(axis='y', labelcolor='tab:red')


def plot_trace(N: int, tile_size: int, out_path: str) -> None:
    naive_addrs, naive_regions, naive_cost = trace_naive(N)
    rmm_addrs, rmm_regions, rmm_cost = trace_rmm(N, tile_size)
    y_max = max(max(naive_addrs), max(rmm_addrs)) + 1

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 11),
        gridspec_kw={'height_ratios': [3, 1, 3, 1]},
    )
    _plot_trace_panel(axes[0], axes[1], naive_addrs, naive_regions,
                       f'NAIVE triple-loop   (N={N})', naive_cost, y_max)
    _plot_trace_panel(axes[2], axes[3], rmm_addrs, rmm_regions,
                       f'RMM + scratchpad   (N={N}, tile={tile_size})',
                       rmm_cost, y_max)
    axes[3].set_xlabel('Read operation index', fontsize=11)
    fig.suptitle(f'Manual matmul read traces  —  naive vs RMM  —  '
                  f'energy ratio  naive / rmm = {naive_cost / rmm_cost:.2f}×',
                  fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()

    print(f'NAIVE  — {len(naive_addrs):,} reads, cost {naive_cost:,}')
    print(f'RMM    — {len(rmm_addrs):,} reads, cost {rmm_cost:,}')
    print(f'Energy ratio (naive / rmm)  = {naive_cost / rmm_cost:.2f}×')


def _plot_reads_per_addr_panel(ax_cnt, ax_cost_bar, addrs, regions, algo_label, cost,
                                 x_max, cnt_ymax, cost_ymax):
    max_addr = max(addrs)
    counts = np.zeros(max_addr + 1, dtype=np.int64)
    for a in addrs:
        counts[a] += 1
    xs = np.arange(1, max_addr + 1)
    per_addr_cost = counts[1:] * np.array([math.isqrt(a - 1) + 1 for a in xs])

    for region, color in REGION_COLORS.items():
        if region not in regions:
            continue
        lo, hi = regions[region]
        reg_xs = np.arange(lo, hi + 1)
        ax_cnt.bar(reg_xs, counts[lo:hi + 1], width=1.0, color=color,
                    label=f'{region}  ({lo}..{hi})', edgecolor='none', zorder=3)
        ax_cost_bar.bar(reg_xs, per_addr_cost[lo - 1:hi], width=1.0, color=color,
                         edgecolor='none', zorder=3)

    ax_cnt.set_ylabel('# reads', fontsize=11)
    ax_cnt.set_xlim(0, x_max)
    ax_cnt.set_ylim(0, cnt_ymax)
    ax_cnt.set_title(f'{algo_label}  —  {len(addrs):,} reads,  total cost = {cost:,}',
                      fontsize=12)
    ax_cnt.grid(True, axis='y', alpha=0.3)
    ax_cnt.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5),
                   framealpha=0.95)
    ax_cost_bar.set_ylabel('cost contribution', fontsize=11)
    ax_cost_bar.set_xlim(0, x_max)
    ax_cost_bar.set_ylim(0, cost_ymax)
    ax_cost_bar.grid(True, axis='y', alpha=0.3)

    # Per-region summary
    print(f'\n{algo_label}  — {len(addrs):,} reads, cost {cost:,}')
    print(f"  {'region':>8s}  {'addrs':>12s}  {'reads':>8s}  {'mean':>6s}  "
          f"{'cost':>10s}  {'%cost':>6s}")
    for region in REGION_COLORS:
        if region not in regions:
            continue
        lo, hi = regions[region]
        reg_reads = int(counts[lo:hi + 1].sum())
        reg_cost = int(per_addr_cost[lo - 1:hi].sum())
        mean_reads = reg_reads / (hi - lo + 1)
        print(f"  {region:>8s}  {lo:>5}..{hi:<5}  {reg_reads:>8,}  "
              f"{mean_reads:>6.1f}  {reg_cost:>10,}  {100 * reg_cost / cost:>5.1f}%")


def plot_reads_per_addr(N: int, tile_size: int, out_path: str) -> None:
    naive_addrs, naive_regions, naive_cost = trace_naive(N)
    rmm_addrs, rmm_regions, rmm_cost = trace_rmm(N, tile_size)

    max_naive = max(naive_addrs)
    max_rmm = max(rmm_addrs)
    x_max = max(max_naive, max_rmm) + 1

    # Compute per-addr counts/cost to set matching y-axis scales.
    def analyse(addrs):
        m = max(addrs)
        cnt = np.zeros(m + 1, dtype=np.int64)
        for a in addrs:
            cnt[a] += 1
        per_cost = cnt[1:] * np.array([math.isqrt(a - 1) + 1 for a in range(1, m + 1)])
        return cnt.max(), per_cost.max()

    naive_cnt_max, naive_cost_max = analyse(naive_addrs)
    rmm_cnt_max, rmm_cost_max = analyse(rmm_addrs)
    cnt_ymax = int(max(naive_cnt_max, rmm_cnt_max) * 1.05) + 1
    cost_ymax = int(max(naive_cost_max, rmm_cost_max) * 1.05) + 1

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=False,
                              gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    _plot_reads_per_addr_panel(axes[0], axes[1], naive_addrs, naive_regions,
                                f'NAIVE triple-loop  (N={N})',
                                naive_cost, x_max, cnt_ymax, cost_ymax)
    _plot_reads_per_addr_panel(axes[2], axes[3], rmm_addrs, rmm_regions,
                                f'RMM + scratchpad  (N={N}, tile={tile_size})',
                                rmm_cost, x_max, cnt_ymax, cost_ymax)
    axes[3].set_xlabel('Physical address', fontsize=12)
    fig.suptitle('Manual matmul:  reads & cost per physical address  —  '
                  f'energy ratio  naive / rmm = {naive_cost / rmm_cost:.2f}×',
                  fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'\nSaved: {out_path}')
    plt.close()


def main() -> None:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    tile = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    out_dir = os.path.dirname(__file__)
    plot_trace(N, tile, os.path.join(out_dir, f'manual_trace_n{N}.png'))
    plot_reads_per_addr(N, tile, os.path.join(out_dir, f'manual_reads_per_addr_n{N}.png'))


if __name__ == '__main__':
    main()
