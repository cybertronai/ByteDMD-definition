#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Visualize the access trace of matmul_rmm_manual (flat scratchpad RMM)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import manual_matmul as mm

REGION_COLORS = {
    'fast_A': 'tab:green',
    'fast_B': 'tab:olive',
    'fast_C': 'tab:cyan',
    'main_A': 'tab:red',
    'main_B': 'tab:orange',
    'main_C': 'tab:purple',
}


def trace_rmm(N, tile_size=4):
    A = [[1] * N for _ in range(N)]
    B = [[1] * N for _ in range(N)]
    log = []
    orig = mm.ManualAllocator.read
    def logged(self, addr):
        log.append(addr)
        return orig(self, addr)
    mm.ManualAllocator.read = logged
    try:
        alloc = mm.ManualAllocator()
        sp = mm.ScratchpadRMM(alloc, N, tile_size)
        ptrA = mm._load_matrix(alloc, A)
        ptrB = mm._load_matrix(alloc, B)
        ptrC = mm._load_matrix(alloc, [[0.0]*N for _ in range(N)])

        def recurse(rA, cA, rB, cB, rC, cC, size):
            if size == tile_size:
                sp.compute_tile(ptrA, ptrB, ptrC, rA, cA, rB, cB, rC, cC)
                return
            h = size // 2
            recurse(rA, cA, rB, cB, rC, cC, h)
            recurse(rA, cA, rB, cB+h, rC, cC+h, h)
            recurse(rA+h, cA, rB, cB+h, rC+h, cC+h, h)
            recurse(rA+h, cA, rB, cB, rC+h, cC, h)
            recurse(rA+h, cA+h, rB+h, cB, rC+h, cC, h)
            recurse(rA+h, cA+h, rB+h, cB+h, rC+h, cC+h, h)
            recurse(rA, cA+h, rB+h, cB+h, rC, cC+h, h)
            recurse(rA, cA+h, rB+h, cB, rC, cC, h)

        recurse(0, 0, 0, 0, 0, 0, N)
        sp.flush_C(ptrC)
    finally:
        mm.ManualAllocator.read = orig

    T = tile_size
    regions = {
        'fast_A': (sp.fast_A, sp.fast_A + T*T - 1),
        'fast_B': (sp.fast_B, sp.fast_B + T*T - 1),
        'fast_C': (sp.fast_C, sp.fast_C + T*T - 1),
        'main_A': (ptrA, ptrA + N*N - 1),
        'main_B': (ptrB, ptrB + N*N - 1),
        'main_C': (ptrC, ptrC + N*N - 1),
    }
    return log, regions, alloc.cost


def classify(addr, regions):
    for label, (lo, hi) in regions.items():
        if lo <= addr <= hi:
            return label
    return 'other'


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    tile = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    addrs, regions, cost = trace_rmm(N, tile)

    xs = np.arange(len(addrs))
    ys = np.array(addrs)
    labels = np.array([classify(int(a), regions) for a in ys])
    per_cost = np.array([math.isqrt(int(a) - 1) + 1 for a in ys])

    fig, ax = plt.subplots(figsize=(16, 6))
    for region, color in REGION_COLORS.items():
        if region not in regions:
            continue
        mask = labels == region
        if mask.any():
            lo, hi = regions[region]
            n_acc = int(mask.sum())
            region_cost = int(per_cost[mask].sum())
            ax.scatter(xs[mask], ys[mask], s=8, alpha=0.6, c=color,
                       label=f'{region} ({lo}..{hi})  {n_acc:,} acc  cost={region_cost:,}',
                       rasterized=True, linewidths=0)

    # Annotate region bands
    for region in REGION_COLORS:
        if region in regions:
            lo, hi = regions[region]
            ax.axhspan(lo, hi, alpha=0.04,
                        color=REGION_COLORS[region])

    ax.set_xlabel('Access index', fontsize=12)
    ax.set_ylabel('Physical address', fontsize=12)
    ax.set_title(
        f'matmul_rmm_manual access trace  —  N={N}, tile={tile}\n'
        f'{len(addrs):,} accesses,  total cost ∑⌈√addr⌉ = {cost:,}   '
        f'(scratchpad addrs 1..{3*tile*tile},  main memory {3*tile*tile+1}..{max(addrs)})',
        fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5),
              framealpha=0.95)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), f'rmm_trace_n{N}.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f'Saved: {out}')

    # Summary
    print(f'\nN={N}, tile={tile}: {len(addrs):,} accesses, cost={cost:,}')
    total = len(addrs)
    for region in REGION_COLORS:
        if region in regions:
            lo, hi = regions[region]
            mask = labels == region
            n = int(mask.sum())
            rc = int(per_cost[mask].sum())
            print(f'  {region:>8s}  addrs {lo:>4}..{hi:<4}  '
                  f'{n:>6,} acc ({100*n/total:5.1f}%)  '
                  f'cost {rc:>8,} ({100*rc/cost:5.1f}%)')


if __name__ == '__main__':
    main()
