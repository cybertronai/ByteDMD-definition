#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Live-bytes vs all-bytes ByteDMD envelope.

For each algorithm (cache-oblivious RMM and one-level tiled matmul), trace the
function at L1, compile with each L3 allocator policy, and plot total cost vs
matrix size N on log-log axes.

The two ByteDMD endpoints bracket every concrete register-allocation strategy:
  upper:  no_reuse (every variable gets its own slot — "memory leak")
  lower:  min_heap (live-bytes greedy reuse — closest to a real compiler)

Asymptotic check on RMM:
  no_reuse   ~ N^3.5   (master theorem with addition step at the root)
  min_heap   ~ N^3 log N
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import bytedmd2 as b2


ALGORITHMS = [
    ('Cache-oblivious RMM (8-way)',  b2.matmul_rmm),
    ('Tiled (one level, tile=ceil(sqrt N))', b2.matmul_tiled),
]

POLICIES = [
    ('no_reuse',   'ByteDMD-classic',               'tab:red',    'o', '-'),
    ('lru_static', 'LIFO slots',                    'tab:orange', 's', '--'),
    ('belady',     'Belady (offline)',              'tab:blue',   'd', ':'),
    ('min_heap',   'ByteDMD-live',                  'tab:green',  '^', '-'),
]


def run_one(func, N: int) -> dict:
    """Trace func on N x N inputs, return cost under every policy + #loads."""
    A, B = b2.make_inputs(N)
    l2, _ = b2.trace(func, (A, B))
    results = {'N': N, 'n_loads': b2.n_loads(l2)}
    for policy_name, _, _, _, _ in POLICIES:
        l3 = b2.ALLOCATORS[policy_name](l2)
        results[policy_name] = b2.cost(l3)
        results[f'{policy_name}_peak'] = b2.peak_addr(l3)
    return results


def collect(Ns):
    table = {}
    for label, func in ALGORITHMS:
        rows = []
        for N in Ns:
            print(f'  {label}  N={N}  ', end='', flush=True)
            row = run_one(func, N)
            rows.append(row)
            print(f"loads={row['n_loads']:,}, no_reuse={row['no_reuse']:,}, min_heap={row['min_heap']:,}")
        table[label] = rows
    return table


def plot(table, Ns, out_path):
    fig, axes = plt.subplots(len(ALGORITHMS), 1, figsize=(10, 6.5 * len(ALGORITHMS)),
                              sharex=True)
    if len(ALGORITHMS) == 1:
        axes = [axes]

    for ax, (label, _) in zip(axes, ALGORITHMS):
        rows = table[label]
        Ns_arr = np.array([r['N'] for r in rows])

        # Reference asymptotes anchored at the largest N (drawn first, light)
        N_max = Ns_arr[-1]
        no_reuse_max = rows[-1]['no_reuse']
        min_heap_max = rows[-1]['min_heap']
        N3p5 = no_reuse_max * (Ns_arr / N_max) ** 3.5
        N3logN = min_heap_max * (Ns_arr / N_max) ** 3 * (np.log2(np.maximum(Ns_arr, 2)) /
                                                          math.log2(max(N_max, 2)))
        ax.loglog(Ns_arr, N3p5, color='gray', linestyle='-', alpha=0.4, linewidth=1.2,
                   label=r'$N^{3.5}$ reference', zorder=1)
        ax.loglog(Ns_arr, N3logN, color='gray', linestyle=':', alpha=0.5, linewidth=1.2,
                   label=r'$N^3 \log N$ reference', zorder=1)

        # Allocator policies on top
        for policy_name, legend, color, marker, linestyle in POLICIES:
            ys = np.array([r[policy_name] for r in rows], dtype=float)
            ax.loglog(Ns_arr, ys, color=color, marker=marker, linestyle=linestyle,
                      linewidth=2.2, markersize=9, label=legend, zorder=3)

        ax.set_xlabel('Matrix size N', fontsize=12)
        ax.set_ylabel('Total ByteDMD cost', fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.95)

    fig.suptitle('Live-bytes vs all-bytes ByteDMD envelope\n'
                 'L1 → L2 (abstract IR) → L3 (concrete addrs);   cost = Σ ⌈√addr⌉',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def plot_ratios(table, Ns, out_path):
    """Show no_reuse / min_heap ratio (the 'envelope width') as a function of N."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, _ in ALGORITHMS:
        rows = table[label]
        Ns_arr = np.array([r['N'] for r in rows])
        ratios = np.array([r['no_reuse'] / r['min_heap'] for r in rows])
        ax.plot(Ns_arr, ratios, marker='o', linewidth=2, markersize=7, label=label)
    # Reference: log N grows as the predicted slack between N^3.5 and N^3 log N is sqrt(N)/log N
    Ns_arr = np.array(Ns)
    ref = np.sqrt(Ns_arr.astype(float)) / np.log2(np.maximum(Ns_arr, 2))
    ref = ref / ref[0] * (table[ALGORITHMS[0][0]][0]['no_reuse'] /
                          table[ALGORITHMS[0][0]][0]['min_heap'])
    ax.plot(Ns_arr, ref, color='gray', linestyle=':', alpha=0.5,
            label=r'$\sqrt{N}/\log N$ reference')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Matrix size N', fontsize=11)
    ax.set_ylabel('Cost ratio:  no_reuse / min_heap', fontsize=11)
    ax.set_title('Envelope width (all-bytes / live-bytes)', fontsize=12)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def main():
    Ns_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if Ns_arg:
        Ns = [int(x) for x in Ns_arg.split(',')]
    else:
        Ns = [2, 4, 8, 16, 32]
    table = collect(Ns)

    out_dir = os.path.dirname(__file__)
    plot(table, Ns, os.path.join(out_dir, 'envelope.png'))
    plot_ratios(table, Ns, os.path.join(out_dir, 'envelope_ratio.png'))

    # Print a tidy comparison table — columns are named policies.
    print()
    display_names = [legend for _, legend, _, _, _ in POLICIES]
    col_w = max(16, max(len(n) for n in display_names))

    for label, _ in ALGORITHMS:
        print(f'\n{label}')
        rows = table[label]
        header = f"  {'N':>3} | " + " | ".join(n.rjust(col_w) for n in display_names)
        print(header)
        print('  ' + '-' * (len(header) - 2))
        for r in rows:
            cells = " | ".join(f"{r[p[0]]:>{col_w},}" for p in POLICIES)
            print(f"  {r['N']:>3} | {cells}")

        # Ratios row: each policy relative to ByteDMD-live
        classic_key = 'no_reuse'  # ByteDMD-classic
        live_key = 'min_heap'     # ByteDMD-live
        print()
        print(f"  Envelope ratio  classic/live  per N:")
        print(f"  " + "  ".join(f"N={r['N']}: {r[classic_key] / r[live_key]:.2f}x"
                                   for r in rows))


if __name__ == '__main__':
    main()
