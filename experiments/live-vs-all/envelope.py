#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Three-column comparison: Classic DMD, DMD-live, Tombstone allocator.

- Classic DMD — "Infinite Graveyard". LRU stack, no liveness compaction;
  dead variables stay in the stack forever and push live data deeper. A
  LOAD of X charges ceil(sqrt(depth)) where depth = #distinct vars
  referenced since X's previous LOAD, dead or alive. RMM: O(N^3.5).
- DMD-live — "Teleporting Cache". LRU stack with liveness compaction;
  when a variable's last LOAD fires, it vaporizes and everything above
  it slides inward for free. Depth = #live vars between X's prev and
  current LOAD. RMM: O(N^3 log N).
- Tombstone — concrete realistic allocator. Mobile LRU stack with holes:
  dead variables leave permanent tombstones at their physical address,
  LOADs bump the accessed value toward the top (filling the highest hole
  above, or extending to a new top), new STOREs also take the highest
  hole (or extend). No global compaction. RMM empirically ~ N^3 log N
  with a constant factor growing slowly in N, sitting neatly between
  DMD-live (below) and Classic DMD (above).

Reference: gemini/15apr26-dmdlive-analysis.md.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import bytedmd_ir as b2
import manual_matmul as mm


ALGORITHMS = [
    ('Cache-oblivious RMM (8-way)',  b2.matmul_rmm),
    ('Tiled (one level, tile=ceil(sqrt N))', b2.matmul_tiled),
]

# (key, display_name, color, marker, linestyle, kind)
# kind='l2' -> computed directly on L2 trace
# kind='l3' -> computed via allocator -> L3 trace
MEASURES = [
    ('mwis_lb',         'MWIS lower',      'black',      'v', ':',  'bound'),
    ('bytedmd_live',    'DMD-live',        'tab:green',  '^', '-',  'l2'),
    ('ripple',          'Ripple Shift',    'tab:purple', 'D', '-',  'l3'),
    ('manual',          'Manual',          'tab:brown',  'x', '-.', 'manual'),
    ('tombstone',       'Tombstone',       'tab:blue',   's', '--', 'l3'),
    ('bytedmd_classic', 'Classic DMD',     'tab:red',    'o', '-',  'l2'),
    ('static_ub',       'Static upper',    'dimgray',    'P', ':',  'bound'),
]


def run_one(func, N: int) -> dict:
    """Trace func on N x N inputs, return cost under every measure."""
    A, B = b2.make_inputs(N)
    l2, _ = b2.trace(func, (A, B))
    results = {'N': N}
    results['bytedmd_classic'] = b2.bytedmd_classic(l2)
    results['bytedmd_live']    = b2.bytedmd_live(l2)
    results['mwis_lb']         = b2.mwis_lower_bound(l2)
    results['static_ub']       = b2.static_upper_bound(l2)
    for key, _, _, _, _, kind in MEASURES:
        if kind == 'l3':
            l3 = b2.ALLOCATORS[key](l2)
            results[key] = b2.cost(l3)
    results['manual'] = mm.matmul_rmm_manual(A, B, tile_size=4)
    return results


def collect(Ns):
    table = {}
    for label, func in ALGORITHMS:
        rows = []
        for N in Ns:
            print(f'  {label}  N={N}', end='', flush=True)
            row = run_one(func, N)
            rows.append(row)
            print(f"  mwis_lb={row['mwis_lb']:,}  live={row['bytedmd_live']:,}"
                  f"  ripple={row['ripple']:,}  manual={row['manual']:,}"
                  f"  classic={row['bytedmd_classic']:,}  static_ub={row['static_ub']:,}")
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

        # Reference asymptotes anchored at largest N (drawn first, light).
        N_max = Ns_arr[-1]
        classic_max = rows[-1]['bytedmd_classic']
        live_max = rows[-1]['bytedmd_live']
        N3p5  = classic_max * (Ns_arr / N_max) ** 3.5
        N3logN = live_max * (Ns_arr / N_max) ** 3 * (np.log2(np.maximum(Ns_arr, 2))
                                                     / math.log2(max(N_max, 2)))
        ax.loglog(Ns_arr, N3logN, color='gray', linestyle=':',  alpha=0.45, linewidth=1.2,
                  label=r'$N^3 \log N$ reference', zorder=1)
        ax.loglog(Ns_arr, N3p5,   color='gray', linestyle='-',  alpha=0.35, linewidth=1.2,
                  label=r'$N^{3.5}$ reference', zorder=1)

        for key, legend, color, marker, linestyle, _ in MEASURES:
            ys = np.array([r[key] for r in rows], dtype=float)
            ax.loglog(Ns_arr, ys, color=color, marker=marker, linestyle=linestyle,
                      linewidth=2.2, markersize=9, label=legend, zorder=3)

        ax.set_xlabel('Matrix size N', fontsize=12)
        ax.set_ylabel('Total cost', fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.95)

    fig.suptitle('Classic DMD  vs  DMD-live  vs  Tombstone (concrete allocator)\n'
                 'L1 → L2 (abstract IR) → L3 (concrete addrs)',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def plot_ratios(table, Ns, out_path):
    """Show classic / live ratio (the 'envelope width') as a function of N."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, _ in ALGORITHMS:
        rows = table[label]
        Ns_arr = np.array([r['N'] for r in rows])
        ratios = np.array([r['bytedmd_classic'] / r['bytedmd_live'] for r in rows])
        ax.plot(Ns_arr, ratios, marker='o', linewidth=2, markersize=7, label=label)
    Ns_arr = np.array(Ns)
    ref = np.sqrt(Ns_arr.astype(float)) / np.log2(np.maximum(Ns_arr, 2))
    ref = ref / ref[0] * (table[ALGORITHMS[0][0]][0]['bytedmd_classic'] /
                          table[ALGORITHMS[0][0]][0]['bytedmd_live'])
    ax.plot(Ns_arr, ref, color='gray', linestyle=':', alpha=0.5,
            label=r'$\sqrt{N}/\log N$ reference')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Matrix size N', fontsize=11)
    ax.set_ylabel('ByteDMD-classic / ByteDMD-live', fontsize=11)
    ax.set_title('Envelope width on L2 metrics', fontsize=12)
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
        Ns = [4, 8, 16, 32, 64]
    table = collect(Ns)

    out_dir = os.path.dirname(__file__)
    plot(table, Ns, os.path.join(out_dir, 'envelope.png'))
    plot_ratios(table, Ns, os.path.join(out_dir, 'envelope_ratio.png'))

    # Comparison table — columns are all measures.
    print()
    display_names = [legend for _, legend, _, _, _, _ in MEASURES]
    col_w = max(16, max(len(n) for n in display_names))

    for label, _ in ALGORITHMS:
        print(f'\n{label}')
        rows = table[label]
        header = f"  {'N':>3} | " + " | ".join(n.rjust(col_w) for n in display_names)
        print(header)
        print('  ' + '-' * (len(header) - 2))
        for r in rows:
            cells = " | ".join(f"{r[m[0]]:>{col_w},}" for m in MEASURES)
            print(f"  {r['N']:>3} | {cells}")

        print()
        print(f"  Envelope ratio  classic/live  per N:")
        print(f"  " + "  ".join(f"N={r['N']}: {r['bytedmd_classic'] / r['bytedmd_live']:.2f}x"
                                   for r in rows))


if __name__ == '__main__':
    main()
