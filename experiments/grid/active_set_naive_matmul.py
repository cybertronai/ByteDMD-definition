#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib"]
# ///
"""Plot the size of the live working set over time for naive_matmul(n=16)
under ByteDMD (live-compacted LRU) semantics. Walks the L2 trace and
records how many variables currently live on the geometric stack after
each event. Under two-stack pricing, input vars are not on the geom
stack until their first read promotes them."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bytedmd_ir import L2Load, L2Store, trace
import algorithms as alg


def live_set_timeline(events, input_vars):
    """Return (times, sizes) — how many vars live on the geom stack right
    after each event, under ByteDMD-live (first-touch promotion for
    inputs + drop-on-last-load for every var)."""
    input_set = set(input_vars)
    pending_args = set(input_vars)

    last_load = {}
    for i, ev in enumerate(events):
        if isinstance(ev, L2Load):
            last_load[ev.var] = i

    live = set()
    times, sizes = [], []
    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            if ev.var in last_load:  # skip never-read vars
                live.add(ev.var)
        elif isinstance(ev, L2Load):
            if ev.var in pending_args:
                pending_args.discard(ev.var)
                if last_load.get(ev.var) != i:
                    live.add(ev.var)  # promoted onto geom
            # If this is the last Load, drop from geom.
            if last_load.get(ev.var) == i:
                live.discard(ev.var)
        times.append(i)
        sizes.append(len(live))
    return times, sizes


def main():
    n = 16
    events, input_vars = trace(alg.matmul_naive_abt,
                               ([[1.0] * n for _ in range(n)],
                                [[1.0] * n for _ in range(n)]))

    times, sizes = live_set_timeline(events, input_vars)
    print(f"events={len(events):,}  peak live set={max(sizes):,}  "
          f"final live set={sizes[-1]:,}")

    out_path = os.path.join(HERE, "traces", "naive_matmul_n_16_liveset.svg")
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(times, sizes, color="tab:blue", linewidth=0.8,
            drawstyle="steps-post", rasterized=False)
    ax.fill_between(times, 0, sizes, color="tab:blue", alpha=0.18,
                    linewidth=0, step="post", rasterized=False)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Live variables on geom stack")
    ax.set_title(f"naive_matmul(n={n}) — ByteDMD live working-set size "
                 f"(peak = {max(sizes):,})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(events))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
