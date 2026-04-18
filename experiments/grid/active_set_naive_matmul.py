#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib"]
# ///
"""Two naive_matmul(n=16) diagnostics under ByteDMD-live + two-stack:

  1. Working-set size over time — how many vars currently live on the
     geometric stack after each L2 event.
  2. Reuse distance over time — the LRU depth at which each L2Load
     finds its variable (arg-stack position for first reads of inputs,
     geom-stack depth for everything else). This is exactly the per-
     access d² term paid under the sqrt(d) cost model."""
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


def reuse_distance_timeline(events, input_vars):
    """Return (times, distances). For each L2Load at index i, distance
    is:
      - The arg-stack position (1-based input order) if this is the
        first read of an input — pricing against the arg stack.
      - Otherwise the LRU depth on the geom stack: 1 + (# vars with
        fresher timestamps). LRU-bump to the top happens after.
    Uses the same semantics as bytedmd_ir._lru_cost(compact=True)."""
    from bytedmd_ir import L2Load as _L2Load, L2Store as _L2Store
    input_arg_idx = {v: i + 1 for i, v in enumerate(input_vars)}
    pending = set(input_arg_idx)

    last_load = {}
    for i, ev in enumerate(events):
        if isinstance(ev, _L2Load):
            last_load[ev.var] = i

    # Stack order by timestamp: newer ts = shallower. Depth computed
    # as rank from the newest end.
    ts_of = {}   # var -> timestamp
    live_ts = set()  # currently-live timestamps
    next_ts = 0

    # Fenwick tree over timestamps — same structure as bytedmd_ir._lru_cost.
    T = len(events) + len(input_arg_idx) + 2
    bit = [0] * (T + 1)

    def bit_add(i, delta):
        while i <= T:
            bit[i] += delta
            i += i & -i

    def bit_prefix(i):
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    times, distances = [], []
    for i, ev in enumerate(events):
        if isinstance(ev, _L2Store):
            if ev.var not in last_load:
                continue
            next_ts += 1
            ts_of[ev.var] = next_ts
            bit_add(next_ts, 1)
        elif isinstance(ev, _L2Load):
            if ev.var in pending:
                pending.discard(ev.var)
                d = input_arg_idx[ev.var]
                times.append(i); distances.append(d)
                if last_load.get(ev.var) != i:
                    next_ts += 1
                    ts_of[ev.var] = next_ts
                    bit_add(next_ts, 1)
                continue
            t = ts_of[ev.var]
            # depth = # live ts with ts' >= t = total_live - prefix(t-1)
            total_live = bit_prefix(T)
            depth = total_live - bit_prefix(t - 1)
            times.append(i); distances.append(depth)
            bit_add(t, -1)
            if last_load[ev.var] == i:
                del ts_of[ev.var]
            else:
                next_ts += 1
                ts_of[ev.var] = next_ts
                bit_add(next_ts, 1)
    return times, distances


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

    rtimes, rdist = reuse_distance_timeline(events, input_vars)
    print(f"loads={len(rtimes):,}  max reuse distance={max(rdist):,}  "
          f"median={sorted(rdist)[len(rdist)//2]:,}")

    out_path2 = os.path.join(HERE, "traces",
                             "naive_matmul_n_16_reuse_distance.svg")
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.scatter(rtimes, rdist, s=0.8, c="tab:purple", alpha=0.35,
               linewidths=0, rasterized=False)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Reuse distance (LRU depth at read)")
    ax.set_title(f"naive_matmul(n={n}) — ByteDMD reuse distance per load "
                 f"(max = {max(rdist):,})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(events))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path2, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path2}")


if __name__ == "__main__":
    main()
