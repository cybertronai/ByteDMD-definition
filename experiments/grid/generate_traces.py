#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Emit a memory-trace PNG for every algorithm in run_grid.ALGOS.

For each algorithm, swap in a logging Allocator, call the manual impl,
harvest the full address sequence, and plot access_index (x) vs
address (y). Writes traces/<slug>.png and prints a one-line summary.
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import manual as man
import run_grid as rg


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def plot_trace(log: list[tuple[str, int]],
               writes: list[tuple[int, int]],
               output_writes: list[tuple[int, int]],
               scratch_peak: int,
               arg_peak: int,
               title: str,
               out_path: str) -> None:
    """Two-stack visualization rendered as a high-DPI antialiased PNG.
    Scratch-stack reads (blue), scratch writes (orange), and output
    writes (red) plot at their real scratch addresses. Output-epilogue
    reads (purple) show the final pass that reads every output cell
    into the return buffer. Arg-stack reads (green) plot shifted DOWN
    by negating their address so they sit below y=0."""
    n = len(log)
    arg_pts_t, arg_pts_y = [], []
    scr_pts_t, scr_pts_y = [], []
    out_pts_t, out_pts_y = [], []
    for t, (space, addr) in enumerate(log):
        if space == "arg":
            arg_pts_t.append(t)
            arg_pts_y.append(-addr)
        elif space == "output":
            out_pts_t.append(t)
            out_pts_y.append(addr)
        else:
            scr_pts_t.append(t)
            scr_pts_y.append(addr)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    # rasterized=False keeps all markers as antialiased vector primitives
    # in the SVG output.
    if scr_pts_t:
        ax.scatter(scr_pts_t, scr_pts_y,
                   s=0.8, c="tab:blue", alpha=0.55,
                   rasterized=True, linewidths=0,
                   label="scratch read")
    if arg_pts_t:
        ax.scatter(arg_pts_t, arg_pts_y,
                   s=0.8, c="tab:green", alpha=0.55,
                   rasterized=True, linewidths=0,
                   label="arg read (shifted -addr)")
    if out_pts_t:
        ax.scatter(out_pts_t, out_pts_y,
                   s=3.0, c="#8B008B", alpha=0.9,
                   rasterized=True, linewidths=0,
                   zorder=5,
                   label="output read (epilogue)")
    if writes:
        wt, wa = zip(*writes)
        ax.scatter(wt, wa,
                   s=1.2, c="tab:orange", alpha=0.65,
                   rasterized=True, linewidths=0,
                   label="scratch write")
    if output_writes:
        wt, wa = zip(*output_writes)
        ax.scatter(wt, wa,
                   s=1.2, c="tab:red", alpha=0.75,
                   rasterized=True, linewidths=0,
                   label="output write")
    if arg_pts_t:
        ax.axhline(0, color="gray", linestyle="--",
                   linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Physical address (scratch positive / arg negative)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if n > 0 or writes or output_writes:
        ax.legend(loc="upper left", markerscale=8, fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    traces_dir = os.path.join(HERE, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    print(f"{'algorithm':<40} {'arg_r':>8} {'scr_r':>8} {'out_r':>8} {'scr_w':>8} {'out_w':>8} {'cost':>12}  file")
    print("-" * 122)
    for name, _fn, _args, manual_fn in rg.ALGOS:
        slug = slugify(name)
        logged = man.Allocator(logging=True)
        man.set_allocator(logged)
        try:
            manual_fn()
        finally:
            man.set_allocator(None)

        out_path = os.path.join(traces_dir, f"{slug}.png")
        title = f"{name}  —  cost = {logged.cost:,}"
        plot_trace(logged.log, logged.writes, logged.output_writes,
                   logged.peak, logged.arg_peak, title, out_path)
        rel = os.path.relpath(out_path, HERE)
        n_arg = sum(1 for space, _ in logged.log if space == "arg")
        n_out = sum(1 for space, _ in logged.log if space == "output")
        n_scr = len(logged.log) - n_arg - n_out
        print(f"{name:<40} {n_arg:>8,} {n_scr:>8,} {n_out:>8,} "
              f"{len(logged.writes):>8,} {len(logged.output_writes):>8,} "
              f"{logged.cost:>12,}  {rel}")


if __name__ == "__main__":
    main()
