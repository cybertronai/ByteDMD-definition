#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Concrete-address Zero-Allocation Fused Strassen trace.

This is the direct implementation of the "efficient strassen" note:

- `run_rmm` is the tiled recursive-matmul baseline.
- `run_fused_strassen` is the zero-allocation fused Strassen schedule.

Both routines emit a physical-address trace plus the discrete sqrt-tier cost
used elsewhere in these experiments.
"""

from __future__ import annotations

import math
from pathlib import Path


class Allocator:
    """Bump-allocated 1-D physical memory with distance-cost logging."""

    def __init__(self) -> None:
        self.cost = 0
        self.ptr = 1
        self.log: list[int] = []

    def alloc(self, size: int) -> int:
        addr = self.ptr
        self.ptr += size
        return addr

    def read(self, addr: int) -> None:
        self.log.append(addr)
        self.cost += math.isqrt(max(0, addr - 1)) + 1


class ScratchpadRMM:
    """Tile-local scratchpad for the recursive matmul baseline."""

    def __init__(self, alloc: Allocator, tile: int) -> None:
        self.alloc = alloc
        self.tile = tile
        tile_area = tile * tile
        self.fast_a = alloc.alloc(tile_area)
        self.fast_b = alloc.alloc(tile_area)
        self.fast_c = alloc.alloc(tile_area)
        self.loaded_c: tuple[int, int] | None = None

    def compute_tile(
        self,
        p_a: int,
        p_b: int,
        p_c: int,
        n: int,
        r_a: int,
        c_a: int,
        r_b: int,
        c_b: int,
        r_c: int,
        c_c: int,
        *,
        is_first: bool,
    ) -> None:
        tile = self.tile
        for i in range(tile):
            for j in range(tile):
                self.alloc.read(p_a + (r_a + i) * n + c_a + j)
        for i in range(tile):
            for j in range(tile):
                self.alloc.read(p_b + (r_b + i) * n + c_b + j)

        for i in range(tile * tile):
            self.alloc.read(self.fast_c + i)
        for i in range(tile):
            for j in range(tile):
                self.alloc.read(self.fast_c + i * tile + j)
                for k in range(tile):
                    self.alloc.read(self.fast_a + i * tile + k)
                    self.alloc.read(self.fast_b + k * tile + j)

        for i in range(tile):
            for j in range(tile):
                self.alloc.read(self.fast_c + i * tile + j)
                if not is_first:
                    self.alloc.read(p_c + (r_c + i) * n + c_c + j)


def run_rmm(n: int, tile: int = 4) -> tuple[list[int], dict[str, tuple[int, int]], int]:
    """Emit the concrete trace for tiled recursive matmul."""

    alloc = Allocator()
    scratch = ScratchpadRMM(alloc, tile)
    p_a = alloc.alloc(n * n)
    p_b = alloc.alloc(n * n)
    p_c = alloc.alloc(n * n)

    def recurse(r_a: int, c_a: int, r_b: int, c_b: int, r_c: int, c_c: int, size: int) -> None:
        if size == tile:
            is_first = scratch.loaded_c != (r_c, c_c)
            scratch.loaded_c = (r_c, c_c)
            scratch.compute_tile(
                p_a,
                p_b,
                p_c,
                n,
                r_a,
                c_a,
                r_b,
                c_b,
                r_c,
                c_c,
                is_first=is_first,
            )
            return

        half = size // 2
        for dr_a, dc_a, dr_b, dc_b, dr_c, dc_c in [
            (0, 0, 0, 0, 0, 0),
            (0, 0, 0, half, 0, half),
            (half, 0, 0, half, half, half),
            (half, 0, 0, 0, half, 0),
            (half, half, half, 0, half, 0),
            (half, half, half, half, half, half),
            (0, half, half, half, 0, half),
            (0, half, half, 0, 0, 0),
        ]:
            recurse(r_a + dr_a, c_a + dc_a, r_b + dr_b, c_b + dc_b, r_c + dr_c, c_c + dc_c, half)

    recurse(0, 0, 0, 0, 0, 0, n)
    regions = {
        "scratch": (1, 3 * tile * tile),
        "main_A": (p_a, p_a + n * n - 1),
        "main_B": (p_b, p_b + n * n - 1),
        "main_C": (p_c, p_c + n * n - 1),
    }
    return alloc.log, regions, alloc.cost


class VirtualScratchpad:
    """Scratchpad that resolves Strassen sums on the fly."""

    def __init__(self, alloc: Allocator, tile: int) -> None:
        self.alloc = alloc
        self.tile = tile
        tile_area = tile * tile
        self.fast_a = alloc.alloc(tile_area)
        self.fast_b = alloc.alloc(tile_area)
        self.fast_c = alloc.alloc(tile_area)

    def compute_fused_tile(
        self,
        p_a: int,
        p_b: int,
        p_c: int,
        n: int,
        ops_a: list[tuple[int, int, int]],
        ops_b: list[tuple[int, int, int]],
        ops_c: list[tuple[int, int, int, bool]],
        r: int,
        c: int,
        *,
        k_off: int,
    ) -> None:
        tile = self.tile

        # Resolve virtual A DAG sum directly from main memory into L1.
        for i in range(tile):
            for j in range(tile):
                for _, rb, cb in ops_a:
                    self.alloc.read(p_a + (rb + r + i) * n + cb + k_off + j)

        # Resolve virtual B DAG sum directly from main memory into L1.
        for i in range(tile):
            for j in range(tile):
                for _, rb, cb in ops_b:
                    self.alloc.read(p_b + (rb + k_off + i) * n + cb + c + j)

        # Standard tile-local matmul loop.
        for i in range(tile * tile):
            self.alloc.read(self.fast_c + i)
        for i in range(tile):
            for j in range(tile):
                self.alloc.read(self.fast_c + i * tile + j)
                for k in range(tile):
                    self.alloc.read(self.fast_a + i * tile + k)
                    self.alloc.read(self.fast_b + k * tile + j)

        # Flush directly into the destination quadrants of C.
        for _, rb, cb, is_first in ops_c:
            for i in range(tile):
                for j in range(tile):
                    self.alloc.read(self.fast_c + i * tile + j)
                    if not is_first:
                        self.alloc.read(p_c + (rb + r + i) * n + cb + c + j)


def run_fused_strassen(n: int, tile: int = 4) -> tuple[list[int], dict[str, tuple[int, int]], int]:
    """Emit the concrete trace for zero-allocation fused Strassen."""

    alloc = Allocator()
    scratch = VirtualScratchpad(alloc, tile)
    p_a = alloc.alloc(n * n)
    p_b = alloc.alloc(n * n)
    p_c = alloc.alloc(n * n)

    half = n // 2
    q11, q12, q21, q22 = (0, 0), (0, half), (half, 0), (half, half)

    recipes = [
        ([(1, *q11), (1, *q22)], [(1, *q11), (1, *q22)], [(1, *q11, True), (1, *q22, True)]),
        ([(1, *q21), (1, *q22)], [(1, *q11)], [(1, *q21, True), (-1, *q22, False)]),
        ([(1, *q11)], [(1, *q12), (-1, *q22)], [(1, *q12, True), (1, *q22, False)]),
        ([(1, *q22)], [(1, *q21), (-1, *q11)], [(1, *q11, False), (1, *q21, False)]),
        ([(1, *q11), (1, *q12)], [(1, *q22)], [(-1, *q11, False), (1, *q12, False)]),
        ([(1, *q21), (-1, *q11)], [(1, *q11), (1, *q12)], [(1, *q22, False)]),
        ([(1, *q12), (-1, *q22)], [(1, *q21), (1, *q22)], [(1, *q11, False)]),
    ]

    for ops_a, ops_b, ops_c in recipes:
        for r, c in [(0, 0), (0, tile), (tile, 0), (tile, tile)]:
            scratch.compute_fused_tile(p_a, p_b, p_c, n, ops_a, ops_b, ops_c, r, c, k_off=0)
            accum_ops_c = [(sign, rb, cb, False) for sign, rb, cb, _ in ops_c]
            scratch.compute_fused_tile(p_a, p_b, p_c, n, ops_a, ops_b, accum_ops_c, r, c, k_off=tile)

    regions = {
        "scratch": (1, 3 * tile * tile),
        "main_A": (p_a, p_a + n * n - 1),
        "main_B": (p_b, p_b + n * n - 1),
        "main_C": (p_c, p_c + n * n - 1),
    }
    return alloc.log, regions, alloc.cost


def plot_trace_comparison(out_path: str | Path, n: int = 16, tile: int = 4) -> Path:
    """Render the RMM vs fused-Strassen physical-address traces."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = Path(out_path)
    r_log, r_regions, r_cost = run_rmm(n, tile)
    s_log, s_regions, s_cost = run_fused_strassen(n, tile)

    region_colors = {
        "scratch": "tab:cyan",
        "main_A": "tab:red",
        "main_B": "tab:orange",
        "main_C": "tab:purple",
    }

    def plot_panel(ax, addrs: list[int], regions: dict[str, tuple[int, int]], label: str, cost: int, y_max: int) -> None:
        ys = np.array(addrs)
        xs = np.arange(len(addrs))
        for name, color in region_colors.items():
            if name not in regions:
                continue
            lo, hi = regions[name]
            mask = (ys >= lo) & (ys <= hi)
            if mask.any():
                ax.scatter(
                    xs[mask],
                    ys[mask],
                    s=6,
                    alpha=0.55,
                    c=color,
                    label=f"{name} ({lo}..{hi})",
                    rasterized=True,
                    linewidths=0,
                )
        ax.set_ylabel("Physical address", fontsize=11)
        ax.set_ylim(0, y_max)
        ax.set_title(f"{label}  |  {len(addrs):,} accesses, cost = {cost:,}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.95)

    y_max = max(max(r_log), max(s_log)) + 50
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    plot_panel(axes[0], r_log, r_regions, f"Standard RMM (N={n}, tile={tile})", r_cost, y_max)
    plot_panel(axes[1], s_log, s_regions, f"Zero-Allocation Fused Strassen (N={n}, tile={tile})", s_cost, y_max)
    axes[1].set_xlabel("Access index", fontsize=11)
    fig.suptitle("Hardware Optimized Strassen vs RMM", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return out_path


def main() -> None:
    n = 16
    tile = 4
    out = plot_trace_comparison("fused_strassen_trace.png", n=n, tile=tile)
    _, _, r_cost = run_rmm(n, tile)
    _, _, s_cost = run_fused_strassen(n, tile)

    print(f"Saved: {out}")
    print(f"RMM Cost:             {r_cost:,}")
    print(f"Fused Strassen Cost:  {s_cost:,}")


if __name__ == "__main__":
    main()
