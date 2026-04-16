#!/usr/bin/env python3
"""Self-contained script to generate manual_trace_n16.png.

Includes all the code inline: ManualAllocator, ScratchpadRMM, naive
matmul, RMM matmul, tracing, and plotting. No external imports besides
matplotlib/numpy. Run:

    pip install matplotlib numpy   # if not already installed
    python manual_trace_n16-standalone.py
"""

import os, subprocess, sys

def _ensure_deps():
    """Auto-create a venv and re-exec if matplotlib/numpy are missing."""
    try:
        import matplotlib, numpy  # noqa: F401
        return
    except ImportError:
        pass
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".standalone-venv")
    venv_python = os.path.join(venv_dir, "bin", "python")
    if not os.path.exists(venv_python):
        print(f"Creating venv at {venv_dir} and installing matplotlib+numpy...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        subprocess.check_call([venv_python, "-m", "pip", "install", "-q",
                               "matplotlib", "numpy"])
    os.execv(venv_python, [venv_python] + sys.argv)

_ensure_deps()

import math
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# 1. Manual Memory Hardware Model
# ============================================================================

class ManualAllocator:
    """Bump-allocated 1-D physical memory.
    - alloc(size) returns a base address.
    - write(addr, val) is free.
    - read(addr) charges ceil(sqrt(addr)).
    """
    def __init__(self):
        self.memory = {}
        self.cost = 0
        self.next_addr = 1

    def alloc(self, size: int) -> int:
        addr = self.next_addr
        self.next_addr += size
        return addr

    def write(self, addr: int, val: float) -> None:
        self.memory[addr] = val

    def read(self, addr: int) -> float:
        self.cost += math.isqrt(addr - 1) + 1
        return self.memory.get(addr, 0.0)


def load_matrix(alloc: ManualAllocator, M: List[List[float]]) -> int:
    n = len(M)
    ptr = alloc.alloc(n * n)
    for i in range(n):
        for j in range(n):
            alloc.write(ptr + i * n + j, M[i][j])
    return ptr


# ============================================================================
# 2. Software-Managed L1 Scratchpad
# ============================================================================

class ScratchpadRMM:
    """Pins three T x T tiles (A, B, C) at the lowest addresses (1..3T^2).
    Reads inside the scratchpad cost at most ceil(sqrt(3T^2))."""

    def __init__(self, alloc: ManualAllocator, n: int, tile_size: int):
        self.alloc = alloc
        self.n = n
        self.T = tile_size
        self.fast_A = alloc.alloc(self.T * self.T)
        self.fast_B = alloc.alloc(self.T * self.T)
        self.fast_C = alloc.alloc(self.T * self.T)
        self.loaded_A: Optional[tuple] = None
        self.loaded_B: Optional[tuple] = None
        self.loaded_C: Optional[tuple] = None
        self.dirty_C = False

    def sync_A(self, ptrA, rA, cA):
        if self.loaded_A == (rA, cA): return
        for i in range(self.T):
            for j in range(self.T):
                val = self.alloc.read(ptrA + (rA + i) * self.n + (cA + j))
                self.alloc.write(self.fast_A + i * self.T + j, val)
        self.loaded_A = (rA, cA)

    def sync_B(self, ptrB, rB, cB):
        if self.loaded_B == (rB, cB): return
        for i in range(self.T):
            for j in range(self.T):
                val = self.alloc.read(ptrB + (rB + i) * self.n + (cB + j))
                self.alloc.write(self.fast_B + i * self.T + j, val)
        self.loaded_B = (rB, cB)

    def sync_C(self, ptrC, rC, cC):
        if self.loaded_C == (rC, cC): return
        self.flush_C(ptrC)
        for i in range(self.T):
            for j in range(self.T):
                val = self.alloc.read(ptrC + (rC + i) * self.n + (cC + j))
                self.alloc.write(self.fast_C + i * self.T + j, val)
        self.loaded_C = (rC, cC)
        self.dirty_C = False

    def flush_C(self, ptrC):
        if self.loaded_C is not None and self.dirty_C:
            rC, cC = self.loaded_C
            for i in range(self.T):
                for j in range(self.T):
                    val = self.alloc.read(self.fast_C + i * self.T + j)
                    self.alloc.write(ptrC + (rC + i) * self.n + (cC + j), val)
            self.dirty_C = False

    def compute_tile(self, ptrA, ptrB, ptrC, rA, cA, rB, cB, rC, cC):
        self.sync_A(ptrA, rA, cA)
        self.sync_B(ptrB, rB, cB)
        self.sync_C(ptrC, rC, cC)
        for i in range(self.T):
            for j in range(self.T):
                c_val = self.alloc.read(self.fast_C + i * self.T + j)
                for k in range(self.T):
                    a_val = self.alloc.read(self.fast_A + i * self.T + k)
                    b_val = self.alloc.read(self.fast_B + k * self.T + j)
                    c_val += a_val * b_val
                self.alloc.write(self.fast_C + i * self.T + j, c_val)
        self.dirty_C = True


# ============================================================================
# 3. Matmul implementations
# ============================================================================

def run_naive(N: int):
    """Naive triple for-loop. Returns (access_log, regions, cost)."""
    A_in = [[1] * N for _ in range(N)]
    B_in = [[1] * N for _ in range(N)]

    alloc = ManualAllocator()
    ptrA = load_matrix(alloc, A_in)
    ptrB = load_matrix(alloc, B_in)
    ptrC = load_matrix(alloc, [[0.0] * N for _ in range(N)])

    log = []
    orig_read = ManualAllocator.read
    def logged_read(self, addr):
        log.append(addr)
        return orig_read(self, addr)
    ManualAllocator.read = logged_read

    for i in range(N):
        for j in range(N):
            c_val = alloc.read(ptrC + i * N + j)
            for k in range(N):
                c_val += alloc.read(ptrA + i * N + k) * alloc.read(ptrB + k * N + j)
            alloc.write(ptrC + i * N + j, c_val)

    ManualAllocator.read = orig_read

    regions = {
        'main_A': (ptrA, ptrA + N * N - 1),
        'main_B': (ptrB, ptrB + N * N - 1),
        'main_C': (ptrC, ptrC + N * N - 1),
    }
    return log, regions, alloc.cost


def run_rmm(N: int, tile_size: int = 4):
    """RMM with scratchpad. Returns (access_log, regions, cost)."""
    A_in = [[1] * N for _ in range(N)]
    B_in = [[1] * N for _ in range(N)]

    alloc = ManualAllocator()
    sp = ScratchpadRMM(alloc, N, tile_size)
    ptrA = load_matrix(alloc, A_in)
    ptrB = load_matrix(alloc, B_in)
    ptrC = load_matrix(alloc, [[0.0] * N for _ in range(N)])

    log = []
    orig_read = ManualAllocator.read
    def logged_read(self, addr):
        log.append(addr)
        return orig_read(self, addr)
    ManualAllocator.read = logged_read

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

    ManualAllocator.read = orig_read

    regions = {
        'fast_A': (sp.fast_A, sp.fast_A + tile_size * tile_size - 1),
        'fast_B': (sp.fast_B, sp.fast_B + tile_size * tile_size - 1),
        'fast_C': (sp.fast_C, sp.fast_C + tile_size * tile_size - 1),
        'main_A': (ptrA, ptrA + N * N - 1),
        'main_B': (ptrB, ptrB + N * N - 1),
        'main_C': (ptrC, ptrC + N * N - 1),
    }
    return log, regions, alloc.cost


# ============================================================================
# 4. Plotting
# ============================================================================

REGION_COLORS = {
    'fast_A': 'tab:green',
    'fast_B': 'tab:olive',
    'fast_C': 'tab:cyan',
    'main_A': 'tab:red',
    'main_B': 'tab:orange',
    'main_C': 'tab:purple',
}


def classify(addr, regions):
    for label, (lo, hi) in regions.items():
        if lo <= addr <= hi:
            return label
    return 'other'


def plot_panel(ax, addrs, regions, algo_label, cost, y_max):
    xs = np.arange(len(addrs))
    ys = np.array(addrs)
    labels = np.array([classify(int(a), regions) for a in ys])
    for region, color in REGION_COLORS.items():
        if region not in regions:
            continue
        mask = labels == region
        if mask.any():
            ax.scatter(xs[mask], ys[mask], s=6, alpha=0.55, c=color,
                       label=f"{region} ({regions[region][0]}..{regions[region][1]})",
                       rasterized=True, linewidths=0)
    ax.set_ylabel('Physical address', fontsize=11)
    ax.set_ylim(0, y_max)
    ax.set_title(f'{algo_label}  —  {len(addrs):,} accesses,  '
                 f'cost ∑⌈√addr⌉ = {cost:,}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5),
              framealpha=0.95)


def main():
    N = 16
    tile_size = 4

    naive_addrs, naive_regions, naive_cost = run_naive(N)
    rmm_addrs, rmm_regions, rmm_cost = run_rmm(N, tile_size)
    y_max = max(max(naive_addrs), max(rmm_addrs)) + 1

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    plot_panel(axes[0], naive_addrs, naive_regions,
               f'NAIVE triple-loop   (N={N})', naive_cost, y_max)
    plot_panel(axes[1], rmm_addrs, rmm_regions,
               f'RMM + scratchpad   (N={N}, tile={tile_size})', rmm_cost, y_max)
    axes[1].set_xlabel('Access index', fontsize=11)
    fig.suptitle(f'Manual matmul access traces  —  naive vs RMM  —  '
                 f'energy ratio  naive / rmm = {naive_cost / rmm_cost:.2f}×',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    out = 'manual_trace_n16.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f'Saved: {out}')
    print(f'NAIVE  — {len(naive_addrs):,} accesses, cost {naive_cost:,}')
    print(f'RMM    — {len(rmm_addrs):,} accesses, cost {rmm_cost:,}')
    print(f'Energy ratio (naive / rmm)  = {naive_cost / rmm_cost:.2f}×')
    plt.close()


if __name__ == '__main__':
    main()
