#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Self-contained script to generate memory access traces for Strassen vs RMM.
Includes manual bump/stack allocator, Scratchpad, access tracing, and scaling analysis.

Run:
    uv run --script strassen_trace.py
"""

import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class Allocator:
    """Bump-allocated physical memory with stack semantics and hardware cost logging."""
    def __init__(self, logging=True):
        self.cost = 0
        self.accesses = 0
        self.ptr = 1
        self.max_ptr = 1
        self.log = [] if logging else None

    def alloc(self, size: int) -> int:
        addr = self.ptr
        self.ptr += size
        if self.ptr > self.max_ptr:
            self.max_ptr = self.ptr
        return addr

    def push(self): return self.ptr
    def pop(self, ptr): self.ptr = ptr

    def read(self, addr: int):
        self.accesses += 1
        if self.log is not None:
            self.log.append(addr)
        self.cost += math.isqrt(max(0, addr - 1)) + 1


class Scratchpad:
    """Software-managed L1 scratchpad for 3 sub-tiles."""
    def __init__(self, alloc: Allocator, T: int):
        self.alloc, self.T = alloc, T
        self.fast = {'A': alloc.alloc(T*T), 'B': alloc.alloc(T*T), 'C': alloc.alloc(T*T)}
        self.loaded = {'A': None, 'B': None, 'C': None}
        self.dirty_C = False

    def sync(self, name: str, ptr: int, stride: int):
        if self.loaded[name] == (ptr, stride): return
        if name == 'C': self.flush_C()

        for i in range(self.T):
            for j in range(self.T):
                self.alloc.read(ptr + i * stride + j)

        self.loaded[name] = (ptr, stride)
        if name == 'C': self.dirty_C = False

    def flush_C(self):
        if self.loaded['C'] is not None and self.dirty_C:
            for i in range(self.T * self.T):
                self.alloc.read(self.fast['C'] + i)
            self.dirty_C = False

    def compute_tile(self, pA, sA, pB, sB, pC, sC):
        self.sync('A', pA, sA)
        self.sync('B', pB, sB)
        self.sync('C', pC, sC)
        T = self.T
        for i in range(T):
            for j in range(T):
                self.alloc.read(self.fast['C'] + i * T + j)
                for k in range(T):
                    self.alloc.read(self.fast['A'] + i * T + k)
                    self.alloc.read(self.fast['B'] + k * T + j)
        self.dirty_C = True


# ============================================================================
# Matmul Implementations
# ============================================================================

def run_rmm(N: int, T: int = 4, logging=True):
    alloc = Allocator(logging)
    sp = Scratchpad(alloc, T)
    pA, pB, pC = alloc.alloc(N*N), alloc.alloc(N*N), alloc.alloc(N*N)

    def recurse(rA, cA, rB, cB, rC, cC, sz):
        if sz <= T:
            sp.compute_tile(pA + rA*N + cA, N, pB + rB*N + cB, N, pC + rC*N + cC, N)
            return
        h = sz // 2
        for drA, dcA, drB, dcB, drC, dcC in [
            (0, 0, 0, 0, 0, 0), (0, 0, 0, h, 0, h),
            (h, 0, 0, h, h, h), (h, 0, 0, 0, h, 0),
            (h, h, h, 0, h, 0), (h, h, h, h, h, h),
            (0, h, h, h, 0, h), (0, h, h, 0, 0, 0)
        ]:
            recurse(rA + drA, cA + dcA, rB + drB, cB + dcB, rC + drC, cC + dcC, h)

    recurse(0, 0, 0, 0, 0, 0, N)
    sp.flush_C()

    regions = {
        'fast_A': (sp.fast['A'], sp.fast['A'] + T*T - 1), 'fast_B': (sp.fast['B'], sp.fast['B'] + T*T - 1),
        'fast_C': (sp.fast['C'], sp.fast['C'] + T*T - 1), 'main_A': (pA, pA + N*N - 1),
        'main_B': (pB, pB + N*N - 1), 'main_C': (pC, pC + N*N - 1),
    }
    return alloc.log, regions, alloc.accesses, alloc.cost


def run_strassen(N: int, T: int = 4, logging=True):
    alloc = Allocator(logging)
    sp = Scratchpad(alloc, T)
    pA, pB, pC = alloc.alloc(N*N), alloc.alloc(N*N), alloc.alloc(N*N)

    def add_mats(p1, s1, p2, s2, h):
        for i in range(h):
            for j in range(h):
                alloc.read(p1 + i*s1 + j)
                alloc.read(p2 + i*s2 + j)

    def recurse(pA_, sA, pB_, sB, pC_, sC, sz):
        if sz <= T:
            sp.compute_tile(pA_, sA, pB_, sB, pC_, sC)
            return

        h = sz // 2
        ckpt = alloc.push()

        SA, SB = alloc.alloc(h*h), alloc.alloc(h*h)
        M = [alloc.alloc(h*h) for _ in range(7)]

        A11, A12 = pA_, pA_ + h
        A21, A22 = pA_ + h*sA, pA_ + h*sA + h
        B11, B12 = pB_, pB_ + h
        B21, B22 = pB_ + h*sB, pB_ + h*sB + h

        add_mats(A11, sA, A22, sA, h); add_mats(B11, sB, B22, sB, h)
        recurse(SA, h, SB, h, M[0], h, h)

        add_mats(A21, sA, A22, sA, h)
        recurse(SA, h, B11, sB, M[1], h, h)

        add_mats(B12, sB, B22, sB, h)
        recurse(A11, sA, SB, h, M[2], h, h)

        add_mats(B21, sB, B11, sB, h)
        recurse(A22, sA, SB, h, M[3], h, h)

        add_mats(A11, sA, A12, sA, h)
        recurse(SA, h, B22, sB, M[4], h, h)

        add_mats(A21, sA, A11, sA, h); add_mats(B11, sB, B12, sB, h)
        recurse(SA, h, SB, h, M[5], h, h)

        add_mats(A12, sA, A22, sA, h); add_mats(B21, sB, B22, sB, h)
        recurse(SA, h, SB, h, M[6], h, h)

        sp.flush_C()
        def read_M(*indices):
            for i in range(h):
                for j in range(h):
                    for idx in indices:
                        alloc.read(M[idx] + i*h + j)

        read_M(0, 3, 4, 6)
        read_M(2, 4)
        read_M(1, 3)
        read_M(0, 1, 2, 5)

        alloc.pop(ckpt)

    recurse(pA, N, pB, N, pC, N, N)
    sp.flush_C()

    regions = {
        'fast_A': (sp.fast['A'], sp.fast['A'] + T*T - 1), 'fast_B': (sp.fast['B'], sp.fast['B'] + T*T - 1),
        'fast_C': (sp.fast['C'], sp.fast['C'] + T*T - 1), 'main_A': (pA, pA + N*N - 1),
        'main_B': (pB, pB + N*N - 1), 'main_C': (pC, pC + N*N - 1),
    }
    if alloc.max_ptr > pC + N*N:
        regions['stack_tmp'] = (pC + N*N, alloc.max_ptr - 1)

    return alloc.log, regions, alloc.accesses, alloc.cost


# ============================================================================
# Plotting & Log-Log Scaling Extraction
# ============================================================================

REGION_COLORS = {
    'fast_A': 'tab:green', 'fast_B': 'tab:olive', 'fast_C': 'tab:cyan',
    'main_A': 'tab:red', 'main_B': 'tab:orange', 'main_C': 'tab:purple',
    'stack_tmp': 'tab:gray'
}

def plot_panel(ax, addrs, regions, algo_label, accesses, cost, y_max):
    if not addrs: return
    ys, xs = np.array(addrs), np.arange(len(addrs))
    for name, color in REGION_COLORS.items():
        if name in regions:
            lo, hi = regions[name]
            mask = (ys >= lo) & (ys <= hi)
            if mask.any():
                ax.scatter(xs[mask], ys[mask], s=6, alpha=0.55, c=color,
                           label=f"{name} ({lo}..{hi})", rasterized=True, linewidths=0)

    ax.set_ylabel('Physical address', fontsize=11)
    ax.set_ylim(0, y_max)
    ax.set_title(f'{algo_label}  |  {accesses:,} accesses, cost ∑⌈√addr⌉ = {cost:,}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5), framealpha=0.95)


def fit(x, y):
    p, log_C = np.polyfit(np.log(x), np.log(y), 1)
    return np.exp(log_C), p


def run_scaling_analysis():
    Ns = [16, 32, 64]
    T = 4

    print("\n" + "="*57)
    print(" EMPIRICAL SCALING CONSTANTS (metric = C * N^p)")
    print("="*57)

    r_acc, r_cost, s_acc, s_cost = [], [], [], []
    for N in Ns:
        _, _, ra, rc = run_rmm(N, T, logging=False)
        _, _, sa, sc = run_strassen(N, T, logging=False)
        r_acc.append(ra); r_cost.append(rc)
        s_acc.append(sa); s_cost.append(sc)

    Cr_a, pr_a = fit(Ns, r_acc)
    Cr_c, pr_c = fit(Ns, r_cost)
    Cs_a, ps_a = fit(Ns, s_acc)
    Cs_c, ps_c = fit(Ns, s_cost)

    print(f"{'Algorithm':<10} | {'Accesses Scaling':<20} | {'Cost Scaling':<20}")
    print("-" * 57)
    print(f"{'RMM':<10} | {Cr_a:7.2f} * N^{pr_a:.3f}   | {Cr_c:7.2f} * N^{pr_c:.3f}")
    print(f"{'Strassen':<10} | {Cs_a:7.2f} * N^{ps_a:.3f}   | {Cs_c:7.2f} * N^{ps_c:.3f}")
    print("="*57 + "\n")
    print("Theoretical Hardware Limits Analysis:")
    print("  RMM Accesses:      O(N^3.000)   (Tight Scratchpad Loop)")
    print("  Strassen Accesses: O(N^2.807)   (Heavy Constant due to 18 matrix ops/level)")
    print("  RMM Cost:          O(N^4.000)   (N^3 accesses * O(N) average memory distance)")
    print("  Strassen Cost:     O(N^3.807)   (N^2.807 accesses * O(N) average memory distance)")


def main():
    N, T = 16, 4

    r_log, r_reg, r_acc, r_cost = run_rmm(N, T)
    s_log, s_reg, s_acc, s_cost = run_strassen(N, T)

    y_max = max(max(r_log), max(s_log)) + 1

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    plot_panel(axes[0], r_log, r_reg, f'RMM + Scratchpad (N={N}, tile={T})', r_acc, r_cost, y_max)
    plot_panel(axes[1], s_log, s_reg, f'Strassen + Scratchpad + Stack Alloc (N={N}, tile={T})', s_acc, s_cost, y_max)
    axes[1].set_xlabel('Access index', fontsize=11)

    fig.suptitle(f'Manual Memory Traces — RMM vs Strassen\nCost Ratio (RMM / Strassen) = {r_cost / s_cost:.2f}×',
                 fontsize=14, y=0.98)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'strassen_trace_n16.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()

    print(f'Saved: {out}')
    run_scaling_analysis()

if __name__ == '__main__':
    main()
