#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Interval graph: for each memory address, plot the time intervals during
which that address is bound to a particular logical value. Colors encode
the logical value itself — so when a DMA load copies A[i][k] from deep
memory into a scratchpad slot, the scratchpad slot and the deep-memory
slot share a color for as long as the scratchpad holds that copy.

Strategies visualized (both from gemini/illustrative-matmul-tiled.md):

  - Naive matmul (no scratchpad, no DMA): A, B, C all laid out starting
    at address 1. Each address holds a single value for the entire run.
  - Tiled matmul (scratchpad + DMA block loads): fast_A, fast_B, fast_C
    pinned to addresses 1..3T^2; A, B, C shifted above. Every outer
    (bi, bj, bk) iteration reloads fast_A and fast_B with a new T*T
    tile; fast_C rotates through the 16 C-blocks as (bi, bj) advances.

Color convention:
  A[i][k] -> viridis(i*N + k)
  B[k][j] -> plasma(k*N + j)
  C[i][j] -> inferno(i*N + j)

Within each matrix the 256 cells get a unique shade from the colormap
(256 distinct colors per matrix, 768 total). Scratchpad cells inherit
the color of whichever source cell they currently mirror.

Usage:
    uv run --script visualize_intervals.py
Produces `intervals_naive.svg` and `intervals_tiled.svg`.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

def value_color(family, i, j, N):
    """Per-value color: family picks colormap, (i, j) picks the shade."""
    t = 0.12 + 0.78 * (i * N + j) / max(1, N * N - 1)
    if family == 'A':
        return cm.viridis(t)
    elif family == 'B':
        return cm.plasma(t)
    elif family == 'C':
        return cm.inferno(t)
    raise ValueError(f'unknown family {family!r}')


# ---------------------------------------------------------------------------
# Interval tracer
# ---------------------------------------------------------------------------

class IntervalTracer:
    """Records, per address, a chain of (start, end, logical_value) spans.

    `bind(addr, family, i, j)` declares that `addr` now holds the value
    (family, i, j); any prior binding on that addr gets its `end`
    stamped with the current time. `read(addr)` advances time by one
    unit — writes are free and don't advance time (matches the
    experiment cost model). `finalize()` stamps every still-open span
    with the final time.
    """

    def __init__(self):
        self.intervals = []
        self.current = {}
        self.t = 0

    def bind(self, addr, family, i, j):
        key = (family, i, j)
        cur = self.current.get(addr)
        if cur is not None and cur['key'] == key:
            return   # same value as before; nothing to record
        if cur is not None:
            cur['end'] = self.t
        new_iv = {
            'addr': addr, 'start': self.t, 'end': None,
            'key': key, 'family': family, 'i': i, 'j': j,
        }
        self.intervals.append(new_iv)
        self.current[addr] = new_iv

    def read(self, addr):
        self.t += 1

    def finalize(self):
        for iv in self.intervals:
            if iv['end'] is None:
                iv['end'] = self.t


# ---------------------------------------------------------------------------
# Strategy tracers (mirror the gemini doc's minimal Python tracers)
# ---------------------------------------------------------------------------

def trace_naive(N):
    tr = IntervalTracer()
    base_A, base_B, base_C = 1, 1 + N * N, 1 + 2 * N * N
    for idx in range(N * N):
        tr.bind(base_A + idx, 'A', idx // N, idx % N)
    for idx in range(N * N):
        tr.bind(base_B + idx, 'B', idx // N, idx % N)
    for idx in range(N * N):
        tr.bind(base_C + idx, 'C', idx // N, idx % N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                tr.read(base_A + i * N + k)
                tr.read(base_B + k * N + j)
                tr.read(base_C + i * N + j)
    tr.finalize()
    return tr, (base_A, base_B, base_C), None


def trace_tiled(N, T):
    tr = IntervalTracer()
    fA, fB, fC = 1, 1 + T * T, 1 + 2 * T * T
    base_A = 1 + 3 * T * T
    base_B = base_A + N * N
    base_C = base_B + N * N
    # Deep A, B, C hold their identity values from t=0.
    for idx in range(N * N):
        tr.bind(base_A + idx, 'A', idx // N, idx % N)
    for idx in range(N * N):
        tr.bind(base_B + idx, 'B', idx // N, idx % N)
    for idx in range(N * N):
        tr.bind(base_C + idx, 'C', idx // N, idx % N)
    # Block-processing schedule.
    for bi in range(0, N, T):
        for bj in range(0, N, T):
            # fast_C mirrors the current (bi, bj) block of C for the
            # duration of all its bk iterations (it's the accumulator).
            for ii in range(T):
                for jj in range(T):
                    tr.bind(fC + ii * T + jj, 'C', bi + ii, bj + jj)
            for bk in range(0, N, T):
                # DMA: fast_A <- A[bi:bi+T, bk:bk+T].
                for ii in range(T):
                    for kk in range(T):
                        tr.bind(fA + ii * T + kk, 'A', bi + ii, bk + kk)
                        tr.read(base_A + (bi + ii) * N + (bk + kk))
                # DMA: fast_B <- B[bk:bk+T, bj:bj+T].
                for kk in range(T):
                    for jj in range(T):
                        tr.bind(fB + kk * T + jj, 'B', bk + kk, bj + jj)
                        tr.read(base_B + (bk + kk) * N + (bj + jj))
                # Compute reads (strictly inside the scratchpad).
                for ii in range(T):
                    for jj in range(T):
                        for kk in range(T):
                            tr.read(fA + ii * T + kk)
                            tr.read(fB + kk * T + jj)
                            tr.read(fC + ii * T + jj)
    tr.finalize()
    return tr, (base_A, base_B, base_C), (fA, fB, fC)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _label_matrix_bands(ax, bases_deep, bases_scratch, N, T):
    base_A, base_B, base_C = bases_deep
    ax.axhline(base_A - 0.5, color='white', lw=0.8)
    ax.axhline(base_B - 0.5, color='white', lw=0.8)
    ax.axhline(base_C - 0.5, color='white', lw=0.8)
    ax.text(-0.01, (base_A + base_B - 1) / 2 - 0.5, 'A',
            transform=ax.get_yaxis_transform(),
            ha='right', va='center', fontsize=11)
    ax.text(-0.01, (base_B + base_C - 1) / 2 - 0.5, 'B',
            transform=ax.get_yaxis_transform(),
            ha='right', va='center', fontsize=11)
    ax.text(-0.01, base_C + N * N / 2 - 0.5, 'C',
            transform=ax.get_yaxis_transform(),
            ha='right', va='center', fontsize=11)
    if bases_scratch is not None:
        fA, fB, fC = bases_scratch
        ax.axhline(fB - 0.5, color='white', lw=0.6)
        ax.axhline(fC - 0.5, color='white', lw=0.6)
        ax.axhline(base_A - 0.5, color='white', lw=1.2)
        ax.text(-0.01, (fA + fB - 1) / 2 - 0.5, 'fast_A',
                transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=9)
        ax.text(-0.01, (fB + fC - 1) / 2 - 0.5, 'fast_B',
                transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=9)
        ax.text(-0.01, (fC + base_A - 1) / 2 - 0.5, 'fast_C',
                transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=9)


def _draw_colorbar_legend(fig, N):
    """One small colorbar strip per family, side by side."""
    cbar_ax = fig.add_axes([0.88, 0.08, 0.02, 0.35])
    cbar_ax2 = fig.add_axes([0.915, 0.08, 0.02, 0.35])
    cbar_ax3 = fig.add_axes([0.95, 0.08, 0.02, 0.35])
    norm = Normalize(vmin=0, vmax=N * N - 1)
    sm_A = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    sm_B = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
    sm_C = cm.ScalarMappable(norm=norm, cmap=cm.inferno)
    cb1 = fig.colorbar(sm_A, cax=cbar_ax)
    cb2 = fig.colorbar(sm_B, cax=cbar_ax2)
    cb3 = fig.colorbar(sm_C, cax=cbar_ax3)
    cb1.set_label('A cell (i*N + k)')
    cb2.set_label('B cell (k*N + j)')
    cb3.set_label('C cell (i*N + j)')
    for cb in (cb1, cb2, cb3):
        cb.ax.tick_params(labelsize=7)


def _draw_intervals(ax, tr, N, bar_height=0.9):
    for iv in tr.intervals:
        color = value_color(iv['family'], iv['i'], iv['j'], N)
        ax.broken_barh(
            [(iv['start'], iv['end'] - iv['start'])],
            (iv['addr'] - bar_height / 2, bar_height),
            facecolors=color,
            edgecolor='none',
        )


def plot_intervals(tr, bases_deep, bases_scratch, N, T, title, path):
    last_addr = max(iv['addr'] for iv in tr.intervals)
    if bases_scratch is None:
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_axes([0.08, 0.08, 0.78, 0.83])
        _draw_intervals(ax, tr, N)
        ax.set_xlim(0, tr.t)
        ax.set_ylim(last_addr + 1, 0)
        ax.set_xlabel('time (priced reads)')
        ax.set_ylabel('address')
        ax.set_title(title)
        _label_matrix_bands(ax, bases_deep, bases_scratch, N, T)
        _draw_colorbar_legend(fig, N)
    else:
        # Two stacked panels: scratchpad zoom on top, full memory below.
        fA, fB, fC = bases_scratch
        scratch_last = bases_deep[0] - 1
        fig = plt.figure(figsize=(14, 11))
        ax_zoom = fig.add_axes([0.08, 0.72, 0.78, 0.21])
        ax_full = fig.add_axes([0.08, 0.08, 0.78, 0.58])

        # Zoom: only scratchpad rows, big bars so color switches are legible.
        _draw_intervals(ax_zoom, tr, N, bar_height=0.9)
        ax_zoom.set_xlim(0, tr.t)
        ax_zoom.set_ylim(scratch_last + 0.5, 0.5)
        ax_zoom.set_ylabel('scratchpad addr')
        ax_zoom.set_title(title + '\n(top: scratchpad zoom, addresses 1..'
                          f'{scratch_last})')
        ax_zoom.set_xticklabels([])
        ax_zoom.axhline(fB - 0.5, color='white', lw=0.8)
        ax_zoom.axhline(fC - 0.5, color='white', lw=0.8)
        ax_zoom.text(-0.01, (fA + fB - 1) / 2,
                     'fast_A', transform=ax_zoom.get_yaxis_transform(),
                     ha='right', va='center', fontsize=10)
        ax_zoom.text(-0.01, (fB + fC - 1) / 2,
                     'fast_B', transform=ax_zoom.get_yaxis_transform(),
                     ha='right', va='center', fontsize=10)
        ax_zoom.text(-0.01, (fC + bases_deep[0] - 1) / 2,
                     'fast_C', transform=ax_zoom.get_yaxis_transform(),
                     ha='right', va='center', fontsize=10)

        # Full memory.
        _draw_intervals(ax_full, tr, N)
        ax_full.set_xlim(0, tr.t)
        ax_full.set_ylim(last_addr + 1, 0)
        ax_full.set_xlabel('time (priced reads)')
        ax_full.set_ylabel('address')
        _label_matrix_bands(ax_full, bases_deep, None, N, T)
        _draw_colorbar_legend(fig, N)
    plt.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    N, T = 16, 4

    naive_tr, naive_bases, _ = trace_naive(N)
    tiled_tr, tiled_bases, tiled_scratch = trace_tiled(N, T)

    plot_intervals(
        naive_tr, naive_bases, None, N, T,
        f'Naive matmul (N={N}) — no scratchpad, no DMA\n'
        f'{len(naive_tr.intervals)} intervals across {naive_tr.t} priced reads',
        os.path.join(here, 'intervals_naive.svg'),
    )
    plot_intervals(
        tiled_tr, tiled_bases, tiled_scratch, N, T,
        f'Tiled matmul (N={N}, T={T}) — scratchpad + DMA block loads\n'
        f'{len(tiled_tr.intervals)} intervals across {tiled_tr.t} priced reads',
        os.path.join(here, 'intervals_tiled.svg'),
    )
    print(
        f'Wrote intervals_naive.svg '
        f'({len(naive_tr.intervals)} intervals) and '
        f'intervals_tiled.svg ({len(tiled_tr.intervals)} intervals).'
    )


if __name__ == '__main__':
    main()
