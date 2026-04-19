#!/usr/bin/env -S /Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Self-contained reproducer for stencil_time_diamond(16x16,T=4).

SELF-CONTAINED: this file imports nothing from ByteDMD; it inlines the
L2 IR, tracer, cost heuristics (space_dmd, bytedmd_live,
bytedmd_classic), two-stack Allocator, plot helpers, and the closure
of algorithm-specific code it needs. Hand this single file to a
collaborator and they can run it directly:

    uv run --script stencil_time_diamond_16x16_t_4.py

Produces three PNGs (into ../traces/ if that directory exists, else
alongside the script) and prints a summary table of all four costs
plus peak live working-set size and max/median reuse distance.
"""
from __future__ import annotations
import os as _os
import sys as _sys
# ===========================================================================
# L2 IR (copied from bytedmd_ir.py) — LOAD / STORE / OP event types plus the
# _Tracer + _Tracked helpers that let plain Python arithmetic produce a
# trace of per-operand reads and per-result writes. Stores are free.
# ===========================================================================

import heapq
import math
import operator
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class L2Store:
    var: int

@dataclass(frozen=True)
class L2Load:
    var: int

@dataclass(frozen=True)
class L2Op:
    name: str
    in_vars: Tuple[int, ...]
    out_var: Optional[int]


L2Event = Union[L2Store, L2Load, L2Op]


class _Tracer:
    def __init__(self) -> None:
        self.events: List[L2Event] = []
        self.next_var = 0
        self.input_vars: List[int] = []

    def fresh(self) -> int:
        self.next_var += 1
        return self.next_var


class _Tracked:
    __slots__ = ("_t", "_v", "val")

    def __init__(self, t: _Tracer, v: int, val) -> None:
        self._t = t
        self._v = v
        self.val = val

    def _binop(self, other, name, fn):
        if isinstance(other, _Tracked):
            in_vars = (self._v, other._v); other_val = other.val
        else:
            in_vars = (self._v,); other_val = other
        for v in in_vars:
            self._t.events.append(L2Load(v))
        result_val = fn(self.val, other_val)
        out_var = self._t.fresh()
        self._t.events.append(L2Op(name, in_vars, out_var))
        self._t.events.append(L2Store(out_var))
        return _Tracked(self._t, out_var, result_val)

    def _rbinop(self, other, name, fn):
        in_vars = (self._v,)
        for v in in_vars:
            self._t.events.append(L2Load(v))
        result_val = fn(other, self.val)
        out_var = self._t.fresh()
        self._t.events.append(L2Op(name, in_vars, out_var))
        self._t.events.append(L2Store(out_var))
        return _Tracked(self._t, out_var, result_val)

    def __add__(self, o):     return self._binop(o, "add", operator.add)
    def __sub__(self, o):     return self._binop(o, "sub", operator.sub)
    def __mul__(self, o):     return self._binop(o, "mul", operator.mul)
    def __truediv__(self, o): return self._binop(o, "div", operator.truediv)
    def __radd__(self, o):    return self._rbinop(o, "add", operator.add)
    def __rsub__(self, o):    return self._rbinop(o, "sub", operator.sub)
    def __rmul__(self, o):    return self._rbinop(o, "mul", operator.mul)


def trace(func: Callable, args: Tuple) -> Tuple[List[L2Event], List[int]]:
    """Trace func(*args). Input scalars live on the argument stack (no
    initial L2Store); first L2Load of each is priced by heuristics
    against the arg-stack position. Trailing epilogue reads every
    scalar in the return value once."""
    t = _Tracer()

    def wrap(v):
        if isinstance(v, list):
            return [wrap(x) for x in v]
        if isinstance(v, tuple):
            return tuple(wrap(x) for x in v)
        if isinstance(v, (int, float)):
            var = t.fresh(); t.input_vars.append(var)
            return _Tracked(t, var, v)
        return v

    wrapped = tuple(wrap(a) for a in args)
    result = func(*wrapped)

    def emit_output_loads(v):
        if isinstance(v, _Tracked):
            t.events.append(L2Load(v._v))
        elif isinstance(v, (list, tuple)):
            for x in v: emit_output_loads(x)
        elif isinstance(v, dict):
            for x in v.values(): emit_output_loads(x)

    emit_output_loads(result)
    return t.events, t.input_vars


# ===========================================================================
# Heuristics: LRU depth (bytedmd_live, bytedmd_classic) and density-ranked
# static allocator (space_dmd). Both accept an input_arg_idx mapping so the
# first L2Load of each input prices against its arg-stack position and
# then promotes onto the geometric stack as if freshly stored.
# ===========================================================================

class _Fenwick:
    __slots__ = ("n", "bit")

    def __init__(self, n: int) -> None:
        self.n = n
        self.bit = [0] * (n + 1)

    def add(self, i: int, delta: int) -> None:
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def prefix(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s


def _lru_cost(events, compact_on_last_load, input_arg_idx=None):
    input_arg_idx = input_arg_idx or {}
    pending = set(input_arg_idx)
    last_load = {}
    if compact_on_last_load:
        for i, ev in enumerate(events):
            if isinstance(ev, L2Load):
                last_load[ev.var] = i

    T = len(events) + len(input_arg_idx) + 1
    bit = _Fenwick(T)
    var_ts = {}
    next_ts = 0
    total = 0

    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            if compact_on_last_load and ev.var not in last_load:
                continue
            next_ts += 1
            var_ts[ev.var] = next_ts
            bit.add(next_ts, 1)
        elif isinstance(ev, L2Load):
            if ev.var in pending:
                arg_idx = input_arg_idx[ev.var]
                total += math.isqrt(max(0, arg_idx - 1)) + 1
                pending.discard(ev.var)
                if compact_on_last_load and last_load.get(ev.var) == i:
                    continue
                next_ts += 1
                var_ts[ev.var] = next_ts
                bit.add(next_ts, 1)
                continue
            t = var_ts[ev.var]
            total_live = bit.prefix(T)
            depth = total_live - bit.prefix(t - 1)
            total += math.isqrt(depth - 1) + 1
            bit.add(t, -1)
            if compact_on_last_load and last_load[ev.var] == i:
                del var_ts[ev.var]
            else:
                next_ts += 1
                var_ts[ev.var] = next_ts
                bit.add(next_ts, 1)
    return total


def bytedmd_classic(events, input_arg_idx=None):
    return _lru_cost(events, compact_on_last_load=False,
                     input_arg_idx=input_arg_idx)


def bytedmd_live(events, input_arg_idx=None):
    return _lru_cost(events, compact_on_last_load=True,
                     input_arg_idx=input_arg_idx)


def space_dmd(events, input_arg_idx=None):
    """Density-ranked static allocator. Pass 1: build (birth, last_use,
    access_count) per var. Pass 2: rank by density. Pass 3: sweep events
    against a Fenwick tree. First L2Load of an input prices against the
    arg-stack position instead of the geom-stack rank."""
    input_arg_idx = input_arg_idx or {}
    birth, last_use = {}, {}
    access_count = defaultdict(int)
    first_load_of_input = {}
    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            birth[ev.var] = i
            last_use.setdefault(ev.var, i)
        elif isinstance(ev, L2Load):
            if ev.var in input_arg_idx and ev.var not in birth:
                birth[ev.var] = i
                first_load_of_input[ev.var] = i
            last_use[ev.var] = i
            access_count[ev.var] += 1

    V = len(birth)
    if V == 0:
        return 0

    def priority(vid):
        lifespan = last_use[vid] - birth[vid] + 1
        density = access_count[vid] / lifespan
        return (-density, -access_count[vid], birth[vid], vid)

    sorted_vids = sorted(birth.keys(), key=priority)
    rank_map = {vid: i + 1 for i, vid in enumerate(sorted_vids)}

    births_at, deaths_at = defaultdict(list), defaultdict(list)
    for vid in birth:
        births_at[birth[vid]].append(vid)
        deaths_at[last_use[vid]].append(vid)

    bit = _Fenwick(V)
    total = 0
    for i, ev in enumerate(events):
        for vid in births_at[i]:
            bit.add(rank_map[vid], 1)
        if isinstance(ev, L2Load):
            if first_load_of_input.get(ev.var) == i:
                arg_idx = input_arg_idx[ev.var]
                total += math.isqrt(max(0, arg_idx - 1)) + 1
            else:
                active_rank = bit.prefix(rank_map[ev.var])
                total += math.isqrt(max(0, active_rank - 1)) + 1
        for vid in deaths_at[i]:
            bit.add(rank_map[vid], -1)
    return total


# ===========================================================================
# Allocator (hand-placed bump-pointer with two independent stacks + write
# tracking) + module-global override hook used by the manual_* functions.
# ===========================================================================

class Allocator:
    __slots__ = ("cost", "ptr", "peak", "arg_ptr", "arg_peak",
                 "log", "writes", "output_writes", "out_start", "out_end")

    def __init__(self, logging: bool = False) -> None:
        self.cost = 0
        self.ptr = 1
        self.peak = 1
        self.arg_ptr = 1
        self.arg_peak = 1
        self.log = [] if logging else None
        self.writes = [] if logging else None
        self.output_writes = [] if logging else None
        self.out_start = None
        self.out_end = None

    def alloc(self, size):
        addr = self.ptr; self.ptr += size
        if self.ptr > self.peak: self.peak = self.ptr
        return addr

    def alloc_arg(self, size):
        addr = self.arg_ptr; self.arg_ptr += size
        if self.arg_ptr > self.arg_peak: self.arg_peak = self.arg_ptr
        return addr

    def push(self): return self.ptr
    def pop(self, p): self.ptr = p

    def set_output_range(self, start, end):
        self.out_start = start; self.out_end = end

    def touch(self, addr):
        self.cost += math.isqrt(max(0, addr - 1)) + 1
        if self.log is not None:
            self.log.append(("scratch", addr))

    def touch_arg(self, addr):
        self.cost += math.isqrt(max(0, addr - 1)) + 1
        if self.log is not None:
            self.log.append(("arg", addr))

    def write(self, addr):
        if self.writes is None:
            return
        t = len(self.log)
        if (self.out_start is not None
                and self.out_start <= addr < self.out_end):
            self.output_writes.append((t, addr))
        else:
            self.writes.append((t, addr))

    def read_output(self):
        if self.out_start is None or self.out_end is None: return
        for addr in range(self.out_start, self.out_end):
            self.cost += math.isqrt(max(0, addr - 1)) + 1
            if self.log is not None:
                self.log.append(("output", addr))


_CURRENT_ALLOC: Optional[Allocator] = None


def set_allocator(a):
    global _CURRENT_ALLOC
    _CURRENT_ALLOC = a


def _alloc():
    return _CURRENT_ALLOC if _CURRENT_ALLOC is not None else Allocator()


# ===========================================================================
# Plotting helpers (copied from generate_traces.py + trace_diagnostics.py).
# Rendered as 200-DPI PNGs so points stay crisp under zoom. Arg reads
# plot shifted DOWN (y = -addr) to live in a separate band below y=0;
# output-epilogue reads draw in dark magenta on top of scratch reads.
# ===========================================================================

def plot_trace(log, writes, output_writes, scratch_peak, arg_peak,
               title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arg_t, arg_y, scr_t, scr_y, out_t, out_y = [], [], [], [], [], []
    for t, (space, addr) in enumerate(log):
        if space == "arg": arg_t.append(t); arg_y.append(-addr)
        elif space == "output": out_t.append(t); out_y.append(addr)
        else: scr_t.append(t); scr_y.append(addr)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    if scr_t:
        ax.scatter(scr_t, scr_y, s=0.8, c="tab:blue", alpha=0.55,
                   rasterized=True, linewidths=0, label="scratch read")
    if arg_t:
        ax.scatter(arg_t, arg_y, s=0.8, c="tab:green", alpha=0.55,
                   rasterized=True, linewidths=0,
                   label="arg read (shifted -addr)")
    if out_t:
        ax.scatter(out_t, out_y, s=0.8, c="#8B008B", alpha=0.75,
                   rasterized=True, linewidths=0, zorder=5,
                   label="output read (epilogue)")
    if writes:
        wt, wa = zip(*writes)
        ax.scatter(wt, wa, s=1.2, c="tab:orange", alpha=0.65,
                   rasterized=True, linewidths=0, label="scratch write")
    if output_writes:
        wt, wa = zip(*output_writes)
        ax.scatter(wt, wa, s=1.2, c="tab:red", alpha=0.75,
                   rasterized=True, linewidths=0, label="output write")
    if arg_t:
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Physical address (scratch positive / arg negative)")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    if log or writes or output_writes:
        ax.legend(loc="upper left", markerscale=8, fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_liveset(times, sizes, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(times, sizes, color="tab:blue", linewidth=0.8,
            drawstyle="steps-post", rasterized=True)
    ax.fill_between(times, 0, sizes, color="tab:blue", alpha=0.18,
                    linewidth=0, step="post", rasterized=True)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Live variables on geom stack")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    if times: ax.set_xlim(0, times[-1] + 1)
    ax.set_ylim(bottom=0); fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reuse_distance(times, distances, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.scatter(times, distances, s=0.8, c="tab:purple", alpha=0.35,
               linewidths=0, rasterized=True)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Reuse distance (LRU depth at read)")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    if times: ax.set_xlim(0, times[-1] + 1)
    ax.set_ylim(bottom=0); fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def walk_live_and_reuse(events, input_vars):
    input_arg_idx = {v: i + 1 for i, v in enumerate(input_vars)}
    pending = set(input_arg_idx)
    last_load = {}
    for i, ev in enumerate(events):
        if isinstance(ev, L2Load):
            last_load[ev.var] = i
    T = len(events) + len(input_arg_idx) + 2
    bit = [0] * (T + 1)
    def bit_add(i, d):
        while i <= T: bit[i] += d; i += i & -i
    def bit_prefix(i):
        s = 0
        while i > 0: s += bit[i]; i -= i & -i
        return s
    ts_of = {}
    next_ts = 0; live_count = 0
    ls_times, ls_sizes, rd_times, rd_distances = [], [], [], []
    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            if ev.var in last_load:
                next_ts += 1; ts_of[ev.var] = next_ts
                bit_add(next_ts, 1); live_count += 1
        elif isinstance(ev, L2Load):
            if ev.var in pending:
                pending.discard(ev.var)
                d = input_arg_idx[ev.var]
                rd_times.append(i); rd_distances.append(d)
                if last_load.get(ev.var) != i:
                    next_ts += 1; ts_of[ev.var] = next_ts
                    bit_add(next_ts, 1); live_count += 1
            else:
                t = ts_of[ev.var]
                total_live = bit_prefix(T)
                depth = total_live - bit_prefix(t - 1)
                rd_times.append(i); rd_distances.append(depth)
                bit_add(t, -1)
                if last_load[ev.var] == i:
                    del ts_of[ev.var]; live_count -= 1
                else:
                    next_ts += 1; ts_of[ev.var] = next_ts
                    bit_add(next_ts, 1)
        ls_times.append(i); ls_sizes.append(live_count)
    return ls_times, ls_sizes, rd_times, rd_distances


# ===========================================================================
# Input shape helpers (copied from run_grid.py).
# ===========================================================================

def mat(n, val=1.0): return [[val] * n for _ in range(n)]
def rect(rows, cols, val=1.0): return [[val] * cols for _ in range(rows)]
def vec(n, val=1.0): return [val] * n
def cube(d0, d1, d2, val=1.0):
    return [[[val] * d2 for _ in range(d1)] for _ in range(d0)]
def tensor4(d0, d1, d2, d3, val=1.0):
    return [[[[val] * d3 for _ in range(d2)] for _ in range(d1)]
            for _ in range(d0)]

# ===========================================================================
# Size constants (copied from run_grid.py).
# ===========================================================================

N_ST_TIME = 16
T_STENCIL = 4

# ===========================================================================
# Algorithm definitions (closure of what the Python impl needs).
# ===========================================================================

def stencil_time_diamond(A, T=4, block=4):
    """Diamond/trapezoidal time-tiling — advance T timesteps inside each
    (i,j) block before moving on. Halos shrink by 1 row/col per step so
    the block at step t is (block + 2*(T-t)) wide. Carries cur in a
    per-block buffer so boundary cells stay hot across sweeps."""
    n = len(A)
    cur = [[A[i][j] + 0 for j in range(n)] for i in range(n)]
    halo = T
    for bi in range(0, n, block):
        for bj in range(0, n, block):
            # Per-block, run T steps using local buffers.
            local_cur = [[cur[i][j] + 0
                          for j in range(max(0, bj - halo),
                                         min(n, bj + block + halo))]
                         for i in range(max(0, bi - halo),
                                        min(n, bi + block + halo))]
            rr = max(0, bi - halo); cc = max(0, bj - halo)
            rows = len(local_cur); cols = len(local_cur[0])
            for t in range(T):
                local_nxt = [[local_cur[ii][jj] + 0
                              for jj in range(cols)] for ii in range(rows)]
                for ii in range(1, rows - 1):
                    for jj in range(1, cols - 1):
                        if (0 < rr + ii < n - 1) and (0 < cc + jj < n - 1):
                            local_nxt[ii][jj] = (local_cur[ii][jj]
                                                 + local_cur[ii-1][jj]
                                                 + local_cur[ii+1][jj]
                                                 + local_cur[ii][jj-1]
                                                 + local_cur[ii][jj+1])
                local_cur = local_nxt
            # Flush the interior (bi..bi+block) back to cur.
            for i in range(bi, min(bi + block, n)):
                for j in range(bj, min(bj + block, n)):
                    local_i = i - rr; local_j = j - cc
                    cur[i][j] = local_cur[local_i][local_j] + 0
    return cur


# ===========================================================================
# Manual-schedule definitions (closure of what the manual impl needs).
# ===========================================================================

def manual_stencil_time_diamond(n: int, T: int = 4, block: int = 4) -> int:
    """Perfectly in-place diamond time-tiling with L1 row caching and
    sliding scalar registers (gemini/optimize-stencil-time-diamond.md).

    Three optimizations layered onto the naive halo-buffered schedule:
      (1) Lazy arg loading: only arg cells that are actually inside
          the Manhattan-distance diamond at the current block get
          touched, and only on their very first visit.
      (2) In-place time-stepping: the second "buf_nxt" array is
          dropped entirely. The sliding horizontal window (c_left,
          c_center, c_right) plus a top-row buffer prev_row lets every
          update read the stale neighbor values before the in-place
          write touches that cell.
      (3) Diamond pruning: each time step t clips to the shrinking
          dependence cone (dist_i + dist_j <= T - 1 - t), skipping
          cells whose values would be overwritten by halo before they
          become needed.

    Layout:
      c_left, c_center, c_right  (addrs 1..3)
      prev_row                   (addrs 4..stride+3)
      buf_cur                    (addrs stride+4..stride²+stride+3)
      cur                        (addrs stride²+stride+4..)
    """
    a = _alloc()
    A = a.alloc_arg(n * n)
    c_left = a.alloc(1); c_center = a.alloc(1); c_right = a.alloc(1)
    stride = block + 2 * T
    prev_row = a.alloc(stride)
    buf_cur = a.alloc(stride * stride)
    cur = a.alloc(n * n)
    a.set_output_range(cur, cur + n * n)

    loaded = [[False] * n for _ in range(n)]

    for bi in range(0, n, block):
        for bj in range(0, n, block):
            rr = max(0, bi - T); cc = max(0, bj - T)
            rows = min(n, bi + block + T) - rr
            cols = min(n, bj + block + T) - cc

            # (a) Lazy-load just the diamond subset into buf_cur.
            for ii in range(rows):
                for jj in range(cols):
                    r_glob = rr + ii
                    c_glob = cc + jj
                    dist_i = 0
                    if r_glob < bi: dist_i = bi - r_glob
                    elif r_glob >= bi + block: dist_i = r_glob - (bi + block - 1)
                    dist_j = 0
                    if c_glob < bj: dist_j = bj - c_glob
                    elif c_glob >= bj + block: dist_j = c_glob - (bj + block - 1)
                    if dist_i + dist_j <= T:
                        if not loaded[r_glob][c_glob]:
                            a.touch_arg(A + r_glob * n + c_glob)
                            loaded[r_glob][c_glob] = True
                        else:
                            a.touch(cur + r_glob * n + c_glob)
                        a.write(buf_cur + ii * stride + jj)

            # (b) In-place time-stepping with sliding register window.
            for t in range(T):
                for jj in range(cols):
                    a.touch(buf_cur + 0 * stride + jj)
                    a.write(prev_row + jj)
                for ii in range(1, rows - 1):
                    a.touch(buf_cur + ii * stride + 0); a.write(c_left)
                    a.touch(buf_cur + ii * stride + 1); a.write(c_center)
                    for jj in range(1, cols - 1):
                        a.touch(buf_cur + ii * stride + jj + 1)
                        a.write(c_right)

                        r_glob = rr + ii
                        c_glob = cc + jj
                        dist_i = 0
                        if r_glob < bi: dist_i = bi - r_glob
                        elif r_glob >= bi + block: dist_i = r_glob - (bi + block - 1)
                        dist_j = 0
                        if c_glob < bj: dist_j = bj - c_glob
                        elif c_glob >= bj + block: dist_j = c_glob - (bj + block - 1)

                        if dist_i + dist_j <= T - 1 - t:
                            if (0 < r_glob < n - 1) and (0 < c_glob < n - 1):
                                a.touch(c_center)
                                a.touch(prev_row + jj)
                                a.touch(buf_cur + (ii + 1) * stride + jj)
                                a.touch(c_left)
                                a.touch(c_right)
                                a.write(buf_cur + ii * stride + jj)

                        # Slide registers forward every column.
                        a.touch(c_center); a.write(prev_row + jj)
                        a.touch(c_center); a.write(c_left)
                        a.touch(c_right);  a.write(c_center)

            # (c) Flush the block interior to cur.
            for i in range(bi, min(bi + block, n)):
                for j in range(bj, min(bj + block, n)):
                    li = i - rr; lj = j - cc
                    a.touch(buf_cur + li * stride + lj)
                    a.write(cur + i * n + j)
    a.read_output()
    return a.cost
# ===========================================================================
# Driver — run under this script's specific algorithm.
# ===========================================================================

NAME   = 'stencil_time_diamond(16x16,T=4)'
SLUG   = 'stencil_time_diamond_16x16_t_4'
FN     = lambda A: stencil_time_diamond(A, T=T_STENCIL, block=4)
ARGS   = (mat(N_ST_TIME),)
MANUAL = lambda: manual_stencil_time_diamond(N_ST_TIME, T=T_STENCIL,
                                                block=4)


def _traces_dir():
    here = _os.path.dirname(_os.path.abspath(__file__))
    sibling = _os.path.normpath(_os.path.join(here, "..", "traces"))
    if _os.path.isdir(sibling):
        return sibling
    return here


def main() -> None:
    events, input_vars = trace(FN, ARGS)
    input_idx = {v: i + 1 for i, v in enumerate(input_vars)}
    costs = {
        "space_dmd":       space_dmd(events, input_idx),
        "bytedmd_live":    bytedmd_live(events, input_idx),
        "manual":          MANUAL(),
        "bytedmd_classic": bytedmd_classic(events, input_idx),
    }

    ls_t, ls_s, rd_t, rd_d = walk_live_and_reuse(events, input_vars)
    peak_live    = max(ls_s) if ls_s else 0
    max_reuse    = max(rd_d) if rd_d else 0
    median_reuse = sorted(rd_d)[len(rd_d) // 2] if rd_d else 0

    logged = Allocator(logging=True)
    set_allocator(logged)
    try: MANUAL()
    finally: set_allocator(None)

    out_dir = _traces_dir()
    plot_trace(logged.log, logged.writes, logged.output_writes,
               logged.peak, logged.arg_peak,
               f"{NAME}  —  cost = {logged.cost:,}",
               _os.path.join(out_dir, f"{SLUG}.png"))
    plot_liveset(ls_t, ls_s,
                 f"{NAME} — live working-set size (peak = {peak_live:,})",
                 _os.path.join(out_dir, f"{SLUG}_liveset.png"))
    plot_reuse_distance(rd_t, rd_d,
        f"{NAME} — reuse distance per load (max = {max_reuse:,})",
        _os.path.join(out_dir, f"{SLUG}_reuse_distance.png"))

    print(f"{NAME}")
    print(f"  events          {len(events):>12,}")
    for k in ("space_dmd", "bytedmd_live", "manual", "bytedmd_classic"):
        print(f"  {k:<15} {costs[k]:>12,}")
    print(f"  peak_live       {peak_live:>12,}")
    print(f"  max_reuse       {max_reuse:>12,}")
    print(f"  median_reuse    {median_reuse:>12,}")


if __name__ == "__main__":
    main()
