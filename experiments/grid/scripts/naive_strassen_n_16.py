#!/usr/bin/env -S /Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Self-contained reproducer for naive_strassen(n=16).

SELF-CONTAINED: this file imports nothing from ByteDMD; it inlines the
L2 IR, tracer, cost heuristics (space_dmd, bytedmd_live,
bytedmd_classic), two-stack Allocator, plot helpers, and the closure
of algorithm-specific code it needs. Hand this single file to a
collaborator and they can run it directly:

    uv run --script naive_strassen_n_16.py

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

N_MM = 16

# ===========================================================================
# Algorithm definitions (closure of what the Python impl needs).
# ===========================================================================

def _split(M):
    n = len(M); h = n // 2
    return ([[M[i][j] for j in range(h)] for i in range(h)],
            [[M[i][j] for j in range(h, n)] for i in range(h)],
            [[M[i][j] for j in range(h)] for i in range(h, n)],
            [[M[i][j] for j in range(h, n)] for i in range(h, n)])

def _join(C11, C12, C21, C22):
    h = len(C11); n = 2 * h
    return [[C11[i][j] if j < h else C12[i][j-h] for j in range(n)] for i in range(h)] + \
           [[C21[i][j] if j < h else C22[i][j-h] for j in range(n)] for i in range(h)]

def _addm(A, B):
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def _subm(A, B):
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

def matmul_strassen(A, B):
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)
    M1 = matmul_strassen(_addm(A11, A22), _addm(B11, B22))
    M2 = matmul_strassen(_addm(A21, A22), B11)
    M3 = matmul_strassen(A11, _subm(B12, B22))
    M4 = matmul_strassen(A22, _subm(B21, B11))
    M5 = matmul_strassen(_addm(A11, A12), B22)
    M6 = matmul_strassen(_subm(A21, A11), _addm(B11, B12))
    M7 = matmul_strassen(_subm(A12, A22), _addm(B21, B22))
    C11 = _addm(_subm(_addm(M1, M4), M5), M7)
    C12 = _addm(M3, M5)
    C21 = _addm(M2, M4)
    C22 = _addm(_subm(_addm(M1, M3), M2), M6)
    return _join(C11, C12, C21, C22)


# ===========================================================================
# Manual-schedule definitions (closure of what the manual impl needs).
# ===========================================================================

def manual_strassen(n: int, T: int = 4) -> int:
    """Standard Strassen. A, B on arg stack (at top); SA/SB/M/sA/sB/sC/C on
    scratch. Recursion threads is_arg flags through since sub-products
    flip pointers back to scratch buffers (M-intermediates)."""
    a = _alloc()
    A = a.alloc_arg(n * n); B = a.alloc_arg(n * n)
    tmp = a.alloc(1)
    sA = a.alloc(T * T); sB = a.alloc(T * T); sC = a.alloc(T * T)
    C = a.alloc(n * n)
    a.set_output_range(C, C + n * n)

    def _t(addr: int, is_arg: bool) -> None:
        (a.touch_arg if is_arg else a.touch)(addr)

    def add_mats(p1: int, s1: int, is1a: bool,
                 p2: int, s2: int, is2a: bool, h: int) -> None:
        for i in range(h):
            for j in range(h):
                _t(p1 + i * s1 + j, is1a)
                _t(p2 + i * s2 + j, is2a)

    def compute_tile(pA: int, sAstr: int, isAarg: bool,
                     pB: int, sBstr: int, isBarg: bool,
                     pC: int, sCstr: int) -> None:
        for ii in range(T):
            for jj in range(T):
                _t(pA + ii * sAstr + jj, isAarg)
                a.write(sA + ii * T + jj)
        for ii in range(T):
            for jj in range(T):
                _t(pB + ii * sBstr + jj, isBarg)
                a.write(sB + ii * T + jj)
        for ii in range(T):
            for jj in range(T):
                a.touch(pC + ii * sCstr + jj)
                a.write(sC + ii * T + jj)
        # MAC with priced intermediates: every kk reads tmp and sC.
        for ii in range(T):
            for jj in range(T):
                for kk in range(T):
                    a.touch(sA + ii * T + kk)
                    a.touch(sB + kk * T + jj)
                    a.write(tmp)
                    a.touch(sC + ii * T + jj)
                    a.touch(tmp)
                    a.write(sC + ii * T + jj)
        for ii in range(T):
            for jj in range(T):
                a.touch(sC + ii * T + jj)
                a.write(pC + ii * sCstr + jj)

    def recurse(pA_: int, sAstr: int, isAa: bool,
                pB_: int, sBstr: int, isBa: bool,
                pC_: int, sCstr: int, sz: int) -> None:
        if sz <= T:
            compute_tile(pA_, sAstr, isAa, pB_, sBstr, isBa, pC_, sCstr)
            return
        h = sz // 2
        ckpt = a.push()
        SA = a.alloc(h * h); SB = a.alloc(h * h)
        M = [a.alloc(h * h) for _ in range(7)]

        A11, A12 = pA_, pA_ + h
        A21, A22 = pA_ + h * sAstr, pA_ + h * sAstr + h
        B11, B12 = pB_, pB_ + h
        B21, B22 = pB_ + h * sBstr, pB_ + h * sBstr + h

        add_mats(A11, sAstr, isAa, A22, sAstr, isAa, h)
        add_mats(B11, sBstr, isBa, B22, sBstr, isBa, h)
        recurse(SA, h, False, SB, h, False, M[0], h, h)

        add_mats(A21, sAstr, isAa, A22, sAstr, isAa, h)
        recurse(SA, h, False, B11, sBstr, isBa, M[1], h, h)

        add_mats(B12, sBstr, isBa, B22, sBstr, isBa, h)
        recurse(A11, sAstr, isAa, SB, h, False, M[2], h, h)

        add_mats(B21, sBstr, isBa, B11, sBstr, isBa, h)
        recurse(A22, sAstr, isAa, SB, h, False, M[3], h, h)

        add_mats(A11, sAstr, isAa, A12, sAstr, isAa, h)
        recurse(SA, h, False, B22, sBstr, isBa, M[4], h, h)

        add_mats(A21, sAstr, isAa, A11, sAstr, isAa, h)
        add_mats(B11, sBstr, isBa, B12, sBstr, isBa, h)
        recurse(SA, h, False, SB, h, False, M[5], h, h)

        add_mats(A12, sAstr, isAa, A22, sAstr, isAa, h)
        add_mats(B21, sBstr, isBa, B22, sBstr, isBa, h)
        recurse(SA, h, False, SB, h, False, M[6], h, h)

        # Assemble C quadrants. M[*] and pC_ are both scratch.
        def read_M(*indices: int) -> None:
            for i in range(h):
                for j in range(h):
                    for idx in indices:
                        a.touch(M[idx] + i * h + j)

        def write_quadrant(qr: int, qc: int) -> None:
            for i in range(h):
                for j in range(h):
                    a.write(pC_ + (qr + i) * sCstr + (qc + j))

        read_M(0, 3, 4, 6); write_quadrant(0, 0)
        read_M(2, 4);        write_quadrant(0, h)
        read_M(1, 3);        write_quadrant(h, 0)
        read_M(0, 1, 2, 5);  write_quadrant(h, h)

        a.pop(ckpt)

    recurse(A, n, True, B, n, True, C, n, n)
    a.read_output()
    return a.cost
# ===========================================================================
# Driver — run under this script's specific algorithm.
# ===========================================================================

NAME   = 'naive_strassen(n=16)'
SLUG   = 'naive_strassen_n_16'
FN     = matmul_strassen
ARGS   = (mat(N_MM), mat(N_MM))
MANUAL = lambda: manual_strassen(N_MM, T=4)


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
