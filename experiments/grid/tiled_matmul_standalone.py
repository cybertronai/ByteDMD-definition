#!/usr/bin/env python3
"""
Self-contained reproduction of the tiled_matmul(n=16) row from the grid table:

    | algorithm          | space_dmd | bytedmd_live | manual | bytedmd_classic |
    | tiled_matmul(n=16) |    98,206 |       74,560 | 86,030 |         143,280 |

Four different cost models, all pricing memory reads by ceil(sqrt(depth_or_addr)):

  1. space_dmd      — "Optimal static compiler" (e.g. TPU scratchpad).
                      Assigns physical addresses once, up-front, based on access
                      density = access_count / lifespan. High-density variables
                      (inner-loop temporaries) get the lowest addresses. Cost of
                      a LOAD = ceil(sqrt(rank among currently-live variables)).
                      Uses a Fenwick tree for O(log V) rank queries.

  2. bytedmd_live   — "LRU cache with garbage collection". Tracks an LRU stack
                      of all live variables. When variable X is LOADed, its cost
                      is ceil(sqrt(depth)) where depth = #live vars above it in
                      the stack. After its LAST LOAD, X is removed ("compacted")
                      so dead vars don't inflate depth for others. Models an
                      ideal hardware LRU cache with perfect liveness info.

  3. manual         — "Hand-placed bump allocator with scratchpad". A physical
                      1-D address space where alloc() returns the next address.
                      The first 3*T^2 addresses hold the scratchpad tiles (sA,
                      sB, sC), followed by main-memory arrays A, B, C.
                      Reading address d costs ceil(sqrt(d)). Writes are free.
                      This models software-managed DMA with explicit tile loads.

  4. bytedmd_classic — "LRU cache, no GC" (the Mattson stack-distance model).
                      Same LRU stack as bytedmd_live, but dead variables are
                      NEVER removed. They stay in the stack forever, pushing
                      live data deeper. This is the worst-case: the "infinite
                      graveyard" where every past allocation pollutes the stack.

The algorithm being measured:
  matmul_tiled(A, B) — one-level blocked matrix multiplication of 16x16 matrices
  with tile size T = round(sqrt(16)) = 4. The six nested loops iterate as:
    for bi, bj (tile rows/cols of C):
      for bk (tile cols of A / rows of B):
        for i, j, k (within the tile):
          C[i][j] += A[i][k] * B[k][j]

  The tracer instruments this via operator overloading: each scalar in A and B
  is wrapped in a _Tracked object. Every +, * operation generates L2 events:
  L2Load(var) for each operand read, L2Op for the operation, L2Store(result)
  for the new value. The resulting event trace (tens of thousands of events)
  is then fed to each cost model.

Usage:
    python3 tiled_matmul_standalone.py
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import operator


# ============================================================================
# IR event types — the "bytecode" that all cost models operate on
# ============================================================================

@dataclass(frozen=True)
class L2Store:
    """A new value is written to variable `var`."""
    var: int

@dataclass(frozen=True)
class L2Load:
    """Variable `var` is read as an operand."""
    var: int

@dataclass(frozen=True)
class L2Op:
    """An arithmetic operation (add, mul, etc.) is performed."""
    name: str
    in_vars: Tuple[int, ...]
    out_var: Optional[int]


L2Event = Union[L2Store, L2Load, L2Op]


# ============================================================================
# Tracer — instruments plain Python arithmetic into L2 events
# ============================================================================

class _Tracer:
    """Global event recorder. Each _Tracked operation appends events here."""
    def __init__(self):
        self.events: List[L2Event] = []
        self.next_var = 0

    def fresh(self) -> int:
        self.next_var += 1
        return self.next_var


class _Tracked:
    """Wraps a Python float/int so that every +, * records L2 events.

    When you write `a + b` where both are _Tracked:
      - L2Load(a.var), L2Load(b.var)   — both operands are "read from memory"
      - L2Op("add", (a.var, b.var), result_var)
      - L2Store(result_var)            — result is "written to memory"
    The result is a new _Tracked wrapping the arithmetic result.

    When one operand is a plain float (e.g. `tracked * 2.0`), only the
    tracked operand generates a LOAD — the constant is "free".
    """
    __slots__ = ("_t", "_v", "val")

    def __init__(self, t: _Tracer, v: int, val):
        self._t = t
        self._v = v
        self.val = val

    def _binop(self, other, name, fn):
        if isinstance(other, _Tracked):
            in_vars = (self._v, other._v)
            other_val = other.val
        else:
            in_vars = (self._v,)
            other_val = other
        for v in in_vars:
            self._t.events.append(L2Load(v))
        result_val = fn(self.val, other_val)
        out_var = self._t.fresh()
        self._t.events.append(L2Op(name, in_vars, out_var))
        self._t.events.append(L2Store(out_var))
        return _Tracked(self._t, out_var, result_val)

    def __add__(self, o): return self._binop(o, "add", operator.add)
    def __mul__(self, o): return self._binop(o, "mul", operator.mul)
    def __radd__(self, o):
        for v in (self._v,):
            self._t.events.append(L2Load(v))
        out = self._t.fresh()
        self._t.events.append(L2Op("add", (self._v,), out))
        self._t.events.append(L2Store(out))
        return _Tracked(self._t, out, operator.add(o, self.val))


def trace(func, args):
    """Run func(*args) with tracing. Returns list of L2 events.

    Each scalar in `args` (nested lists of floats) is wrapped in _Tracked.
    The function runs normally, but every arithmetic op is recorded.
    """
    t = _Tracer()
    def wrap(v):
        if isinstance(v, list):
            return [wrap(x) for x in v]
        if isinstance(v, (int, float)):
            var = t.fresh()
            t.events.append(L2Store(var))
            return _Tracked(t, var, v)
        return v
    wrapped = tuple(wrap(a) for a in args)
    func(*wrapped)
    return t.events


# ============================================================================
# Algorithm: one-level tiled matmul (the function being measured)
# ============================================================================

def matmul_tiled(A, B, tile=None):
    """C = A @ B using one level of blocking with tile size T.

    The default tile = round(sqrt(n)). For n=16, T=4.
    Six nested loops: bi, bj iterate over T×T blocks of C;
    bk iterates over the shared dimension; i, j, k iterate within tiles.
    """
    n = len(A)
    if tile is None:
        tile = max(1, int(round(n ** 0.5)))
    C = [[None] * n for _ in range(n)]
    for bi in range(0, n, tile):
        for bj in range(0, n, tile):
            for bk in range(0, n, tile):
                for i in range(bi, min(bi + tile, n)):
                    for j in range(bj, min(bj + tile, n)):
                        for k in range(bk, min(bk + tile, n)):
                            if C[i][j] is None:
                                C[i][j] = A[i][k] * B[k][j]
                            else:
                                C[i][j] = C[i][j] + A[i][k] * B[k][j]
    return C


# ============================================================================
# Fenwick tree — O(log n) prefix sums, used by all LRU/rank-based models
# ============================================================================

class _Fenwick:
    """1-indexed Binary Indexed Tree. Supports point updates and prefix queries.

    Used to efficiently compute "how many items have index <= k" in O(log n),
    which translates to LRU stack depth or spatial rank queries.
    """
    __slots__ = ("n", "bit")

    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 1)

    def add(self, i: int, delta: int):
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def prefix(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s


# ============================================================================
# Cost model 1: SpaceDMD (density-ranked static allocation)
# ============================================================================

def space_dmd(events: Sequence[L2Event]) -> int:
    """Models an optimal ahead-of-time (AOT) static allocator like a TPU.

    Pass 1: Scan events to find each variable's birth time, last use time,
            and total access count.
    Pass 2: Compute density = access_count / lifespan for each variable.
            Sort all variables by density (descending). Assign rank 1 to
            the densest variable, rank 2 to the next, etc. This determines
            the static physical address layout.
    Pass 3: Sweep through time. At each step, birth/death events toggle
            variables in a Fenwick tree. On each LOAD, the variable's
            "active rank" (rank among currently-live variables) is queried
            via prefix sum. Cost = ceil(sqrt(active_rank)).

    Key insight: hot variables (small lifespan, many accesses) automatically
    get rank 1 (the fastest address), while cold bulk arrays sit farther out.
    """
    birth: Dict[int, int] = {}
    last_use: Dict[int, int] = {}
    access_count: Dict[int, int] = defaultdict(int)
    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            birth[ev.var] = i
            last_use.setdefault(ev.var, i)
        elif isinstance(ev, L2Load):
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

    births_at: Dict[int, List[int]] = defaultdict(list)
    deaths_at: Dict[int, List[int]] = defaultdict(list)
    for vid in birth:
        births_at[birth[vid]].append(vid)
        deaths_at[last_use[vid]].append(vid)

    bit = _Fenwick(V)
    total = 0
    for i, ev in enumerate(events):
        for vid in births_at[i]:
            bit.add(rank_map[vid], 1)
        if isinstance(ev, L2Load):
            active_rank = bit.prefix(rank_map[ev.var])
            total += math.isqrt(max(0, active_rank - 1)) + 1
        for vid in deaths_at[i]:
            bit.add(rank_map[vid], -1)
    return total


# ============================================================================
# Cost models 2 & 4: LRU stack (bytedmd_live and bytedmd_classic)
# ============================================================================

def _lru_cost(events: Sequence[L2Event], compact_on_last_load: bool) -> int:
    """Shared LRU stack engine for both bytedmd_live and bytedmd_classic.

    Maintains an LRU stack using timestamps and a Fenwick tree:
    - Each variable gets a timestamp when STOREd (pushed to top of stack).
    - On LOAD, the variable's depth = #vars with timestamp >= its timestamp.
      Cost = ceil(sqrt(depth)). Then its timestamp is refreshed (LRU bump).
    - If compact_on_last_load=True (bytedmd_live): on a variable's LAST LOAD,
      it is removed from the stack entirely. Dead vars free up space.
    - If compact_on_last_load=False (bytedmd_classic): dead vars stay forever,
      accumulating as "ghosts" that push live data deeper.
    """
    last_load: Dict[int, int] = {}
    if compact_on_last_load:
        for i, ev in enumerate(events):
            if isinstance(ev, L2Load):
                last_load[ev.var] = i

    T = len(events) + 1
    bit = _Fenwick(T)
    var_ts: Dict[int, int] = {}
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


def bytedmd_live(events):
    """LRU stack WITH liveness compaction. Dead vars are removed on last LOAD."""
    return _lru_cost(events, compact_on_last_load=True)


def bytedmd_classic(events):
    """LRU stack WITHOUT liveness compaction. Dead vars stay forever."""
    return _lru_cost(events, compact_on_last_load=False)


# ============================================================================
# Cost model 3: Manual (hand-placed bump allocator with scratchpad)
# ============================================================================

def manual_tiled_matmul(n: int, T: int = None) -> int:
    """Physical cost of tiled matmul with a software-managed scratchpad.

    Memory layout (1-D bump-allocated, address 1 onwards):
      Addresses 1..T^2        : scratchpad tile sA (fast, low sqrt cost)
      Addresses T^2+1..2T^2   : scratchpad tile sB
      Addresses 2T^2+1..3T^2  : scratchpad tile sC
      Addresses 3T^2+1..3T^2+N^2 : main memory A
      Then B, then C.

    For each (bi, bj) output tile:
      1. Load C tile from main memory into sC (read each C element).
      2. For each bk:
         a. Load A tile into sA (read each A element from main memory).
         b. Load B tile into sB (read each B element from main memory).
         c. MAC loop: for each (ii, jj), read sC once, then for each kk
            read sA and sB. The scratchpad reads are cheap (low addresses).
      3. Flush: read sC to write back (free) to main memory C.

    Cost per read at address d = ceil(sqrt(d)) = isqrt(d-1) + 1.
    Writes are free (no cost charged).
    """
    if T is None:
        T = max(1, int(round(n ** 0.5)))
    cost = 0
    # Allocate scratchpad tiles at the lowest addresses
    sA = 1;             ptr = 1 + T * T
    sB = ptr;           ptr += T * T
    sC = ptr;           ptr += T * T
    # Main memory arrays follow
    A = ptr;            ptr += n * n
    B = ptr;            ptr += n * n
    C = ptr;            ptr += n * n

    def touch(addr):
        nonlocal cost
        cost += math.isqrt(max(0, addr - 1)) + 1

    for bi in range(0, n, T):
        for bj in range(0, n, T):
            # Load C tile into scratchpad sC
            for ii in range(min(T, n - bi)):
                for jj in range(min(T, n - bj)):
                    touch(C + (bi + ii) * n + (bj + jj))
            for bk in range(0, n, T):
                # Load A tile into scratchpad sA
                for ii in range(min(T, n - bi)):
                    for kk in range(min(T, n - bk)):
                        touch(A + (bi + ii) * n + (bk + kk))
                # Load B tile into scratchpad sB
                for kk in range(min(T, n - bk)):
                    for jj in range(min(T, n - bj)):
                        touch(B + (bk + kk) * n + (bj + jj))
                # MAC: multiply-accumulate in scratchpad
                for ii in range(min(T, n - bi)):
                    for jj in range(min(T, n - bj)):
                        touch(sC + ii * T + jj)     # read accumulator
                        for kk in range(min(T, n - bk)):
                            touch(sA + ii * T + kk)  # read A tile element
                            touch(sB + kk * T + jj)  # read B tile element
            # Flush sC back to main memory (read sC, write C is free)
            for ii in range(min(T, n - bi)):
                for jj in range(min(T, n - bj)):
                    touch(sC + ii * T + jj)
    return cost


# ============================================================================
# Main — generate and print all four numbers
# ============================================================================

def main():
    n = 16

    # Step 1: Trace the tiled matmul to get L2 events
    A = [[1.0] * n for _ in range(n)]
    B = [[1.0] * n for _ in range(n)]
    events = trace(matmul_tiled, (A, B))
    print(f"Traced matmul_tiled(n={n}): {len(events):,} L2 events\n")

    # Step 2: Evaluate all four cost models on the same trace
    sd = space_dmd(events)
    bl = bytedmd_live(events)
    mn = manual_tiled_matmul(n)
    bc = bytedmd_classic(events)

    print(f"| {'algorithm':<25} | {'space_dmd':>10} | {'bytedmd_live':>12} | {'manual':>8} | {'bytedmd_classic':>15} |")
    print(f"|{'-'*27}|{'-'*12}|{'-'*14}|{'-'*10}|{'-'*17}|")
    print(f"| {'tiled_matmul(n=16)':<25} | {sd:>10,} | {bl:>12,} | {mn:>8,} | {bc:>15,} |")

    print(f"\nExpected:                  |     98,206 |       74,560 |   86,030 |         143,280 |")


if __name__ == "__main__":
    main()
