"""
ByteDMD — unified framework with three IR levels.

Level 1 (L1): Python source. Algorithms written as plain functions.
Level 2 (L2): Abstract IR. Sequence of LOAD/STORE/OP events with var IDs only.
              No physical addresses.
Level 3 (L3): Concrete IR. Same events with addresses assigned by an allocator.

Two L2-level metrics — priced directly on the trace by running an LRU stack
and charging ceil(sqrt(depth)) per LOAD:

  bytedmd_classic : LRU stack with NO liveness compaction. Dead variables
                    stay in the stack forever and pollute deeper rings. The
                    cost of a LOAD of X depends on the number of distinct
                    variables referenced since X's previous LOAD, regardless
                    of whether they are still live.
  bytedmd_live    : LRU stack WITH liveness compaction. A variable is
                    dropped from the stack immediately after its last LOAD.
                    The cost of a LOAD of X is thus determined by the
                    *live bytes* between its previous LOAD and the current
                    one.

Intermediate register-allocation policies (L2 -> L3) map variables to
physical addrs in {1, 2, 3, ...}; cost on L3 is sum ceil(sqrt(addr)) per
LOAD. These sit inside the envelope the two L2 metrics form:

  min_heap   : freed slots returned to a min-heap; always reuse the lowest.
  lru_static : freed slots returned in LIFO order (stack discipline).
  belady     : offline oracle; picks the lowest free addr with future-load
               information. Coincides with min_heap on matmul since every
               intermediate is read exactly once.
  no_reuse   : sanity baseline; never recycles. Pins each variable to its
               allocation order on a 2-D layout rather than sliding in an
               LRU stack.

Envelope claim:
  bytedmd_classic(trace) >= cost(any allocator above) >= bytedmd_live(trace).
"""

from __future__ import annotations

import heapq
import math
import operator
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union


# ============================================================================
# IR data types
# ============================================================================

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


@dataclass(frozen=True)
class L3Store:
    var: int
    addr: int

@dataclass(frozen=True)
class L3Load:
    var: int
    addr: int

@dataclass(frozen=True)
class L3Op:
    name: str
    in_vars: Tuple[int, ...]
    in_addrs: Tuple[int, ...]
    out_var: Optional[int]
    out_addr: Optional[int]


L2Event = Union[L2Store, L2Load, L2Op]
L3Event = Union[L3Store, L3Load, L3Op]


# ============================================================================
# L1 -> L2 : tracer via operator overloading
# ============================================================================

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

    def __repr__(self) -> str:
        return f"Tracked(v{self._v}={self.val})"

    def _binop(self, other, name: str, fn: Callable):
        if isinstance(other, _Tracked):
            in_vars: Tuple[int, ...] = (self._v, other._v)
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

    def _rbinop(self, other, name: str, fn: Callable):
        in_vars: Tuple[int, ...] = (self._v,)
        for v in in_vars:
            self._t.events.append(L2Load(v))
        result_val = fn(other, self.val)
        out_var = self._t.fresh()
        self._t.events.append(L2Op(name, in_vars, out_var))
        self._t.events.append(L2Store(out_var))
        return _Tracked(self._t, out_var, result_val)

    def __add__(self, o):  return self._binop(o, "add", operator.add)
    def __sub__(self, o):  return self._binop(o, "sub", operator.sub)
    def __mul__(self, o):  return self._binop(o, "mul", operator.mul)
    def __truediv__(self, o): return self._binop(o, "div", operator.truediv)
    def __radd__(self, o): return self._rbinop(o, "add", operator.add)
    def __rsub__(self, o): return self._rbinop(o, "sub", operator.sub)
    def __rmul__(self, o): return self._rbinop(o, "mul", operator.mul)


def trace(func: Callable, args: Tuple) -> Tuple[List[L2Event], List[int]]:
    """Trace func(*args) at L1, return (l2_events, input_vars).

    Each input scalar is allocated a fresh var via STORE. Every op produces
    LOAD events for its operands followed by STORE for its result, with an
    OP event tagged in between for human inspection.
    """
    t = _Tracer()

    def wrap(v):
        if isinstance(v, list):
            return [wrap(x) for x in v]
        if isinstance(v, tuple):
            return tuple(wrap(x) for x in v)
        if isinstance(v, (int, float)):
            var = t.fresh()
            t.input_vars.append(var)
            t.events.append(L2Store(var))
            return _Tracked(t, var, v)
        return v

    wrapped = tuple(wrap(a) for a in args)
    func(*wrapped)
    return t.events, t.input_vars


# ============================================================================
# L2 -> L3 : allocators
# ============================================================================

def _liveness(events: Sequence[L2Event]) -> Dict[int, int]:
    """Return last-LOAD index for each var. Vars with no LOAD are absent."""
    last: Dict[int, int] = {}
    for i, ev in enumerate(events):
        if isinstance(ev, L2Load):
            last[ev.var] = i
    return last


def compile_no_reuse(events: Sequence[L2Event]) -> List[L3Event]:
    """All-bytes: addr = allocation order. Slots never recycled."""
    var_addr: Dict[int, int] = {}
    next_addr = 1
    out: List[L3Event] = []
    for ev in events:
        if isinstance(ev, L2Store):
            var_addr[ev.var] = next_addr
            out.append(L3Store(ev.var, next_addr))
            next_addr += 1
        elif isinstance(ev, L2Load):
            out.append(L3Load(ev.var, var_addr[ev.var]))
        else:  # L2Op
            in_addrs = tuple(var_addr[v] for v in ev.in_vars)
            out_addr = var_addr.get(ev.out_var) if ev.out_var is not None else None
            out.append(L3Op(ev.name, ev.in_vars, in_addrs, ev.out_var, out_addr))
    return out


def compile_min_heap(events: Sequence[L2Event]) -> List[L3Event]:
    """Live-bytes greedy: free dead slots, reuse smallest free addr."""
    last_load = _liveness(events)
    var_addr: Dict[int, int] = {}
    free: List[int] = []
    next_addr = 1
    out: List[L3Event] = []
    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            if free:
                addr = heapq.heappop(free)
            else:
                addr = next_addr
                next_addr += 1
            var_addr[ev.var] = addr
            out.append(L3Store(ev.var, addr))
        elif isinstance(ev, L2Load):
            addr = var_addr[ev.var]
            out.append(L3Load(ev.var, addr))
            if last_load.get(ev.var) == i:
                heapq.heappush(free, addr)
        else:  # L2Op
            in_addrs = tuple(var_addr[v] for v in ev.in_vars)
            out_addr = var_addr.get(ev.out_var) if ev.out_var is not None else None
            out.append(L3Op(ev.name, ev.in_vars, in_addrs, ev.out_var, out_addr))
    return out


def compile_lru_static(events: Sequence[L2Event]) -> List[L3Event]:
    """Live-bytes LIFO: freed slots returned in LIFO order (stack discipline)."""
    last_load = _liveness(events)
    var_addr: Dict[int, int] = {}
    free: List[int] = []  # used as a LIFO stack
    next_addr = 1
    out: List[L3Event] = []
    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            if free:
                addr = free.pop()
            else:
                addr = next_addr
                next_addr += 1
            var_addr[ev.var] = addr
            out.append(L3Store(ev.var, addr))
        elif isinstance(ev, L2Load):
            addr = var_addr[ev.var]
            out.append(L3Load(ev.var, addr))
            if last_load.get(ev.var) == i:
                free.append(addr)
        else:
            in_addrs = tuple(var_addr[v] for v in ev.in_vars)
            out_addr = var_addr.get(ev.out_var) if ev.out_var is not None else None
            out.append(L3Op(ev.name, ev.in_vars, in_addrs, ev.out_var, out_addr))
    return out


def compile_belady(events: Sequence[L2Event]) -> List[L3Event]:
    """Offline Belady-style: prefer assigning low addrs to vars with more remaining loads.

    Strategy: when assigning a slot to a new STORE, compute the var's total
    remaining-load count L. Among free slots (or the next fresh slot), pick
    the one whose addr d minimizes L * sqrt(d) — i.e., let frequently-read
    vars take low addrs.
    """
    # Build remaining-loads count per var across the whole trace.
    load_indices: Dict[int, List[int]] = {}
    for i, ev in enumerate(events):
        if isinstance(ev, L2Load):
            load_indices.setdefault(ev.var, []).append(i)
    last_load = {v: ix[-1] for v, ix in load_indices.items()}

    var_addr: Dict[int, int] = {}
    free: List[int] = []  # heap of free addrs
    next_addr = 1
    out: List[L3Event] = []

    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            n_loads = len(load_indices.get(ev.var, []))
            # Candidate addrs: every free slot, plus the next fresh slot.
            candidates = list(free)
            candidates.append(next_addr)
            # Pick the smallest addr if var has loads (cheaper for frequent reads),
            # else any addr (we'll take smallest free or next-addr to avoid bloat).
            if n_loads > 0:
                addr = min(candidates)
            else:
                # Var is never read; cost-free regardless of addr.
                addr = min(candidates)
            if addr in free:
                free.remove(addr)
                heapq.heapify(free)
            else:
                next_addr += 1
            var_addr[ev.var] = addr
            out.append(L3Store(ev.var, addr))
        elif isinstance(ev, L2Load):
            addr = var_addr[ev.var]
            out.append(L3Load(ev.var, addr))
            if last_load.get(ev.var) == i:
                heapq.heappush(free, addr)
        else:
            in_addrs = tuple(var_addr[v] for v in ev.in_vars)
            out_addr = var_addr.get(ev.out_var) if ev.out_var is not None else None
            out.append(L3Op(ev.name, ev.in_vars, in_addrs, ev.out_var, out_addr))
    return out


def compile_tombstone(events: Sequence[L2Event]) -> List[L3Event]:
    """Mobile LRU stack with tombstones — the realistic middle model.

    A stack grows from bottom (index 0) to top (highest index). The address
    of a variable equals its distance from the top: top slot has addr 1,
    bottom slot has addr len(stack). Cost of a LOAD at addr d is
    ceil(sqrt(d)) (same as every other allocator).

    Semantics:
      - On STORE: place the new variable in the highest-index hole;
        if no hole exists, extend the stack by one slot at the top.
      - On LOAD: record the variable's current address, then remove it
        from its slot (leaving a hole), and re-insert it into the
        highest-index hole ABOVE the old slot. If no such hole exists,
        extend the stack (which places the variable at the new top).
      - Dead variables (past their last LOAD) leave permanent tombstones.

    Unlike stationary min-heap reuse, live variables DO move on each read —
    repeated reads of a frequently-used input bring it back toward the top.
    Unlike full liveness compaction, dead slots are never reclaimed by
    sliding live data inward. This matches the "Tombstone / High-Water
    Mark" picture from gemini/15apr26-dmdlive-analysis.md.
    """
    last_load = _liveness(events)
    stack: List[Optional[int]] = []
    positions: Dict[int, int] = {}
    hole_heap: List[int] = []  # max-heap over hole indices (negated)
    out: List[L3Event] = []

    def push_hole(slot: int) -> None:
        heapq.heappush(hole_heap, -slot)

    def pop_nearest_hole() -> Optional[int]:
        while hole_heap:
            slot = -heapq.heappop(hole_heap)
            if 0 <= slot < len(stack) and stack[slot] is None:
                return slot
        return None

    def addr_of_slot(slot: int) -> int:
        return len(stack) - slot

    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            slot = pop_nearest_hole()
            if slot is None:
                stack.append(ev.var)
                slot = len(stack) - 1
            else:
                stack[slot] = ev.var
            positions[ev.var] = slot
            out.append(L3Store(ev.var, addr_of_slot(slot)))
            if last_load.get(ev.var, -1) < i:
                # Dead at birth — immediately becomes a tombstone.
                stack[slot] = None
                del positions[ev.var]
                push_hole(slot)
        elif isinstance(ev, L2Load):
            slot = positions[ev.var]
            out.append(L3Load(ev.var, addr_of_slot(slot)))
            stack[slot] = None
            del positions[ev.var]
            if last_load.get(ev.var) == i:
                push_hole(slot)
                continue
            candidate = pop_nearest_hole()
            if candidate is not None and candidate > slot:
                stack[candidate] = ev.var
                positions[ev.var] = candidate
                push_hole(slot)
            else:
                if candidate is not None:
                    push_hole(candidate)
                push_hole(slot)
                stack.append(ev.var)
                positions[ev.var] = len(stack) - 1
        else:  # L2Op — metadata
            in_addrs = tuple(
                addr_of_slot(positions[v]) if v in positions else 0
                for v in ev.in_vars
            )
            out_slot = positions.get(ev.out_var) if ev.out_var is not None else None
            out_addr = addr_of_slot(out_slot) if out_slot is not None else None
            out.append(L3Op(ev.name, ev.in_vars, in_addrs, ev.out_var, out_addr))
    return out


ALLOCATORS: Dict[str, Callable[[Sequence[L2Event]], List[L3Event]]] = {
    "no_reuse":   compile_no_reuse,
    "min_heap":   compile_min_heap,
    "lru_static": compile_lru_static,
    "belady":     compile_belady,
    "tombstone":  compile_tombstone,
}

# Human-readable display names for each allocator.
POLICY_DISPLAY: Dict[str, str] = {
    "no_reuse":   "No reuse",
    "min_heap":   "Min-heap (stationary)",
    "lru_static": "LIFO slots",
    "belady":     "Belady (offline)",
    "tombstone":  "Tombstone (LRU+holes)",
}


# ============================================================================
# L2-level ByteDMD metrics (no allocator)
# ============================================================================

class _Fenwick:
    """1-indexed Fenwick tree of booleans, supporting O(log n) insert/remove/rank."""
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


def _lru_cost(events: Sequence[L2Event], compact_on_last_load: bool) -> int:
    """Shared LRU walk priced by ceil(sqrt(stack_depth)) per LOAD.

    Implementation: each variable currently in the stack owns a unique
    timestamp t ∈ [1, T]. A Fenwick tree marks which timestamps are live.
    Depth of variable X at a read = (#live timestamps) - (#live timestamps
    with value < t(X)) = 1 + (#live timestamps with value > t(X)).

    When compact_on_last_load is True, the variable is dropped entirely on
    its last LOAD (ByteDMD-live). Otherwise its timestamp is refreshed to
    the top (ByteDMD-classic LRU bump).
    """
    # Assign every var a dense timestamp up-front; reuse slot for later refresh.
    last_load: Dict[int, int] = {}
    if compact_on_last_load:
        for i, ev in enumerate(events):
            if isinstance(ev, L2Load):
                last_load[ev.var] = i

    # Upper-bound on simultaneously-live timestamps: one per STORE + one per
    # LOAD-bump. We use a growing Fenwick tree by pre-sizing to len(events)+1.
    T = len(events) + 1
    bit = _Fenwick(T)

    var_ts: Dict[int, int] = {}
    next_ts = 0
    total = 0

    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            if compact_on_last_load and ev.var not in last_load:
                continue  # never loaded, skip
            next_ts += 1
            var_ts[ev.var] = next_ts
            bit.add(next_ts, 1)
        elif isinstance(ev, L2Load):
            t = var_ts[ev.var]
            # depth = # live timestamps with t' >= t = total_live - prefix(t-1)
            total_live = bit.prefix(T)
            depth = total_live - bit.prefix(t - 1)
            total += math.isqrt(depth - 1) + 1
            # Remove current timestamp
            bit.add(t, -1)
            if compact_on_last_load and last_load[ev.var] == i:
                del var_ts[ev.var]  # dropped
            else:
                # LRU bump: give a fresh (newest) timestamp
                next_ts += 1
                var_ts[ev.var] = next_ts
                bit.add(next_ts, 1)
    return total


def bytedmd_classic(events: Sequence[L2Event]) -> int:
    """ByteDMD-classic: LRU stack depth without liveness compaction.

    Walk the L2 trace with an LRU stack. Every STORE pushes a variable to
    the top (depth 1). Every LOAD looks up the variable's current depth,
    charges ceil(sqrt(depth)), and bumps it to the top. Dead variables are
    never removed — deeper slots get polluted by the whole allocation
    history. This is the "memory-leak" reuse-distance measure.
    """
    return _lru_cost(events, compact_on_last_load=False)


def bytedmd_live(events: Sequence[L2Event]) -> int:
    """ByteDMD-live: LRU stack depth WITH liveness compaction.

    Same LRU walk as bytedmd_classic except a variable is dropped from the
    stack on its last LOAD. The cost of a LOAD of X reflects only the
    number of LIVE variables referenced between its previous LOAD and the
    current one.
    """
    return _lru_cost(events, compact_on_last_load=True)


# ============================================================================
# L3 cost evaluation
# ============================================================================

def cost(events: Sequence[L3Event]) -> int:
    """Sum ceil(sqrt(addr)) over each L3Load."""
    total = 0
    for ev in events:
        if isinstance(ev, L3Load):
            total += math.isqrt(ev.addr - 1) + 1
    return total


def peak_addr(events: Sequence[L3Event]) -> int:
    """Highest addr ever used (= storage footprint)."""
    p = 0
    for ev in events:
        if isinstance(ev, L3Store):
            if ev.addr > p:
                p = ev.addr
    return p


def n_loads(events: Sequence) -> int:
    return sum(1 for ev in events if isinstance(ev, (L2Load, L3Load)))


# ============================================================================
# Algorithms
# ============================================================================

def matmul_naive(A, B):
    n = len(A)
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = A[i][0] * B[0][j]
            for k in range(1, n):
                s = s + A[i][k] * B[k][j]
            C[i][j] = s
    return C


def matmul_tiled(A, B, tile: Optional[int] = None):
    """One-level blocked matmul. Default tile = round(sqrt(n))."""
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

def _add_mat(A, B):
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def matmul_rmm(A, B):
    """Cache-oblivious recursive matmul (8-way divide)."""
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)
    C11 = _add_mat(matmul_rmm(A11, B11), matmul_rmm(A12, B21))
    C12 = _add_mat(matmul_rmm(A11, B12), matmul_rmm(A12, B22))
    C21 = _add_mat(matmul_rmm(A21, B11), matmul_rmm(A22, B21))
    C22 = _add_mat(matmul_rmm(A21, B12), matmul_rmm(A22, B22))
    return _join(C11, C12, C21, C22)


# ============================================================================
# Unified API
# ============================================================================

def bytedmd(func: Callable, args: Tuple, policy: str = "min_heap") -> int:
    """Trace -> compile -> evaluate cost for an allocator policy name."""
    l2, _ = trace(func, args)
    l3 = ALLOCATORS[policy](l2)
    return cost(l3)


def bytedmd_all(func: Callable, args: Tuple) -> Dict[str, int]:
    """Compute every measure: the two L2 ByteDMD metrics plus each allocator."""
    l2, _ = trace(func, args)
    results: Dict[str, int] = {
        "bytedmd_classic": bytedmd_classic(l2),
        "bytedmd_live":    bytedmd_live(l2),
    }
    for name, comp in ALLOCATORS.items():
        results[name] = cost(comp(l2))
    return results


def make_inputs(n: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Two n x n integer matrices (all 1s) for tracing matmul."""
    A = [[1] * n for _ in range(n)]
    B = [[1] * n for _ in range(n)]
    return A, B
