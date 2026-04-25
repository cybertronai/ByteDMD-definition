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

    Two-stack model: input scalars live on an argument stack in input
    order, NOT the geometric stack. No L2Store is emitted for them; the
    heuristic prices their first L2Load at the arg-stack cost, then
    treats them as just-stored on the geometric stack.

    Trailing epilogue: every scalar in the return value is loaded once
    at the end, modelling the program-exit output read.
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
            return _Tracked(t, var, v)
        return v

    wrapped = tuple(wrap(a) for a in args)
    result = func(*wrapped)

    def emit_output_loads(v):
        if isinstance(v, _Tracked):
            t.events.append(L2Load(v._v))
        elif isinstance(v, (list, tuple)):
            for x in v:
                emit_output_loads(x)
        elif isinstance(v, dict):
            for x in v.values():
                emit_output_loads(x)

    emit_output_loads(result)
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


def compile_ripple(events: Sequence[L2Event]) -> List[L3Event]:
    """Ripple-shift cascaded eviction — realistic hardware allocator.

    Fixes the "stack inflation bug" in compile_tombstone: when a LOAD finds
    no hole above its slot, tombstone APPENDS to a new top, pushing every
    dormant variable outward by 1. For RMM at N=32 this inflates the
    physical stack to ~31 k slots while the true live working set stays
    near 2.6 k — the dormant parents drift into the abyss and the cost
    drifts toward Classic DMD.

    Ripple Shift models a real cascaded-eviction cache (like a shift
    register or systolic array). The touched variable is placed at the
    premium center (addr = 1); dormant variables shift outward by 1 until
    the cascade meets a hole, which absorbs the ripple. Crucially, the
    physical footprint stays clamped to the live high-water mark.

    Depth is computed over a Fenwick tree indexed by timestamps: each
    occupied slot (live or tombstone) has a unique timestamp; higher
    timestamps are closer to the top. Depth of variable v at timestamp
    t = total_active - prefix(t - 1) = count of slots with timestamp >= t.
    Reference: gemini/ripple-shift.md.
    """
    last_load = _liveness(events)
    T_max = len(events) + 2
    bit = _Fenwick(T_max)

    var_ts: Dict[int, int] = {}
    holes: List[int] = []  # max-heap of hole timestamps (negated)
    out: List[L3Event] = []
    next_ts = 0

    def pop_max_hole(min_ts: int) -> Optional[int]:
        """Pop the largest hole timestamp strictly greater than min_ts."""
        while holes:
            h = -holes[0]
            if h > min_ts:
                heapq.heappop(holes)
                return h
            break
        return None

    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            next_ts += 1
            t = next_ts
            var_ts[ev.var] = t
            bit.add(t, 1)
            out.append(L3Store(ev.var, 1))

            h = pop_max_hole(0)
            if h is not None:
                bit.add(h, -1)

            if last_load.get(ev.var, -1) < i:
                heapq.heappush(holes, -t)
                del var_ts[ev.var]

        elif isinstance(ev, L2Load):
            t = var_ts[ev.var]
            total_active = bit.prefix(T_max)
            depth = total_active - bit.prefix(t - 1)
            out.append(L3Load(ev.var, depth))

            next_ts += 1
            new_t = next_ts
            var_ts[ev.var] = new_t
            bit.add(new_t, 1)

            h = pop_max_hole(t)
            if h is not None:
                bit.add(h, -1)
                heapq.heappush(holes, -t)
            else:
                bit.add(t, -1)

            if last_load.get(ev.var) == i:
                heapq.heappush(holes, -new_t)
                del var_ts[ev.var]

        else:  # L2Op — metadata only
            total_active = bit.prefix(T_max)
            in_addrs = tuple(
                (total_active - bit.prefix(var_ts[v] - 1)) if v in var_ts else 0
                for v in ev.in_vars
            )
            out_addr = None
            if ev.out_var is not None and ev.out_var in var_ts:
                out_addr = total_active - bit.prefix(var_ts[ev.out_var] - 1)
            out.append(L3Op(ev.name, ev.in_vars, in_addrs, ev.out_var, out_addr))

    return out


ALLOCATORS: Dict[str, Callable[[Sequence[L2Event]], List[L3Event]]] = {
    "no_reuse":   compile_no_reuse,
    "min_heap":   compile_min_heap,
    "lru_static": compile_lru_static,
    "belady":     compile_belady,
    "tombstone":  compile_tombstone,
    "ripple":     compile_ripple,
}

# Human-readable display names for each allocator.
POLICY_DISPLAY: Dict[str, str] = {
    "no_reuse":   "No reuse",
    "min_heap":   "Min-heap (stationary)",
    "lru_static": "LIFO slots",
    "belady":     "Belady (offline)",
    "tombstone":  "Tombstone (LRU+holes)",
    "ripple":     "Ripple Shift",
}


# ============================================================================
# Static allocation bounds (MWIS lower bound + min-heap upper bound)
# ============================================================================

import bisect


@dataclass
class _Interval:
    var_id: int
    start: int
    end: int
    reads: int


def _extract_intervals(events: Sequence[L2Event]) -> List[_Interval]:
    """Build liveness intervals with per-variable read counts."""
    starts: Dict[int, int] = {}
    ends: Dict[int, int] = {}
    reads: Dict[int, int] = {}
    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            starts[ev.var] = i
            if ev.var not in ends:
                ends[ev.var] = i
        elif isinstance(ev, L2Load):
            ends[ev.var] = i
            reads[ev.var] = reads.get(ev.var, 0) + 1
    out: List[_Interval] = []
    for var, start in starts.items():
        r = reads.get(var, 0)
        if r > 0:
            out.append(_Interval(var, start, ends[var], r))
    return out


def _mwis_weight(intervals: List[_Interval]) -> int:
    """Maximum Weight Independent Set on an interval graph via O(N log N) DP.

    Returns the maximum total reads that can be served by a single physical
    address — equivalently, the max-weight set of non-overlapping intervals.
    """
    if not intervals:
        return 0
    sorted_ivs = sorted(intervals, key=lambda x: x.end)
    n = len(sorted_ivs)
    ends = [iv.end for iv in sorted_ivs]

    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        iv = sorted_ivs[i - 1]
        idx = bisect.bisect_left(ends, iv.start)
        take = iv.reads + dp[idx]
        leave = dp[i - 1]
        dp[i] = max(take, leave)
    return dp[n]


def mwis_lower_bound(events: Sequence[L2Event]) -> int:
    """Strict lower bound on the cost of any static (stationary-addr) allocator.

    Uses "water-pouring": the MWIS weight W is the maximum total reads any
    single physical address can serve (because non-overlapping intervals can
    share an address, but overlapping ones cannot). We pack W reads into
    addr 1 at cost 1 each, W reads into addr 2 at cost ceil(sqrt(2)) each,
    etc., until all reads are accounted for. Since we're distributing reads
    as cheaply as possible without violating physical constraints, the result
    is a strict lower bound on the optimal static allocation cost.

    Reference: gemini/mwis-bounds.md.
    """
    intervals = _extract_intervals(events)
    if not intervals:
        return 0
    max_reads_per_addr = _mwis_weight(intervals)
    total_reads = sum(iv.reads for iv in intervals)

    total_cost = 0
    remaining = total_reads
    addr = 1
    while remaining > 0:
        take = min(remaining, max_reads_per_addr)
        total_cost += take * (math.isqrt(addr - 1) + 1)
        remaining -= take
        addr += 1
    return total_cost


def static_upper_bound(events: Sequence[L2Event]) -> int:
    """Achievable upper bound: cost of the greedy min-heap static allocator.

    Because min-heap is a valid physical assignment (each variable gets a
    fixed address, freed slots are recycled to the smallest available), its
    cost is an upper bound on the optimal static allocation.
    """
    return cost(compile_min_heap(events))


def _extract_cliques(events: Sequence[L2Event],
                     intervals: List[_Interval]) -> List[List[int]]:
    """Sweep the trace to find maximal cliques of the interval graph.

    A clique is a set of variables all alive at the same moment.  We record
    a snapshot at every STORE (new variable enters) and at every last-LOAD
    (variable about to leave), then keep only maximal ones.
    """
    valid_vars = {iv.var_id for iv in intervals}
    last_load: Dict[int, int] = {}
    for i, ev in enumerate(events):
        if isinstance(ev, L2Load) and ev.var in valid_vars:
            last_load[ev.var] = i

    active: set = set()
    all_cliques: list = []
    for i, ev in enumerate(events):
        if isinstance(ev, L2Store) and ev.var in valid_vars:
            active.add(ev.var)
            all_cliques.append(frozenset(active))
        elif isinstance(ev, L2Load) and ev.var in valid_vars:
            if last_load.get(ev.var) == i:
                all_cliques.append(frozenset(active))
                active.discard(ev.var)

    # Keep only maximal cliques (no subset of another).
    cliques_sorted = sorted(all_cliques, key=len, reverse=True)
    maximal: list = []
    for c in cliques_sorted:
        if not any(c.issubset(mc) for mc in maximal):
            maximal.append(c)
    return maximal


def _mwis_dp(intervals: List[_Interval]) -> Tuple[int, List[_Interval]]:
    """Weighted interval scheduling (MWIS): returns (weight, selected_intervals).

    O(N log N) DP with backtracking to extract the selected set.
    """
    if not intervals:
        return 0, []
    sorted_ivs = sorted(intervals, key=lambda x: x.end)
    n = len(sorted_ivs)
    ends = [iv.end for iv in sorted_ivs]

    # p[i] = index of latest interval ending strictly before sorted_ivs[i] starts
    p = [0] * n
    for i in range(n):
        p[i] = bisect.bisect_left(ends, sorted_ivs[i].start)

    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        take = sorted_ivs[i - 1].reads + dp[p[i - 1]]
        dp[i] = max(dp[i - 1], take)

    # Backtrack to find which intervals are selected.
    selected: List[_Interval] = []
    i = n
    while i > 0:
        take = sorted_ivs[i - 1].reads + dp[p[i - 1]]
        if take >= dp[i - 1]:
            selected.append(sorted_ivs[i - 1])
            i = p[i - 1]
        else:
            i -= 1
    return dp[n], selected


def lp_lower_bound(events: Sequence[L2Event]) -> float:
    """Tight continuous lower bound via discrete-calculus + greedy MWIS layering.

    Uses the identity  sqrt(k) = sum_{j=1}^{k} (sqrt(j) - sqrt(j-1))  and
    computes M_c (max reads packable into c addresses) for c = 1..omega by
    peeling off successive MWIS layers from the interval graph.

    For interval graphs (perfect graphs), greedy MWIS layering yields the
    exact optimal c-coloring weight, equivalent to the LP relaxation from
    gemini/unimodular-lp.md but computed in O(omega * N log N) instead of
    solving omega separate LPs.

    The bound uses continuous sqrt (not ceil), so it is strictly <=
    the discrete ceil(sqrt) cost — a valid lower bound.
    """
    intervals = _extract_intervals(events)
    if not intervals:
        return 0.0
    R_total = sum(iv.reads for iv in intervals)

    # Peel MWIS layers to build M[c] for c = 0, 1, 2, ...
    # Cap at MAX_LAYERS to keep runtime manageable for large traces.
    # The bound remains valid (just slightly looser) because truncated
    # layers contribute diminishing sqrt-difference weights.
    MAX_LAYERS = 300
    remaining = list(intervals)
    M: List[int] = [0]
    for _ in range(MAX_LAYERS):
        if not remaining:
            break
        w, selected = _mwis_dp(remaining)
        if w == 0:
            break
        M.append(M[-1] + w)
        selected_ids = {id(iv) for iv in selected}
        remaining = [iv for iv in remaining if id(iv) not in selected_ids]
    omega = len(M) - 1
    # Pad so M[omega] = R_total (all intervals eventually assigned).
    if M[-1] < R_total:
        M.append(R_total)
        omega += 1

    lb = 0.0
    for j in range(1, omega + 1):
        cost_diff = math.sqrt(j) - math.sqrt(j - 1)
        remaining_reads = R_total - M[j - 1]
        if remaining_reads <= 0:
            break
        lb += cost_diff * remaining_reads
    return lb


def _static_opt_intervals(
    events: Sequence[L2Event],
    input_arg_idx: Optional[Dict[int, int]] = None,
):
    """Yield (t_start, t_end, floor) for each stable interval where the
    set of live vars and their density ranking is constant. floor =
    Σ_i ρ_{(i)} · sqrt(i) over currently-live vars. This is the
    *geometric-stack* contribution; under the Two-Stack model the
    compulsory arg-stack first-load cost is added separately by
    `static_opt_lb`. Inputs are scoped [first L2Load, last L2Load]
    here — they live on the (free) arg stack before first use, and
    only enter the geometric stack on promotion. Their first read is
    excluded from the density (`reads -= 1`) since it is paid against
    the arg stack, not the geometric stack.
    """
    input_arg_idx = input_arg_idx or {}
    starts: Dict[int, int] = {}
    ends: Dict[int, int] = {}
    reads: Dict[int, int] = {}
    stored: set = set()
    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            stored.add(ev.var)
            if ev.var not in starts:
                starts[ev.var] = i
            if ev.var not in ends:
                ends[ev.var] = i
        elif isinstance(ev, L2Load):
            if ev.var not in starts:
                if ev.var in input_arg_idx:
                    # Two-Stack: input vars are born on the geometric
                    # stack at the moment of their first load; before
                    # that they sit (free) on the arg stack.
                    starts[ev.var] = i
                else:
                    starts[ev.var] = 0 if ev.var not in stored else i
            ends[ev.var] = i
            reads[ev.var] = reads.get(ev.var, 0) + 1

    # The compulsory first read of each input is charged against the
    # arg stack by `static_opt_lb`, so drop it from the geometric
    # density to avoid double-counting.
    for v in input_arg_idx:
        if v in reads:
            reads[v] -= 1

    densities: Dict[int, float] = {}
    for v, r in reads.items():
        if r <= 0:
            continue
        lifespan = max(1, ends[v] - starts[v] + 1)
        densities[v] = r / lifespan
    if not densities:
        return

    births = sorted((starts[v], densities[v], v) for v in densities)
    deaths = sorted((ends[v] + 1, densities[v], v) for v in densities)

    active: List[Tuple[float, int]] = []
    t_prev = 0
    bi = di = 0
    n = len(births)

    while bi < n or di < n:
        t_b = births[bi][0] if bi < n else float("inf")
        t_d = deaths[di][0] if di < n else float("inf")
        t_next = t_b if t_b <= t_d else t_d

        if t_next > t_prev:
            s = 0.0
            for rank, (rho, _v) in enumerate(reversed(active), start=1):
                s += rho * math.sqrt(rank)
            yield (t_prev, t_next, s)

        while bi < n and births[bi][0] == t_next:
            _, rho, v = births[bi]
            bisect.insort(active, (rho, v))
            bi += 1
        while di < n and deaths[di][0] == t_next:
            _, rho, v = deaths[di]
            pos = bisect.bisect_left(active, (rho, v))
            active.pop(pos)
            di += 1

        t_prev = t_next


def static_opt_floor_curve(
    events: Sequence[L2Event],
    input_arg_idx: Optional[Dict[int, int]] = None,
) -> Tuple[List[int], List[float]]:
    """Per-tick geometric-stack TU LP floor (the integrand of the geometric
    portion of `static_opt_lb`).

    Returns (times, floors) suitable for a `drawstyle="steps-post"` plot:
    floors[k] is held over [times[k], times[k+1]). The area under the
    curve equals the geometric-stack portion of `static_opt_lb`; the
    compulsory arg-stack first-load cost is reported separately by
    `static_opt_lb`. See gemini/optimal-static-floor.md and
    gemini/fix-spacedmd-bug.md.
    """
    times: List[int] = []
    floors: List[float] = []
    last_end = 0
    for t_start, t_end, s in _static_opt_intervals(events, input_arg_idx):
        if not times:
            times.append(t_start)
            floors.append(s)
        elif s != floors[-1]:
            times.append(t_start)
            floors.append(s)
        last_end = t_end
    if times and last_end > times[-1]:
        # Right-edge marker so steps-post draws the final segment.
        times.append(last_end)
        floors.append(floors[-1])
    return times, floors


def static_opt_lb(
    events: Sequence[L2Event],
    input_arg_idx: Optional[Dict[int, int]] = None,
) -> float:
    """Totally-unimodular LP lower bound on optimal static-allocator cost
    (see gemini/optimal-static-floor.md).

    For each tick t, the minimum-energy fractional layout by the
    Rearrangement Inequality places live vars at physical ranks
    1, 2, ... in decreasing order of global density
    ρ_V = reads(V) / lifespan(V), so the per-tick floor is

        Floor(t) = Σ_{i=1}^{A_t} ρ_{(i)} * sqrt(i)

    with ρ_{(i)} ≥ ρ_{(i+1)} over the vars live at t. Summing over all
    ticks yields a strict fractional lower bound on any static
    allocator — each variable pays at least ρ * sqrt(rank) per tick of
    its lifetime, and total-over-lifetime = reads * sqrt(rank) is the
    cost if the allocator pinned it to its globally optimal density-
    rank slot. Continuous sqrt (not ceil) so the value is ≤ the
    integer ceil-cost of any realized static layout.

    Relation to dynamic optDMD (bytedmd_opt): optDMD can teleport
    variables between reads for free, so its bound can be much lower
    than this when the trace has long-dormant variables (the "phase"
    counter-example in the reference). Therefore this bound can
    *exceed* bytedmd_opt and is informative exactly for traces where
    static allocation is the constraint (large overlapping lifetimes).

    Two-Stack semantics (gemini/fix-spacedmd-bug.md): inputs sit on the
    free arg stack until their first load, at which point they enter
    the geometric stack with the rest of the live set. Their compulsory
    first read is priced at the arg-stack position (matching
    bytedmd_live / bytedmd_classic / space_dmd) and is added on top of
    the LP floor; the first read is not counted in the density used
    inside the LP, so the geometric Pigeonhole sum does not double-
    charge for it.
    """
    input_arg_idx = input_arg_idx or {}

    # Compulsory arg-stack cost: first load of each input pays
    # ceil(sqrt(arg_idx)) at the moment of promotion. Matches
    # _lru_cost's first-load handling.
    first_load_cost = 0.0
    seen: set = set()
    for ev in events:
        if isinstance(ev, L2Load) and ev.var in input_arg_idx \
                and ev.var not in seen:
            first_load_cost += math.isqrt(
                max(0, input_arg_idx[ev.var] - 1)) + 1
            seen.add(ev.var)

    geom_cost = sum(
        (t_end - t_start) * s
        for t_start, t_end, s in _static_opt_intervals(
            events, input_arg_idx))

    return first_load_cost + geom_cost


def greedy_freq_upper_bound(events: Sequence[L2Event]) -> float:
    """Achievable upper bound: frequency-first greedy static allocation.

    Sort variables by read count (descending), then assign each to the
    lowest-numbered address whose existing intervals don't overlap.
    Uses continuous sqrt for comparability with lp_lower_bound.
    """
    intervals = _extract_intervals(events)
    sorted_ivs = sorted(intervals, key=lambda x: x.reads, reverse=True)
    tracks: List[List[Tuple[int, int]]] = []
    total_cost = 0.0

    for iv in sorted_ivs:
        assigned = -1
        for k, track in enumerate(tracks):
            overlap = False
            for ts, te in track:
                if iv.start <= te and iv.end >= ts:
                    overlap = True
                    break
            if not overlap:
                track.append((iv.start, iv.end))
                assigned = k + 1
                break
        if assigned == -1:
            tracks.append([(iv.start, iv.end)])
            assigned = len(tracks)
        total_cost += iv.reads * math.sqrt(assigned)
    return total_cost


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


def _lru_cost(events: Sequence[L2Event],
              compact_on_last_load: bool,
              input_arg_idx: Optional[Dict[int, int]] = None) -> int:
    """Shared LRU walk priced by ceil(sqrt(stack_depth)) per LOAD, with
    optional two-stack argument pricing.

    Two-stack model: if `input_arg_idx` is provided, mapping input var →
    1-based position on the arg stack, the FIRST L2Load of an input var
    is priced at ceil(sqrt(arg_idx)) — read from the arg stack — and then
    the var is "promoted" onto the geometric stack as if freshly Stored.
    Subsequent L2Loads price at geom-stack depth as usual.

    When compact_on_last_load is True, the variable is dropped entirely on
    its last LOAD (ByteDMD-live). Otherwise its timestamp is refreshed to
    the top (ByteDMD-classic LRU bump).
    """
    input_arg_idx = input_arg_idx or {}
    pending = set(input_arg_idx)  # inputs whose first read hasn't happened

    last_load: Dict[int, int] = {}
    if compact_on_last_load:
        for i, ev in enumerate(events):
            if isinstance(ev, L2Load):
                last_load[ev.var] = i

    # Upper-bound on simultaneously-live timestamps: one per STORE + one per
    # LOAD-bump, plus one per input promotion. Pre-size to len(events)+K+1.
    T = len(events) + len(input_arg_idx) + 1
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
            if ev.var in pending:
                # First read: from arg stack, priced by input position.
                arg_idx = input_arg_idx[ev.var]
                total += math.isqrt(max(0, arg_idx - 1)) + 1
                pending.discard(ev.var)
                if compact_on_last_load and last_load.get(ev.var) == i:
                    continue  # read once and never again — no geom insert
                # Promote onto geometric stack as if just Stored.
                next_ts += 1
                var_ts[ev.var] = next_ts
                bit.add(next_ts, 1)
                continue
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


def bytedmd_classic(events: Sequence[L2Event],
                    input_arg_idx: Optional[Dict[int, int]] = None) -> int:
    """ByteDMD-classic: LRU stack depth without liveness compaction.

    Walk the L2 trace with an LRU stack. Every STORE pushes a variable to
    the top (depth 1). Every LOAD looks up the variable's current depth,
    charges ceil(sqrt(depth)), and bumps it to the top. Dead variables are
    never removed. If input_arg_idx is given, input vars pay arg-stack cost
    on first read, then are promoted to the geometric stack.
    """
    return _lru_cost(events, compact_on_last_load=False,
                     input_arg_idx=input_arg_idx)


def bytedmd_live(events: Sequence[L2Event],
                 input_arg_idx: Optional[Dict[int, int]] = None) -> int:
    """ByteDMD-live: LRU stack depth WITH liveness compaction.

    Same LRU walk as bytedmd_classic except a variable is dropped from the
    stack on its last LOAD. If input_arg_idx is given, input vars pay
    arg-stack cost on first read, then are promoted to the geometric stack.
    """
    return _lru_cost(events, compact_on_last_load=True,
                     input_arg_idx=input_arg_idx)


def opt_reuse_distances(
    events: Sequence[L2Event],
    input_arg_idx: Optional[Dict[int, int]] = None,
) -> Tuple[List[int], List[int]]:
    """Bélády MIN stack distance per load (see gemini/belady-min-lower-bound.md).

    For a load of V at time t_next with previous load at t_prev, the OPT
    stack distance is 1 + max over τ ∈ (t_prev, t_next) of the number of
    distinct live variables W whose next use at τ is strictly before
    t_next. By the Pigeonhole theorem in the referenced proof, MIN misses
    V at cache capacity c iff this max_rank exceeds c — so it simulates
    the eviction decision of any offline oracle with perfect future
    knowledge. Mattson's inclusion property makes this the smallest
    possible reuse distance achievable by any caching policy.

    First loads (no previous reference) pay the arg-stack position if the
    variable is an input, matching the bytedmd_classic/live convention;
    non-input cold stores are reported as distance 1.

    Implementation: for each reuse pair (a, b) in the global trace, the
    interval [max(a, t_prev+1), min(b-1, t_next-1)] is the span during
    which its variable contributes to V's rank (given b < t_next). A
    sweep-line over these intervals gives the peak overlap. Pairs are
    indexed by b for an O(log N) range query per load.
    """
    import bisect as _bisect
    from collections import defaultdict as _dd

    input_arg_idx = input_arg_idx or {}

    load_times: Dict[int, List[int]] = _dd(list)
    for i, ev in enumerate(events):
        if isinstance(ev, L2Load):
            load_times[ev.var].append(i)

    # All (b, a, var) reuse-interval endpoints, sorted by b.
    pairs: List[Tuple[int, int, int]] = []
    for var, times in load_times.items():
        for j in range(len(times) - 1):
            pairs.append((times[j + 1], times[j], var))
    pairs.sort()
    b_vals = [p[0] for p in pairs]

    out_times: List[int] = []
    out_dists: List[int] = []

    for i, ev in enumerate(events):
        if not isinstance(ev, L2Load):
            continue
        v = ev.var
        vts = load_times[v]
        pos = _bisect.bisect_left(vts, i)
        if pos == 0:
            out_times.append(i)
            out_dists.append(input_arg_idx.get(v, 1))
            continue
        t_prev = vts[pos - 1]
        lo = _bisect.bisect_right(b_vals, t_prev)
        hi = _bisect.bisect_left(b_vals, i)
        sw: List[Tuple[int, int]] = []
        for b, a, W in pairs[lo:hi]:
            if W == v:
                continue
            s = a if a > t_prev else t_prev + 1
            e = b - 1 if b - 1 < i - 1 else i - 1
            if s > e:
                continue
            sw.append((s, 1))
            sw.append((e + 1, -1))
        if not sw:
            rank = 1
        else:
            sw.sort()
            cur = 1
            rank = 1
            for _, d in sw:
                cur += d
                if cur > rank:
                    rank = cur
        out_times.append(i)
        out_dists.append(rank)
    return out_times, out_dists


def bytedmd_opt(events: Sequence[L2Event],
                input_arg_idx: Optional[Dict[int, int]] = None) -> int:
    """ByteDMD-opt: Bélády MIN lower bound on 2D-Manhattan routing cost.

    For each L2Load of V at time t_next, charges ceil(sqrt(max_rank[V]))
    where max_rank is the Bélády stack distance from `opt_reuse_distances`.
    By the proof in gemini/belady-min-lower-bound.md, this is a strict
    lower bound on the total energy of any dynamic spatial allocator
    (including bytedmd_live, bytedmd_classic, space_dmd, and the manual
    hand-placed schedule) under ceil(sqrt(addr)) pricing.
    """
    _, dists = opt_reuse_distances(events, input_arg_idx)
    total = 0
    for d in dists:
        total += math.isqrt(max(0, d - 1)) + 1
    return total


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
