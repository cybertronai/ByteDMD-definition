"""SpaceDMD — density-ranked spatial liveness heuristic.

Models an ahead-of-time (AOT) static compiler that pins variables to
physical slots based on access density (accesses per unit of lifespan).
High-density variables (hot scratchpads, inner-loop temporaries) get
low addresses; low-density variables (bulk main memory) sit farther out.

Cost of a LOAD = ceil(sqrt(rank_among_currently_live_vars)).

Based on gemini/space-dmd.md.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Dict, Optional, Sequence

from bytedmd_ir import L2Event, L2Load, L2Store


class _Fenwick:
    """1-indexed Fenwick tree over booleans (live/dead)."""
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


def space_dmd(events: Sequence[L2Event],
              input_arg_idx: Optional[Dict[int, int]] = None) -> int:
    """Sum of ceil(sqrt(live-rank)) per LOAD, where rank is assigned once
    up-front by access density = access_count / lifespan.

    Two-stack: when input_arg_idx is given (input var → 1-based arg
    position), input vars pay ceil(sqrt(arg_idx)) on their first Load (arg
    stack); subsequent Loads pay density-ranked cost on the geometric
    stack. Inputs have no L2Store in the trace; their birth is at the
    first L2Load.

    Pass 1: scan events, compute (birth, last_use, access_count) per var.
    Pass 2: sort vars by density desc (tie-break: access_count desc,
            birth asc, vid asc) and assign ranks 1..V.
    Pass 3: sweep events; at each index, (births → reads → deaths) against
            a Fenwick tree of live ranks so prefix(rank[v]) = live-rank of v.
    """
    input_arg_idx = input_arg_idx or {}
    pending = set(input_arg_idx)

    # Pass 1
    birth: dict[int, int] = {}
    last_use: dict[int, int] = {}
    access_count: dict[int, int] = defaultdict(int)
    first_load_of_input: dict[int, int] = {}
    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            birth[ev.var] = i
            last_use.setdefault(ev.var, i)
        elif isinstance(ev, L2Load):
            if ev.var in input_arg_idx and ev.var not in birth:
                # First Load of an input: treat as birth on the geom stack.
                # The cost of this Load itself comes from the arg stack.
                birth[ev.var] = i
                first_load_of_input[ev.var] = i
            last_use[ev.var] = i
            access_count[ev.var] += 1

    V = len(birth)
    if V == 0:
        return 0

    # Pass 2: density-ranked ordering. Tie-breaker is a pseudo-random
    # hash of the var id, NOT birth time / vid (gemini/fix-tiebreaker-
    # spacedmd-bug.md). Chronological tie-breakers let space_dmd silently
    # implement a "Tie-Breaker Teleportation" — equal-density vars whose
    # births are sorted in FIFO order get sequentially evicted from
    # rank 1 just as the next one is read, effectively a free dynamic
    # conveyor belt that no true static allocator can perform. Hashing
    # the vid scatters tied vars randomly across the live footprint and
    # closes the loophole, restoring space_dmd ≥ static_opt_lb.
    def priority(vid: int) -> tuple:
        lifespan = last_use[vid] - birth[vid] + 1
        density = access_count[vid] / lifespan
        h = int(hashlib.md5(str(vid).encode()).hexdigest(), 16)
        return (-density, -access_count[vid], h)

    sorted_vids = sorted(birth.keys(), key=priority)
    rank_map = {vid: i + 1 for i, vid in enumerate(sorted_vids)}

    # Pass 3: time sweep
    births_at = defaultdict(list)
    deaths_at = defaultdict(list)
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
