"""OptSpaceDMD — MWIS-inspired auto-scratchpad heuristic.

From gemini/optspacedmd.md. Fractures each variable's lifetime into
inter-access intervals (between a store and its first load, or between
two consecutive loads — each load "frees a DMA copy to a new address").
Charges each interval ceil(sqrt(peak_overlap)) where peak_overlap is
the maximum count of simultaneously-active intervals during its own
lifespan, evaluated STREAMING by end-time: intervals are registered in
the segment tree and queried as they end, so later-ending intervals
are not counted against earlier-ending ones. This streaming discipline
turns out to give a remarkably tight approximation of the hand-placed
manual cost on tiled / recursive matmul (within ~4-10%).

Complexity: O(E log T), where E = #L2 events, T ≤ E.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from typing import List, Sequence

from bytedmd_ir import L2Event, L2Load, L2Store

sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))


class _LazySegmentTree:
    """Lazy-prop range-add + range-max over 1..size."""
    __slots__ = ("size", "tree", "lazy")

    def __init__(self, size: int) -> None:
        self.size = size
        self.tree = [0] * (4 * size + 4)
        self.lazy = [0] * (4 * size + 4)

    def add_range(self, l: int, r: int, val: int) -> None:
        if l > r:
            return
        self._add(1, 1, self.size, l, r, val)

    def query_max(self, l: int, r: int) -> int:
        if l > r:
            return 0
        return self._query(1, 1, self.size, l, r)

    def _add(self, node, start, end, l, r, val):
        if r < start or l > end:
            return
        if l <= start and end <= r:
            self.tree[node] += val
            self.lazy[node] += val
            return
        lz = self.lazy[node]
        if lz:
            self.tree[2 * node] += lz;     self.lazy[2 * node] += lz
            self.tree[2 * node + 1] += lz; self.lazy[2 * node + 1] += lz
            self.lazy[node] = 0
        mid = (start + end) >> 1
        self._add(2 * node, start, mid, l, r, val)
        self._add(2 * node + 1, mid + 1, end, l, r, val)
        t1, t2 = self.tree[2 * node], self.tree[2 * node + 1]
        self.tree[node] = t1 if t1 > t2 else t2

    def _query(self, node, start, end, l, r):
        if r < start or l > end:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        lz = self.lazy[node]
        if lz:
            self.tree[2 * node] += lz;     self.lazy[2 * node] += lz
            self.tree[2 * node + 1] += lz; self.lazy[2 * node + 1] += lz
            self.lazy[node] = 0
        mid = (start + end) >> 1
        p1 = self._query(2 * node, start, mid, l, r)
        p2 = self._query(2 * node + 1, mid + 1, end, l, r)
        return p1 if p1 > p2 else p2


def opt_space_dmd(events: Sequence[L2Event]) -> int:
    """Streaming by-end-time peak overlap; charges ceil(sqrt(peak)) per
    interval. Matches the literal pseudocode in gemini/optspacedmd.md."""
    last_seen: dict[int, int] = {}
    by_end: dict[int, List[int]] = defaultdict(list)
    for i, ev in enumerate(events):
        t = i + 1
        if isinstance(ev, L2Store):
            last_seen[ev.var] = t
        elif isinstance(ev, L2Load):
            if ev.var in last_seen:
                by_end[t].append(last_seen[ev.var])
                last_seen[ev.var] = t

    if not by_end:
        return 0

    T_max = len(events)
    tree = _LazySegmentTree(T_max)
    total = 0
    for e in range(1, T_max + 1):
        starts = by_end.get(e)
        if not starts:
            continue
        for s in starts:
            tree.add_range(s, e - 1, 1)
        for s in starts:
            peak = tree.query_max(s, e - 1)
            total += math.isqrt(max(0, peak - 1)) + 1
    return total
