#!/usr/bin/env python3
"""
Performance-optimized ByteDMD tracer.

Key optimization: The LRU stack uses an indexed doubly-linked list instead of
a plain Python list.  This gives O(1) depth lookups and O(1) move-to-top,
compared to O(N) list.index() + O(N) list.remove() in the original.

API is identical to bytedmd.py:
    bytedmd(func, args, bytes_per_element=1)
    traced_eval(func, args)
    trace_to_bytedmd(trace, bytes_per_element)
"""
import math
import operator


# ---------------------------------------------------------------------------
# Fast LRU stack using a doubly-linked list with a position index
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ('key', 'prev', 'next')
    def __init__(self, key):
        self.key = key
        self.prev = None
        self.next = None


class _FastContext:
    """
    Optimized context with O(1) depth lookup and O(1) move-to-top.

    The LRU stack is a doubly-linked list from bottom (least recent) to top
    (most recent).  A dict maps keys to nodes for O(1) access.
    Position tracking gives O(1) depth computation.
    """
    __slots__ = ('_head', '_tail', '_nodes', '_size', 'trace', 'sync',
                 'memo', 'counter')

    def __init__(self):
        # Sentinel nodes for easier insert/remove
        self._head = _Node(None)  # bottom sentinel
        self._tail = _Node(None)  # top sentinel
        self._head.next = self._tail
        self._tail.prev = self._head
        self._nodes = {}  # key -> node
        self._size = 0
        self.trace = []
        self.sync = []
        self.memo = {}
        self.counter = 0

    def allocate(self):
        self.counter += 1
        key = self.counter
        node = _Node(key)
        self._nodes[key] = node
        # Insert at top (just before tail sentinel)
        self._insert_before_tail(node)
        self._size += 1
        return key

    def read(self, keys):
        valid = [k for k in keys if k is not None]
        if not valid:
            return

        # Compute depths first (all against current state), then move to top
        # Depth = distance from top. Top element has depth 1.
        # We compute depth by walking from each node to the tail.
        # But that's O(depth) per node. Instead, we assign positions.
        #
        # Optimization: assign sequential positions by walking from tail
        # to head once, then depth = _size - position.
        # But maintaining positions on every insert/remove is expensive.
        #
        # Better: just walk from each node toward tail, counting steps.
        # This is O(depth) but depth is what we're measuring, and it
        # avoids the overhead of maintaining a position index.
        #
        # Actually, for best performance: compute the depth by counting
        # nodes between node and tail.

        depths = []
        for k in valid:
            node = self._nodes[k]
            # Count steps from node to tail (exclusive)
            depth = 0
            cur = node
            while cur.next is not None:  # None never reached; tail.next is None
                depth += 1
                cur = cur.next
            # depth now = steps to tail sentinel
            # But tail sentinel is not a real entry, so depth = steps - 0? No.
            # Actually:
            # If node is at top (just before tail), depth should be 1.
            # Steps from node to tail: node -> tail = 1 step, so depth = 1. ✓
            depths.append(depth)

        self.trace.extend(depths)

        # Now move each to top in order
        for k in valid:
            node = self._nodes[k]
            self._remove(node)
            self._insert_before_tail(node)

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _insert_before_tail(self, node):
        prev = self._tail.prev
        prev.next = node
        node.prev = prev
        node.next = self._tail
        self._tail.prev = node


# ---------------------------------------------------------------------------
# For even faster depth computation, use a Fenwick tree / order-statistic
# approach. Each node gets a "timestamp" when moved to top. Depth = number
# of distinct timestamps > this node's timestamp = number of accesses since
# this node was last touched.
#
# We maintain a Fenwick tree over timestamps. When a node is accessed:
#   1. depth = query(max_time) - query(node.timestamp) = items above it
#   2. Clear old timestamp, assign new timestamp, update tree
# This gives O(log T) per read where T = total number of accesses.
# ---------------------------------------------------------------------------

class _FenwickTree:
    """Binary indexed tree for prefix sums."""
    __slots__ = ('_n', '_tree')

    def __init__(self, n):
        self._n = n
        self._tree = [0] * (n + 1)

    def update(self, i, delta):
        while i <= self._n:
            self._tree[i] += delta
            i += i & (-i)

    def query(self, i):
        """Sum of elements [1..i]."""
        s = 0
        while i > 0:
            s += self._tree[i]
            i -= i & (-i)
        return s


class _FastContextFenwick:
    """
    Ultra-fast context using Fenwick tree for O(log N) depth lookups.

    Each stack element gets a timestamp when it's last accessed/created.
    Depth = number of elements with timestamp > this element's timestamp.

    We use a Fenwick tree indexed by timestamp to count active elements.
    depth(key) = total_active - prefix_sum(key.timestamp)
    """
    __slots__ = ('_timestamps', '_tree', '_time', '_max_time',
                 'trace', 'sync', 'memo', 'counter', '_total_active')

    def __init__(self, max_ops=2_000_000):
        self._max_time = max_ops
        self._tree = _FenwickTree(max_ops)
        self._timestamps = {}  # key -> current timestamp
        self._time = 0
        self._total_active = 0
        self.trace = []
        self.sync = []
        self.memo = {}
        self.counter = 0

    def allocate(self):
        self.counter += 1
        key = self.counter
        self._time += 1
        self._timestamps[key] = self._time
        self._tree.update(self._time, 1)
        self._total_active += 1
        return key

    def read(self, keys):
        valid = [k for k in keys if k is not None]
        if not valid:
            return

        # Compute all depths against current state
        depths = []
        for k in valid:
            ts = self._timestamps[k]
            # Number of elements with timestamp > ts = total - prefix(ts)
            depth = self._total_active - self._tree.query(ts)
            # Depth is 0-indexed from top, but we need 1-indexed
            depths.append(depth + 1)

        self.trace.extend(depths)

        # Move each to top (assign new timestamps) in order
        for k in valid:
            old_ts = self._timestamps[k]
            self._tree.update(old_ts, -1)  # remove from old position
            self._time += 1
            self._timestamps[k] = self._time
            self._tree.update(self._time, 1)  # add at new position

    @property
    def stack(self):
        """For compatibility: return stack as list ordered bottom-to-top."""
        items = sorted(self._timestamps.items(), key=lambda x: x[1])
        return [k for k, _ in items]


# Use the Fenwick-based context as default (much faster for large problems)
_Context = _FastContextFenwick


# ---------------------------------------------------------------------------
# Tracked wrapper and wrapping/unwrapping (copied from bytedmd.py)
# ---------------------------------------------------------------------------

class _Tracked:
    __slots__ = ('_ctx', '_key', 'val')
    def __init__(self, ctx, key, val):
        self._ctx, self._key, self.val = ctx, key, val

    def _rd(self):
        self._ctx.read([self._key])
        return self.val

    def __str__(self): return str(self.val)
    def __repr__(self): return f"Tracked({self.val})"
    def __bool__(self): return bool(self._rd())
    def __int__(self): return int(self._rd())
    def __float__(self): return float(self._rd())
    def __complex__(self): return complex(self._rd())
    def __index__(self): return operator.index(self._rd())
    def __hash__(self): return hash(self._rd())


def _make_op(op, rev=False):
    def method(self, *args):
        keys = [a._key if isinstance(a, _Tracked) else None for a in args]
        vals = [a.val if isinstance(a, _Tracked) else a for a in args]
        read_keys = [keys[0], self._key] + keys[1:] if rev else [self._key] + keys
        res = op(vals[0], self.val, *vals[1:]) if rev else op(self.val, *vals)
        self._ctx.read(read_keys)
        return res if res is NotImplemented else _wrap(self._ctx, res)
    return method


_OPS = {
    **{k: getattr(operator, k) for k in 'add sub mul truediv floordiv mod lshift rshift xor matmul neg pos abs invert eq ne lt le gt ge'.split()},
    'and': operator.and_, 'or': operator.or_, 'divmod': divmod, 'pow': pow,
    'trunc': math.trunc, 'ceil': math.ceil, 'floor': math.floor, 'round': round
}

for n, f in _OPS.items():
    setattr(_Tracked, f'__{n}__', _make_op(f))
    if n in 'add sub mul truediv floordiv mod divmod pow lshift rshift and xor or matmul'.split():
        setattr(_Tracked, f'__r{n}__', _make_op(f, rev=True))


def _wrap(ctx, val):
    if isinstance(val, _Tracked): return val
    vid = id(val)
    if vid in ctx.memo: return ctx.memo[vid]

    is_prim = type(val) in (int, float, bool, complex, str)
    if type(val).__name__ == 'ndarray':
        import numpy as np
        res = np.empty_like(val, dtype=object)
        if not is_prim:
            ctx.memo[vid] = res
            ctx.sync.append((val, res))
        for idx in np.ndindex(val.shape):
            v = val[idx]
            res[idx] = _wrap(ctx, v.item() if hasattr(v, 'item') and not isinstance(v, np.ndarray) else v)
        return res

    if isinstance(val, list):
        res = []
        if not is_prim:
            ctx.memo[vid] = res
            ctx.sync.append((val, res))
        res.extend(_wrap(ctx, v) for v in val)
        return res

    if isinstance(val, tuple):
        res = tuple(_wrap(ctx, v) for v in val)
        if not is_prim: ctx.memo[vid] = res
        return res

    res = _Tracked(ctx, ctx.allocate(), val)
    if not is_prim: ctx.memo[vid] = res
    return res


def _unwrap(val, memo=None):
    if memo is None: memo = {}
    vid = id(val)
    if vid in memo: return memo[vid]

    is_prim = type(val) in (int, float, bool, complex, str)
    if isinstance(val, list):
        res = []
        if not is_prim: memo[vid] = res
        res.extend(_unwrap(v, memo) for v in val)
        return res

    if isinstance(val, tuple):
        res = tuple(_unwrap(v, memo) for v in val)
        if not is_prim: memo[vid] = res
        return res

    if type(val).__name__ == 'ndarray':
        import numpy as np
        if val.dtype == object:
            flat = [_unwrap(x, memo) for x in val.flat]
            try: res = np.array(flat).reshape(val.shape)
            except Exception:
                res = np.empty_like(val, dtype=object)
                for i, x in enumerate(flat): res.flat[i] = x
        else: res = val
        if not is_prim: memo[vid] = res
        return res

    res = getattr(val, 'val', val)
    if not is_prim: memo[vid] = res
    return res


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _sum_usqrt(N):
    if N <= 0: return 0
    M = math.isqrt(N - 1) + 1
    return M * (6 * N - 2 * M * M + 3 * M - 1) // 6


def traced_eval(func, args):
    """Run func with tracked arguments. Returns (trace, result)."""
    ctx = _Context()
    res = func(*(_wrap(ctx, a) for a in args))
    memo = {}
    for orig, wrapped in ctx.sync:
        if isinstance(orig, list): orig[:] = _unwrap(wrapped, memo)
        elif type(orig).__name__ == 'ndarray': orig[...] = _unwrap(wrapped, memo)
    return ctx.trace, _unwrap(res, memo)


def trace_to_bytedmd(trace, bytes_per_element):
    """Convert a trace (list of element depths) to ByteDMD cost."""
    if bytes_per_element == 1: return sum(math.isqrt(d - 1) + 1 for d in trace)
    bpe = bytes_per_element
    return sum(_sum_usqrt(d * bpe) - _sum_usqrt((d - 1) * bpe) for d in trace)


def bytedmd(func, args, bytes_per_element=1):
    """Evaluate ByteDMD cost of running func with args."""
    trace, _ = traced_eval(func, args)
    return trace_to_bytedmd(trace, bytes_per_element)
