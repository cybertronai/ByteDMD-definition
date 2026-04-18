"""
ByteDMD tracer with three pluggable memory management strategies.

Strategy 1: Unmanaged
    Free is a no-op. Temporaries pile up forever, pushing live data deeper.
    This matches the current bytedmd.py behavior — abstract addresses are
    never recycled. Models a memory-leaking allocator.

Strategy 2: Tombstone (Traditional GC / Hole-fill)
    Freed slots become tombstones. New allocations reuse the most recently
    freed tombstone in place (LIFO of holes). Old data BELOW a freed slot
    still sees the same depth (depth includes the dead slot, because in
    real hardware the cache line still occupies a physical address).
    Models a standard malloc/free or a stop-the-world GC that doesn't
    compact.

Strategy 3: Aggressive (Instant compaction)
    Freed slots are removed from the stack immediately. Items below slide
    UP toward MRU. Models an idealized compacting GC where dead bytes get
    instantly returned to faster cache.

All three strategies now treat function inputs specially: untouched
arguments live on a separate argument stack, their first read is priced
there, and only then are they promoted into the managed geometric stack.
Measurements also include a final full read of the returned value.

The cost model is element-level (bytes_per_element=1). Two cost variants
are computed for every trace:
  - DISCRETE: cost = sum(ceil(sqrt(depth))). The original bytedmd.py form.
  - CONTINUOUS: cost = sum( (2/3) * (d^1.5 - (d-1)^1.5) ) per read,
    which is the exact integral form ∫_{d-1}^{d} sqrt(x) dx for one
    element. This matches the "Continuous ByteDMD model" used in the
    closed-form analytical formulas (Gemini's analysis), where reading
    a block of volume V starting at depth D costs
        ∫_D^{D+V} sqrt(x) dx  =  (2/3) [(D+V)^1.5 − D^1.5].
    Summing the per-element form over a block of V elements located at
    depths D+1, D+2, ..., D+V gives exactly that integral, so the
    per-read continuous cost is internally consistent with the analytical
    block model.
"""

import math


def usqrt(d):
    """ceil(sqrt(d)) for d > 0 (discrete ByteDMD per-element cost)."""
    return math.isqrt(d - 1) + 1 if d > 0 else 0


def trace_to_cost_discrete(trace):
    """Total discrete ByteDMD cost = sum(ceil(sqrt(d))) over all reads."""
    return sum(usqrt(d) for d in trace)


def trace_to_cost_continuous(trace):
    """Total continuous ByteDMD cost = sum( ∫_{d-1}^{d} sqrt(x) dx ).

    Each per-element read at integer depth d contributes the exact
    integral (2/3) (d^1.5 - (d-1)^1.5). Summing across consecutive
    contiguous reads is equivalent to the block integral
    (2/3) ((D+V)^1.5 - D^1.5).
    """
    return sum((2.0 / 3.0) * (d ** 1.5 - (d - 1) ** 1.5) for d in trace)


# Default cost function: continuous, since this experiment is comparing
# against the continuous closed-form formulas. The discrete form is also
# computed and reported for completeness.
def trace_to_cost(trace, mode='continuous'):
    if mode == 'discrete':
        return trace_to_cost_discrete(trace)
    elif mode == 'continuous':
        return trace_to_cost_continuous(trace)
    raise ValueError(f"Unknown mode: {mode}")


class Context:
    """LRU stack with pluggable free() semantics.

    Implementation: a doubly-linked list of nodes, each holding a key
    and alive flag. We keep a dict `slot[key] -> node` for O(1) lookup
    and a `top_index` count to compute depth in O(distance from top).
    For correctness we use a simpler approach: a Python list with a
    parallel `key_to_index` dict that we keep in sync after each
    mutation. This makes index() O(1) but mutations (insert/delete)
    require updating the index dict for shifted entries.

    To keep mutations cheap, we represent the LRU stack as a Python
    list of (key, alive) but locate keys via a dict. We rebuild the
    dict-of-positions lazily.
    """

    def __init__(self, strategy='unmanaged'):
        assert strategy in ('unmanaged', 'tombstone', 'aggressive')
        self.strategy = strategy
        self.stack = []        # entries: positive int (alive key) or -1 (dead)
        self.pos = {}          # key -> current index in self.stack
        self.input_stack = []  # untouched input arguments live here until first read
        self.input_pos = {}    # key -> current index in self.input_stack
        self.trace = []
        self.counter = 0

    def _refresh_pos(self):
        """Recompute pos dict after a stack mutation that shifts indices."""
        self.pos = {k: i for i, k in enumerate(self.stack) if k != -1}

    def _refresh_input_pos(self):
        self.input_pos = {k: i for i, k in enumerate(self.input_stack)}

    def allocate(self, *, is_input=False):
        self.counter += 1
        key = self.counter
        if is_input:
            self.input_stack.append(key)
            self.input_pos[key] = len(self.input_stack) - 1
            return key
        if self.strategy == 'tombstone':
            # Scan from top down for the most recent tombstone.
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i] == -1:
                    self.stack[i] = key
                    self.pos[key] = i
                    return key
        self.stack.append(key)
        self.pos[key] = len(self.stack) - 1
        return key

    def read(self, key):
        input_idx = self.input_pos.get(key)
        if input_idx is not None:
            depth = len(self.input_stack) - input_idx
            self.trace.append(depth)
            del self.input_stack[input_idx]
            del self.input_pos[key]
            for j in range(input_idx, len(self.input_stack)):
                self.input_pos[self.input_stack[j]] = j
            self.stack.append(key)
            self.pos[key] = len(self.stack) - 1
            return depth
        idx = self.pos[key]
        depth = len(self.stack) - idx
        self.trace.append(depth)
        # Move key from idx to top. The entries between idx+1 and end
        # all shift down by 1.
        del self.stack[idx]
        self.stack.append(key)
        # Update pos for everything that shifted: entries previously at
        # indices > idx now sit at idx, idx+1, ...
        for j in range(idx, len(self.stack) - 1):
            k = self.stack[j]
            if k != -1:
                self.pos[k] = j
        self.pos[key] = len(self.stack) - 1
        return depth

    def free(self, key):
        input_idx = self.input_pos.get(key)
        if input_idx is not None:
            del self.input_stack[input_idx]
            del self.input_pos[key]
            for j in range(input_idx, len(self.input_stack)):
                self.input_pos[self.input_stack[j]] = j
            return
        if self.strategy == 'unmanaged':
            return
        idx = self.pos.get(key)
        if idx is None:
            return
        if self.strategy == 'tombstone':
            self.stack[idx] = -1
            del self.pos[key]
        elif self.strategy == 'aggressive':
            del self.stack[idx]
            del self.pos[key]
            for j in range(idx, len(self.stack)):
                k = self.stack[j]
                if k != -1:
                    self.pos[k] = j


class Tracked:
    """Element wrapper. Records reads via dunder methods, frees on __del__."""

    def __init__(self, ctx, key, val):
        self._ctx = ctx
        self._key = key
        self.val = val

    def __del__(self):
        # Hook into Python GC to simulate memory reclamation.
        # Wrapped in try/except because __del__ may run during interpreter
        # shutdown when self._ctx may be gone.
        try:
            self._ctx.free(self._key)
        except Exception:
            pass

    def _do_binop(self, other, op):
        if isinstance(other, Tracked):
            self._ctx.read(self._key)
            self._ctx.read(other._key)
            v = op(self.val, other.val)
        else:
            self._ctx.read(self._key)
            v = op(self.val, other)
        return Tracked(self._ctx, self._ctx.allocate(), v)

    def __add__(self, other): return self._do_binop(other, lambda a, b: a + b)
    def __sub__(self, other): return self._do_binop(other, lambda a, b: a - b)
    def __mul__(self, other): return self._do_binop(other, lambda a, b: a * b)
    def __radd__(self, other): return self._do_binop(other, lambda a, b: b + a)
    def __rsub__(self, other): return self._do_binop(other, lambda a, b: b - a)
    def __rmul__(self, other): return self._do_binop(other, lambda a, b: b * a)


def wrap_matrix(ctx, mat):
    """Convert a list-of-lists matrix into Tracked objects."""
    return [[Tracked(ctx, ctx.allocate(is_input=True), v) for v in row] for row in mat]


def _read_return_value(value):
    if isinstance(value, Tracked):
        value._ctx.read(value._key)
        return
    if isinstance(value, list):
        for item in value:
            _read_return_value(item)
        return
    if isinstance(value, tuple):
        for item in value:
            _read_return_value(item)
        return


def measure(matmul_fn, A, B, strategy, inplace=False):
    """Run matmul_fn under `strategy` and report cost+footprint.

    If inplace=False, the function signature is matmul_fn(A, B) -> C and
    we measure the entire computation including the construction of C.

    If inplace=True, the function signature is matmul_fn(A, B, C) where
    C is a pre-allocated zero matrix and the result accumulates into C.
    The measurement covers the in-place call.
    """
    ctx = Context(strategy=strategy)
    A_w = wrap_matrix(ctx, A)
    B_w = wrap_matrix(ctx, B)
    if inplace:
        C_init = [[0] * len(A) for _ in range(len(A))]
        C_w = wrap_matrix(ctx, C_init)

    peak = [len(ctx.stack)]
    orig_alloc = ctx.allocate
    def tracking_alloc(*, is_input=False):
        k = orig_alloc(is_input=is_input)
        peak[0] = max(peak[0], len(ctx.stack))
        return k
    ctx.allocate = tracking_alloc

    if inplace:
        matmul_fn(A_w, B_w, C_w)
        _read_return_value(C_w)
        del C_w
    else:
        result = matmul_fn(A_w, B_w)
        _read_return_value(result)
        del result
    del A_w
    del B_w

    return {
        'cost_discrete': trace_to_cost_discrete(ctx.trace),
        'cost_continuous': trace_to_cost_continuous(ctx.trace),
        'n_reads': len(ctx.trace),
        'peak_stack': peak[0],
    }
