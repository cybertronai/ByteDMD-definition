"""
ByteDMD tracer — proxy-based, with liveness-driven tombstones.

Wraps function arguments in `_Tracked` proxy objects that intercept
arithmetic, comparison, and indexing via Python dunder methods. Each
intercepted operation records a read against an LRU stack and the depth
is later converted to a ByteDMD cost via ceil(sqrt(depth)).

The tracer performs **liveness analysis**: it first executes the function to
record the full operation sequence, then replays it with perfect knowledge
of each value's last read. After a value's final read, it is replaced by a
tombstone at the top of the stack. Tombstones preserve the physical cache
footprint (dead cache lines still occupy space until overwritten). Each new
allocation recycles exactly one tombstone (the bottom-most).

API:
    traced_eval(func, args)                   -> (trace, result)
    bytedmd(func, args, bytes_per_element=1)  -> int
    trace_to_bytedmd(trace, bytes_per_element) -> int
    inspect_ir(func, args)                    -> list of IR events
    format_ir(ir)                             -> str  (pretty-printed IR)
"""

import math
import operator


class _Context:
    __slots__ = ('stack', 'trace', 'sync', 'memo', 'counter', 'ir')
    def __init__(self):
        self.stack, self.trace, self.sync, self.memo, self.counter, self.ir = [], [], [], {}, 0, []

    def allocate(self):
        self.counter += 1
        self.stack.append(self.counter)
        return self.counter

    def read(self, keys):
        valid = [k for k in keys if k is not None]
        if not valid: return []
        depths = [len(self.stack) - self.stack.index(k) for k in valid]
        self.trace.extend(depths)
        for k, d in zip(valid, depths):
            self.stack.remove(k)
            self.stack.append(k)
            self.ir.append(('READ', k, d))
        return depths


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
    name = op.__name__
    def method(self, *args):
        keys = [a._key if isinstance(a, _Tracked) else None for a in args]
        vals = [a.val if isinstance(a, _Tracked) else a for a in args]

        read_keys = [keys[0], self._key] + keys[1:] if rev else [self._key] + keys
        res = op(vals[0], self.val, *vals[1:]) if rev else op(self.val, *vals)
        depths = self._ctx.read(read_keys)
        valid_keys = [k for k in read_keys if k is not None]

        if res is NotImplemented:
            self._ctx.ir.append(('OP', name, valid_keys, depths, None))
            return res
        wrapped = _wrap(self._ctx, res)
        out_key = wrapped._key if isinstance(wrapped, _Tracked) else None
        # _wrap just emitted ('STORE', out_key); reorder so OP precedes STORE.
        if out_key is not None and self._ctx.ir and self._ctx.ir[-1] == ('STORE', out_key):
            self._ctx.ir.pop()
            self._ctx.ir.append(('OP', name, valid_keys, depths, out_key))
            self._ctx.ir.append(('STORE', out_key))
        else:
            self._ctx.ir.append(('OP', name, valid_keys, depths, out_key))
        return wrapped
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
    ctx.ir.append(('STORE', res._key))
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


def _sum_usqrt(N):
    if N <= 0: return 0
    M = math.isqrt(N - 1) + 1
    return M * (6 * N - 2 * M * M + 3 * M - 1) // 6


def _replay_with_liveness(raw_ir):
    """Replay a pass-1 IR with liveness-based tombstones.

    Pass 1 records the operation sequence (STOREs, READs, OPs) without any
    eviction.  This function scans the full sequence to determine the last
    read of every tracked key, then replays it on a fresh stack, inserting
    a tombstone (at the top) immediately after each value's final read.
    New allocations recycle one tombstone (the bottom-most).

    Returns (trace, ir) — the depth trace and a new IR with correct depths
    and DROP events.
    """
    # Find last read index of each key.
    last_read = {}
    read_idx = 0
    for ev in raw_ir:
        if ev[0] == 'READ':
            last_read[ev[1]] = read_idx
            read_idx += 1

    stack = []
    trace = []
    ir = []
    read_idx = 0

    for ev in raw_ir:
        tag = ev[0]

        if tag == 'STORE':
            key = ev[1]
            try:
                stack.remove(None)          # recycle one tombstone
            except ValueError:
                pass
            stack.append(key)
            ir.append(('STORE', key))

        elif tag == 'READ':
            key = ev[1]
            depth = len(stack) - stack.index(key)
            trace.append(depth)
            stack.remove(key)
            stack.append(key)
            ir.append(('READ', key, depth))
            if last_read[key] == read_idx:  # value is dead after this read
                stack.remove(key)
                stack.append(None)          # tombstone at top
                ir.append(('DROP', key))
            read_idx += 1

        elif tag == 'OP':
            _, name, keys, _old_depths, out_key = ev
            n = len(keys)
            new_depths = trace[-n:] if n > 0 else []
            ir.append(('OP', name, keys, list(new_depths), out_key))

    return trace, ir


def traced_eval(func, args):
    """Run func with tracked arguments. Returns (trace, result)."""
    ctx = _Context()
    res = func(*(_wrap(ctx, a) for a in args))
    memo = {}
    for orig, wrapped in ctx.sync:
        if isinstance(orig, list): orig[:] = _unwrap(wrapped, memo)
        elif type(orig).__name__ == 'ndarray': orig[...] = _unwrap(wrapped, memo)
    trace, _ = _replay_with_liveness(ctx.ir)
    return trace, _unwrap(res, memo)


def trace_to_bytedmd(trace, bytes_per_element):
    """Convert a trace (list of element depths) to ByteDMD cost."""
    if bytes_per_element == 1: return sum(math.isqrt(d - 1) + 1 for d in trace)
    bpe = bytes_per_element
    return sum(_sum_usqrt(d * bpe) - _sum_usqrt((d - 1) * bpe) for d in trace)


def inspect_ir(func, args):
    """Run func and return its ByteDMD intermediate representation.

    The tracer first executes the function to record the operation sequence,
    then replays it with liveness analysis: each value is tombstoned after
    its final read.

    The IR is a list of events:

      ('STORE', k)                                — value vk allocated on top
      ('READ',  k, d)                            — read vk at depth d, LRU-bump
      ('OP',   name, [ki...], [di...], out_key)  — summary of the preceding reads
      ('DROP', k)                                — vk tombstoned (last read done)

    Format with format_ir() for a human-readable listing.
    """
    ctx = _Context()
    func(*(_wrap(ctx, a) for a in args))
    _, ir = _replay_with_liveness(ctx.ir)
    return ir


def format_ir(ir):
    """Pretty-print an IR returned by inspect_ir()."""
    out = []
    total = 0
    for ev in ir:
        if ev[0] == 'STORE':
            out.append(f"STORE v{ev[1]}")
        elif ev[0] == 'DROP':
            out.append(f"DROP  v{ev[1]}")
        elif ev[0] == 'READ':
            _, key, depth = ev
            cost = math.isqrt(depth - 1) + 1
            out.append(f"  READ v{key}@{depth}  cost={cost}")
        else:
            _, name, keys, depths, _ok = ev
            cost = sum(math.isqrt(d - 1) + 1 for d in depths)
            total += cost
            rd = ", ".join(f"v{k}@{d}" for k, d in zip(keys, depths))
            out.append(f"OP    {name}({rd})  cost={cost}")
    out.append(f"# total cost = {total}")
    return "\n".join(out)


def bytedmd(func, args, bytes_per_element=1):
    """Evaluate ByteDMD cost of running func with args."""
    trace, _ = traced_eval(func, args)
    return trace_to_bytedmd(trace, bytes_per_element)