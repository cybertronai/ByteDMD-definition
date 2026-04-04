"""
Implements ByteDMD cost model

bytedmd(add, (1, 2)) calls add(1, 2) and returns ByteDMD_cost
"""

import math


def usqrt(x):
    """Ceiling of square root."""
    return math.isqrt(x - 1) + 1 if x > 0 else 0


class _TrackedContext:
    """Shared LRU context used during traced execution."""

    def __init__(self):
        self.stack = []     # LRU stack (rightmost = MRU)
        self.trace = []     # list of access distances
        self._counter = 0

    def allocate(self):
        """Allocate a new entry at the top of the stack and return its key."""
        self._counter += 1
        self.stack.append(self._counter)
        return self._counter

    def read_all_then_move(self, keys):
        """Read all operands at current distances, then sequentially move all to top."""
        valid_keys = [k for k in keys if k is not None]
        
        # Calculate element depth relative to top (MRU)
        self.trace.extend(len(self.stack) - self.stack.index(k) for k in valid_keys)
        
        # Update LRU stack
        for k in valid_keys:
            self.stack.remove(k)
            self.stack.append(k)


class _TrackedValue:
    """A scalar value that records read operations on a shared LRU stack."""

    def __init__(self, ctx, key, value):
        self._ctx = ctx
        self._key = key
        self.value = value


def _make_method(op, is_cmp=False, is_rev=False):
    """Generic factory evaluating bin/unary/cmp operations with read penalties."""
    def method(self, *args):
        if not args:  # Unary
            self._ctx.read_all_then_move([self._key])
            res = getattr(self.value, op)()
        else:  # Binary or Comparison
            other = args[0]
            okey, oval = (other._key, other.value) if isinstance(other, _TrackedValue) else (None, other)
            self._ctx.read_all_then_move([okey, self._key] if is_rev else [self._key, okey])
            
            # Delegate reverse handling naturally to wrapped primitive (`int.__rsub__`, etc.)
            res = getattr(self.value, op)(oval)
            
        if is_cmp or res is NotImplemented:
            return res
        if isinstance(res, tuple):
            return tuple(_TrackedValue(self._ctx, self._ctx.allocate(), x) for x in res)
        return _TrackedValue(self._ctx, self._ctx.allocate(), res)
    return method


# Wire up Python magic methods dynamically
for _op in 'add sub mul truediv floordiv mod divmod pow lshift rshift and xor or matmul'.split():
    setattr(_TrackedValue, f'__{_op}__', _make_method(f'__{_op}__'))
    setattr(_TrackedValue, f'__r{_op}__', _make_method(f'__r{_op}__', is_rev=True))

for _op in 'neg pos abs invert'.split():
    setattr(_TrackedValue, f'__{_op}__', _make_method(f'__{_op}__'))

for _op in 'eq ne lt le gt ge'.split():
    setattr(_TrackedValue, f'__{_op}__', _make_method(f'__{_op}__', is_cmp=True))


def _wrap(ctx, val):
    """Recursively convert ndarrays/lists of items into nested lists of tracked scalars."""
    if getattr(val, 'ndim', 0) > 0 or isinstance(val, (list, tuple)):
        return [_wrap(ctx, v) for v in val]
    return _TrackedValue(ctx, ctx.allocate(), val)


def _unwrap(val):
    """Recursively safely unwrap returned _TrackedValue scalars back to python primitives."""
    if isinstance(val, list):
        return [_unwrap(v) for v in val]
    if isinstance(val, tuple):
        return tuple(_unwrap(v) for v in val)
    if isinstance(val, _TrackedValue):
        return val.value
    return val


def traced_eval(func, args):
    """Run func with traced arguments. Returns (trace, result)."""
    ctx = _TrackedContext()
    return ctx.trace, _unwrap(func(*[_wrap(ctx, val) for val in args]))


def bytedmd(func, args, bytes_per_element=1):
    """Evaluate ByteDMD cost of running func with args."""
    trace, _ = traced_eval(func, args)
    return sum(usqrt(int(d * bytes_per_element)) for d in trace)