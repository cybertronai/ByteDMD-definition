"""
Implements ByteDMD cost model

bytedmd(add, (1, 2)) calls add(1, 2) and returns ByteDMD_cost
"""

import math
import sys


def usqrt(x):
    """Ceiling of square root."""
    return math.isqrt(x - 1) + 1


class _TrackedContext:
    """Shared LRU context used during traced execution."""

    def __init__(self):
        self.stack = []     # LRU stack (rightmost = MRU)
        self.trace = []  # list of access distances
        self._counter = 0

    def allocate(self):
        """Allocate a new entry at the top of the stack and return its key."""
        self._counter += 1
        key = self._counter
        self.stack.append(key)
        return key

    def read_all_then_move(self, keys):
        """Read all operands at current distances, then move all to top."""
        for key in keys:
            if key is None:
                continue
            depth = 0
            for k in reversed(self.stack):
                if k == key:
                    self.trace.append(depth + 1)
                    break
                depth += 1

        # Update LRU stack
        for key in keys:
            if key is not None:
                self.stack.remove(key)
                self.stack.append(key)


class _TrackedValue:
    """A scalar value that records read operations on a shared LRU stack."""

    def __init__(self, ctx, key, value):
        self._ctx = ctx
        self._key = key
        self.value = value

    def _do_op(self, other, op_name, reverse=False):
        if isinstance(other, _TrackedValue):
            other_key, other_val = other._key, other.value
        else:
            other_key, other_val = None, other

        keys = [other_key, self._key] if reverse else [self._key, other_key]
        self._ctx.read_all_then_move(keys)

        val1, val2 = (other_val, self.value) if reverse else (self.value, other_val)
        res_val = getattr(val1, op_name)(val2)

        if res_val is NotImplemented:
            return NotImplemented
        if isinstance(res_val, tuple):
            return tuple(_TrackedValue(self._ctx, self._ctx.allocate(), x) for x in res_val)
        return _TrackedValue(self._ctx, self._ctx.allocate(), res_val)

    def _do_unary(self, op_name):
        self._ctx.read_all_then_move([self._key])
        res_val = getattr(self.value, op_name)()
        return _TrackedValue(self._ctx, self._ctx.allocate(), res_val)

    def _do_cmp(self, other, op_name):
        if isinstance(other, _TrackedValue):
            other_key, other_val = other._key, other.value
        else:
            other_key, other_val = None, other

        keys = [self._key, other_key]
        self._ctx.read_all_then_move(keys)

        return getattr(self.value, op_name)(other_val)


# Binary arithmetic & bitwise operations
_binops = 'add sub mul truediv floordiv mod divmod pow lshift rshift and xor or matmul'
for _op in _binops.split():
    def _make_fwd(name):
        def method(self, o): return self._do_op(o, name)
        return method
    def _make_rev(name):
        def method(self, o): return self._do_op(o, name, reverse=True)
        return method
    setattr(_TrackedValue, f'__{_op}__', _make_fwd(f'__{_op}__'))
    setattr(_TrackedValue, f'__r{_op}__', _make_rev(f'__r{_op}__'))

# Unary operations
for _op in 'neg pos abs invert'.split():
    def _make_unary(name):
        def method(self): return self._do_unary(name)
        return method
    setattr(_TrackedValue, f'__{_op}__', _make_unary(f'__{_op}__'))

# Comparisons (returned unwrapped for correct control flow)
for _op in 'eq ne lt le gt ge'.split():
    def _make_cmp(name):
        def method(self, o): return self._do_cmp(o, name)
        return method
    setattr(_TrackedValue, f'__{_op}__', _make_cmp(f'__{_op}__'))


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
    """Run func with traced arguments.
    Returns (trace, result) where trace is a list of distances.
    """
    ctx = _TrackedContext()
    traced_args = [_wrap(ctx, val) for val in args]

    ret = func(*traced_args)
    return ctx.trace, _unwrap(ret)


def bytedmd(func, args, bytes_per_element=1):
    trace, result = traced_eval(func, args)
    return sum(usqrt(int(d * bytes_per_element)) for d in trace)
