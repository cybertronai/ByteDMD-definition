"""
Implements ByteDMD cost model

measure_dmd(add, (1, 2)) calls add(1, 2) and returns (ByteDMD_cost, 3)
"""

import math
import operator


def usqrt(x):
    """Ceiling of square root."""
    return math.isqrt(x - 1) + 1


class _TrackedContext:
    """Shared LRU context used during traced execution."""

    def __init__(self):
        self.stack = []     # LRU stack (rightmost = MRU)
        self.sizes = {}     # key -> size in bytes
        self.trace = []  # list of byte-access distances
        self._counter = 0

    def allocate(self, size):
        """Allocate a new block at the top of the stack and return its key."""
        self._counter += 1
        key = self._counter
        self.sizes[key] = size
        self.stack.append(key)
        return key

    def read_all_then_move(self, keys):
        """Read all operands at current distances, then move all to top."""
        for key in keys:
            if key not in self.sizes:
                continue
            depth = 0
            size = self.sizes[key]
            for k in reversed(self.stack):
                if k == key:
                    self.trace.extend(range(depth + size, depth, -1))
                    break
                depth += self.sizes[k]
                
        # Update LRU stack
        for key in keys:
            if key in self.sizes:
                self.stack.remove(key)
                self.stack.append(key)


class _TrackedValue:
    """A scalar value that records read operations on a shared LRU stack."""

    def __init__(self, ctx, key, value, nbytes=1):
        self._ctx = ctx
        self._key = key
        self.value = value
        self.nbytes = nbytes

    def _do_op(self, other, op_func, reverse=False):
        if isinstance(other, _TrackedValue):
            other_key, other_val = other._key, other.value
        else:
            other_key, other_val = None, other

        # Read operands: all distances computed before any LRU update
        keys = [other_key, self._key] if reverse else [self._key, other_key]
        self._ctx.read_all_then_move(keys)

        val1, val2 = (other_val, self.value) if reverse else (self.value, other_val)
        res_val = op_func(val1, val2)
        res_size = getattr(res_val, 'nbytes', 1)
        
        return _TrackedValue(
            self._ctx, 
            self._ctx.allocate(res_size), 
            res_val, 
            res_size
        )

    def __add__(self, o): return self._do_op(o, operator.add)
    def __radd__(self, o): return self._do_op(o, operator.add, reverse=True)
    def __sub__(self, o): return self._do_op(o, operator.sub)
    def __rsub__(self, o): return self._do_op(o, operator.sub, reverse=True)
    def __mul__(self, o): return self._do_op(o, operator.mul)
    def __rmul__(self, o): return self._do_op(o, operator.mul, reverse=True)
    def __truediv__(self, o): return self._do_op(o, operator.truediv)
    def __rtruediv__(self, o): return self._do_op(o, operator.truediv, reverse=True)
    
    def __gt__(self, o): return self._do_op(o, operator.gt)
    def __lt__(self, o): return self._do_op(o, operator.lt)
    def __ge__(self, o): return self._do_op(o, operator.ge)
    def __le__(self, o): return self._do_op(o, operator.le)

    def __neg__(self):
        self._ctx.read_all_then_move([self._key])
        res_val = -self.value
        res_size = getattr(res_val, 'nbytes', 1)
        return _TrackedValue(
            self._ctx, 
            self._ctx.allocate(res_size), 
            res_val, 
            res_size
        )


def _wrap(ctx, val):
    """Recursively convert ndarrays/lists of items into nested lists of tracked scalars."""
    if getattr(val, 'ndim', 0) > 0 or isinstance(val, (list, tuple)):
        return [_wrap(ctx, v) for v in val]
    size = getattr(val, 'nbytes', 1)
    return _TrackedValue(ctx, ctx.allocate(size), val, size)


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


def measure_dmd(func, args):
    trace, result = traced_eval(func, args)
    return sum(usqrt(d) for d in trace), result



