import math
import operator

"""
Examples and checks for the ByteDMD metric of complexity described in README.md

Each of the functions below takes a function func, arguments, calls func and returns a tuple [<x>, result]

whereas <x> is the following:
measureDMD -- DMD cost, ie, sum of square roots of data access distances, byte-level boundaries
measureDMDSquared -- squared DMD cost, ie, sum of raw data access distances (each distance is the square of the corresponding sqrt(d) DMD term), always integer-valued
byteReadTrace - list of byte access distances, in order of read

"""

class _TrackedContext:
    """Shared LRU context used during traced execution."""

    def __init__(self):
        # Python 3.7+ dict maintains insertion order (LRU stack, rightmost = MRU)
        self.memory = {}  # key (int) -> size in bytes
        self.accesses = []  # list of distances
        self._counter = 0

    def register(self, key, size):
        """Allocate or move a block to the top (MRU) of the stack."""
        self.memory.pop(key, None)
        self.memory[key] = size

    def read(self, key):
        """Record byte distances for a variable at its current stack position."""
        if key not in self.memory:
            return
        depth = 0
        size = self.memory[key]
        for k in reversed(list(self.memory)):
            if k == key:
                self.accesses.extend(depth + size - i for i in range(size))
                break
            depth += self.memory[k]

    def move_to_top(self, key):
        """Move a variable to the top (MRU) of the stack."""
        if key in self.memory:
            self.register(key, self.memory[key])

    def read_all_then_move(self, keys):
        """Read all operands at current distances, then move all to top."""
        for k in keys:
            self.read(k)
        for k in keys:
            self.move_to_top(k)

    def next_key(self):
        """Return a unique key."""
        self._counter += 1
        return self._counter

    def push_result(self, size):
        """Allocate a new block at the top of the stack and return its key."""
        key = self.next_key()
        self.register(key, size)
        return key


class _TrackedValue:
    """A scalar value that records read operations on a shared LRU stack."""

    def __init__(self, ctx, key, value, nbytes=1):
        self._ctx = ctx
        self._key = key
        self.value = value
        self.nbytes = nbytes

    def _do_op(self, other, op_func, is_compare=False, reverse=False):
        if isinstance(other, _TrackedValue):
            other_key, other_val, other_nb = other._key, other.value, other.nbytes
        else:
            other_key, other_val, other_nb = None, other, 1

        # Read operands: all distances computed before any LRU update
        keys = [other_key, self._key] if reverse else [self._key, other_key]
        self._ctx.read_all_then_move([k for k in keys if k is not None])

        res_size = 1 if is_compare else max(self.nbytes, other_nb)
        res_key = self._ctx.push_result(res_size)

        val1, val2 = (other_val, self.value) if reverse else (self.value, other_val)
        return _TrackedValue(self._ctx, res_key, op_func(val1, val2), res_size)

    def __add__(self, o): return self._do_op(o, operator.add)
    def __radd__(self, o): return self._do_op(o, operator.add, reverse=True)
    def __sub__(self, o): return self._do_op(o, operator.sub)
    def __rsub__(self, o): return self._do_op(o, operator.sub, reverse=True)
    def __mul__(self, o): return self._do_op(o, operator.mul)
    def __rmul__(self, o): return self._do_op(o, operator.mul, reverse=True)
    def __truediv__(self, o): return self._do_op(o, operator.truediv)
    def __rtruediv__(self, o): return self._do_op(o, operator.truediv, reverse=True)

    def __gt__(self, o): return self._do_op(o, operator.gt, is_compare=True)
    def __lt__(self, o): return self._do_op(o, operator.lt, is_compare=True)
    def __ge__(self, o): return self._do_op(o, operator.ge, is_compare=True)
    def __le__(self, o): return self._do_op(o, operator.le, is_compare=True)

    def __neg__(self):
        self._ctx.read_all_then_move([self._key])
        res_key = self._ctx.push_result(self.nbytes)
        return _TrackedValue(self._ctx, res_key, -self.value, self.nbytes)


class _TrackedArray:
    """Array whose element accesses are tracked on a shared LRU stack."""

    def __init__(self, ctx, keys, values, elem_nbytes):
        self._ctx = ctx
        self._keys = keys
        self._values = values
        self._elem_nbytes = elem_nbytes

    def __getitem__(self, idx):
        val = self._values[idx]
        if _is_array(val):
            return _TrackedArray(self._ctx, self._keys[idx], val, self._elem_nbytes)
        return _TrackedValue(self._ctx, self._keys[idx], val, self._elem_nbytes)

    def __len__(self):
        return len(self._values)


def _is_array(val):
    return hasattr(val, '__len__') and hasattr(val, '__getitem__') and hasattr(val, 'dtype')


def _register_elements(ctx, val, elem_size):
    """Recursively register array elements on the LRU stack. Returns keys structure."""
    if getattr(val, 'ndim', 0) > 1:
        return [_register_elements(ctx, val[j], elem_size) for j in range(len(val))]
    else:
        keys = []
        for j in range(len(val)):
            key = ctx.next_key()
            ctx.register(key, elem_size)
            keys.append(key)
        return keys


def _simulate(func, args):
    """Run func with traced arguments.

    Returns (accesses, result) where accesses is a list of distances
    and result is the function's return value.
    """
    ctx = _TrackedContext()
    arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
    traced_args = []

    for name, val in zip(arg_names, args):
        if _is_array(val):
            elem_size = val.dtype.itemsize
            keys = _register_elements(ctx, val, elem_size)
            traced_args.append(_TrackedArray(ctx, keys, val, elem_size))
        else:
            size = getattr(val, 'nbytes', 1)
            key = ctx.next_key()
            ctx.register(key, size)
            traced_args.append(_TrackedValue(ctx, key, val, size))

    ret = func(*traced_args)
    if isinstance(ret, _TrackedValue):
        ret = ret.value
    return ctx.accesses, ret


def measureDMD(func, *args):
    """Measure ByteDMD cost and call func.

    Returns (cost, result) where cost is the sum of sqrt(distance) for each
    byte read, and result is the function's return value.
    """
    accesses, result = _simulate(func, args)
    cost = sum(math.sqrt(d) for d in accesses)
    return cost, result


def measureDMDSquared(func, *args):
    """Squared variant of measureDMD: sums raw distances d instead of sqrt(d).

    Since DMD costs are sqrt(d), this is equivalent to summing the squares of
    individual DMD costs: d = (sqrt(d))^2. Always integer-valued for
    integer-byte-sized types.

    Returns (cost, result).
    """
    accesses, result = _simulate(func, args)
    cost = sum(accesses)
    return cost, result


def byteReadTrace(func, *args):
    """Return (trace, result) where trace is a list of distances for each byte
    read, and result is the function's return value.
    """
    accesses, result = _simulate(func, args)
    return accesses, result
