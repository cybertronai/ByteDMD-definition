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
        self.memory = {}  # var_name -> size in bytes
        self.accesses = []
        self._counter = 0

    def register(self, name, size):
        """Allocate or move a variable block to the top (MRU) of the stack."""
        self.memory.pop(name, None)
        self.memory[name] = size

    def read_and_record(self, operands):
        """Record reads of variables and update the LRU stack."""
        # Deduplicate while preserving read order
        unique_ops = list(dict.fromkeys(operands))

        # Compute all byte distances at the current state BEFORE updating the stack
        for op in unique_ops:
            if op not in self.memory:
                continue
            depth = 0
            size = self.memory[op]
            # Iterate from MRU to LRU
            for k in reversed(list(self.memory)):
                if k == op:
                    self.accesses.extend(
                        (op, i, depth + size - i) for i in range(size)
                    )
                    break
                depth += self.memory[k]

        # Update LRU (move accessed variables to the top in the order they were read)
        for op in unique_ops:
            if op in self.memory:
                self.register(op, self.memory[op])

    def push_result(self, size):
        """Allocate space for an intermediate operation result."""
        name = f"_r{self._counter}"
        self._counter += 1
        self.register(name, size)
        return name


class _TrackedValue:
    """A scalar value that records read operations on a shared LRU stack."""

    def __init__(self, ctx, name, value, nbytes=1):
        self._ctx = ctx
        self.name = name
        self.value = value
        self.nbytes = nbytes

    def _do_op(self, other, op_func, is_compare=False, reverse=False):
        if isinstance(other, _TrackedValue):
            other_name, other_val, other_nb = other.name, other.value, other.nbytes
        else:
            other_name, other_val, other_nb = None, other, 1

        # Determine operand read order
        names = [other_name, self.name] if reverse else [self.name, other_name]
        self._ctx.read_and_record([n for n in names if n is not None])

        res_size = 1 if is_compare else max(self.nbytes, other_nb)
        res_name = self._ctx.push_result(res_size)
        
        val1, val2 = (other_val, self.value) if reverse else (self.value, other_val)
        return _TrackedValue(self._ctx, res_name, op_func(val1, val2), res_size)

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
        self._ctx.read_and_record([self.name])
        rname = self._ctx.push_result(self.nbytes)
        return _TrackedValue(self._ctx, rname, -self.value, self.nbytes)


class _TrackedArray:
    """Array whose element accesses are tracked on a shared LRU stack."""

    def __init__(self, ctx, name, values, elem_nbytes):
        self._ctx = ctx
        self._name = name
        self._values = values
        self._elem_nbytes = elem_nbytes

    def __getitem__(self, idx):
        val = self._values[idx]
        name = f"{self._name}_{idx}"
        if _is_array(val):
            return _TrackedArray(self._ctx, name, val, self._elem_nbytes)
        return _TrackedValue(self._ctx, name, val, self._elem_nbytes)

    def __len__(self):
        return len(self._values)


def _is_array(val):
    return hasattr(val, '__len__') and hasattr(val, '__getitem__') and hasattr(val, 'dtype')


def _register_elements(ctx, name, val, elem_size):
    """Recursively register array elements on the LRU stack."""
    if getattr(val, 'ndim', 0) > 1:
        for j, sub_val in enumerate(val):
            _register_elements(ctx, f"{name}_{j}", sub_val, elem_size)
    else:
        for j in range(len(val)):
            ctx.register(f"{name}_{j}", elem_size)


def _simulate(func, args):
    """Run func with traced arguments.

    Returns (accesses, result) where accesses is a list of
    (name, byte_idx, distance) tuples and result is the function's return value.
    """
    ctx = _TrackedContext()
    arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
    traced_args = []

    for name, val in zip(arg_names, args):
        if _is_array(val):
            elem_size = val.dtype.itemsize
            _register_elements(ctx, name, val, elem_size)
            traced_args.append(_TrackedArray(ctx, name, val, elem_size))
        else:
            size = getattr(val, 'nbytes', 1)
            ctx.register(name, size)
            traced_args.append(_TrackedValue(ctx, name, val, size))

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
    cost = sum(math.sqrt(d) for _, _, d in accesses)
    return cost, result


def measureDMDSquared(func, *args):
    """Like measureDMD but sums raw distances instead of sqrt(distance).

    Returns (cost, result).
    """
    accesses, result = _simulate(func, args)
    cost = sum(d for _, _, d in accesses)
    return cost, result


def byteReadTrace(func, *args):
    """Return (trace, result) where trace is a list of distances for each byte
    read, and result is the function's return value.
    """
    accesses, result = _simulate(func, args)
    return [d for _, _, d in accesses], result
