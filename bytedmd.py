"""
ByteDMD tracer — fully associative with demand-paged initialization.

  1. Simultaneous pricing: All inputs are priced against the pre-instruction
     stack state before LRU bumping, guaranteeing commutativity.
  2. Demand-paged initialization: Arguments start in "DRAM". First access
     triggers a cold miss priced strictly outside the active L1 footprint.
  3. Natural LRU aging: Dead variables sink naturally to the bottom of the
     stack ("Infinite Graveyard").
"""

import math
import operator

class _Context:
    __slots__ = ('stack', 'trace', 'sync', 'memo', 'counter', 'ir')
    
    def __init__(self):
        self.stack, self.trace, self.sync, self.ir = [], [], [], []
        self.memo = {}
        self.counter = 0

    def allocate(self, deferred=False):
        """Allocate a tracking ID. If not deferred, push to L1 stack."""
        self.counter += 1
        if not deferred:
            self.stack.append(self.counter)
        return self.counter

    def read(self, keys):
        """Prices keys simultaneously against pre-instruction stack."""
        valid = [k for k in keys if k is not None]
        if not valid: return []

        # Fast, order-preserving deduplication
        unique = list(dict.fromkeys(valid))
        depths_map = {}
        cold_keys = []
        L = len(self.stack)

        # 1. Price simultaneously against the active physical universe
        for k in unique:
            try:
                depths_map[k] = L - self.stack.index(k)
            except ValueError:
                # Cold Miss: Paged from just outside the known universe
                cold_keys.append(k)
                depths_map[k] = L + len(cold_keys)

        # Bring cold misses into the cache
        self.stack.extend(cold_keys)

        # 2. Emit events and batch LRU bump
        self.trace.extend(depths_map[k] for k in valid)
        for k in valid:
            self.ir.append(('READ', k, depths_map[k]))

        for k in unique:
            self.stack.remove(k)
            self.stack.append(k)

        return [depths_map[k] for k in valid]


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
        keys = [getattr(a, '_key', None) for a in args]
        vals = [getattr(a, 'val', a) for a in args]

        read_keys = [keys[0], self._key] + keys[1:] if rev else [self._key] + keys
        res = op(vals[0], self.val, *vals[1:]) if rev else op(self.val, *vals)
        
        depths = self._ctx.read(read_keys)
        valid_keys = [k for k in read_keys if k is not None]

        if res is NotImplemented:
            self._ctx.ir.append(('OP', name, valid_keys, depths, None))
            return res

        # Record IR length to perfectly insert OP before any new STOREs (fixes tuple returns)
        ir_len_before = len(self._ctx.ir)
        wrapped = _wrap(self._ctx, res)
        out_key = getattr(wrapped, '_key', None)
        
        self._ctx.ir.insert(ir_len_before, ('OP', name, valid_keys, depths, out_key))
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


def _wrap(ctx, val, deferred=False):
    if isinstance(val, _Tracked): return val
    vid = id(val)
    if vid in ctx.memo: return ctx.memo[vid]

    typ = type(val)
    is_prim = typ in (int, float, bool, complex, str)
    
    if typ.__name__ == 'ndarray':
        import numpy as np
        res = np.empty_like(val, dtype=object)
        if not is_prim:
            ctx.memo[vid] = res
            ctx.sync.append((val, res))
        for idx in np.ndindex(val.shape):
            v = val[idx]
            res[idx] = _wrap(ctx, v.item() if hasattr(v, 'item') and not isinstance(v, np.ndarray) else v, deferred)
        return res

    if isinstance(val, (list, tuple)):
        res = typ(_wrap(ctx, v, deferred) for v in val)
        if not is_prim:
            ctx.memo[vid] = res
            if typ is list: ctx.sync.append((val, res))
        return res

    key = ctx.allocate(deferred)
    res = _Tracked(ctx, key, val)
    if not deferred: 
        ctx.ir.append(('STORE', key))
    if not is_prim: ctx.memo[vid] = res
    return res


def _unwrap(val, memo=None):
    if memo is None: memo = {}
    vid = id(val)
    if vid in memo: return memo[vid]

    typ = type(val)
    is_prim = typ in (int, float, bool, complex, str)
    
    if isinstance(val, (list, tuple)):
        res = typ(_unwrap(v, memo) for v in val)
        if not is_prim: memo[vid] = res
        return res

    if typ.__name__ == 'ndarray':
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


def traced_eval(func, args):
    ctx = _Context()
    wrapped_args = tuple(_wrap(ctx, a, deferred=True) for a in args)
    res = func(*wrapped_args)
    
    memo = {}
    for orig, wrapped in ctx.sync:
        if isinstance(orig, list): orig[:] = _unwrap(wrapped, memo)
        elif type(orig).__name__ == 'ndarray': orig[...] = _unwrap(wrapped, memo)
        
    return ctx.trace, _unwrap(res, memo)


def trace_to_bytedmd(trace, bytes_per_element):
    if bytes_per_element == 1: return sum(math.isqrt(d - 1) + 1 for d in trace)
    bpe = bytes_per_element
    return sum(_sum_usqrt(d * bpe) - _sum_usqrt((d - 1) * bpe) for d in trace)


def inspect_ir(func, args):
    ctx = _Context()
    wrapped_args = tuple(_wrap(ctx, a, deferred=True) for a in args)
    func(*wrapped_args)
    return ctx.ir


def format_ir(ir):
    out, total = [], 0
    for ev in ir:
        if ev[0] == 'STORE':
            out.append(f"STORE v{ev[1]}")
        elif ev[0] == 'READ':
            cost = math.isqrt(ev[2] - 1) + 1
            out.append(f"  READ v{ev[1]}@{ev[2]}  cost={cost}")
        else:
            _, name, keys, depths, _ok = ev
            cost = sum(math.isqrt(d - 1) + 1 for d in depths)
            total += cost
            rd = ", ".join(f"v{k}@{d}" for k, d in zip(keys, depths))
            out.append(f"OP    {name}({rd})  cost={cost}")
    out.append(f"# total cost = {total}")
    return "\n".join(out)


def _collect_keys(wrapped, name, names):
    """Walk a wrapped structure to map tracking keys to human-readable names."""
    if isinstance(wrapped, _Tracked):
        names[wrapped._key] = name
    elif isinstance(wrapped, (list, tuple)):
        for i, v in enumerate(wrapped):
            _collect_keys(v, f"{name}[{i}]", names)
    elif type(wrapped).__name__ == 'ndarray':
        import numpy as np
        for idx in np.ndindex(wrapped.shape):
            sub = name + ''.join(f'[{i}]' for i in idx)
            _collect_keys(wrapped[idx], sub, names)


_OP_SYMBOLS = {
    'add': '+', 'sub': '-', 'mul': '*', 'truediv': '/', 'floordiv': '//',
    'mod': '%', 'pow': '**', 'matmul': '@', 'lshift': '<<', 'rshift': '>>',
    'and_': '&', 'or_': '|', 'xor': '^', 'eq': '==', 'ne': '!=',
    'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>=', 'neg': '-', 'pos': '+',
    'abs': 'abs', 'invert': '~',
}


def trace_ir(func, args):
    """Replay an IR step-by-step with variable names and stack state.

    Uses function parameter names for arguments (e.g., ``a``, ``A[0][1]``)
    and derives names for intermediates from the operations that produced
    them (e.g., ``a*b``, ``(a*b)+(c*d)``).

    Returns the formatted string (also printed).
    """
    import inspect
    ctx = _Context()
    wrapped_args = tuple(_wrap(ctx, a, deferred=True) for a in args)
    func(*wrapped_args)
    ir = ctx.ir

    # Build key → name map from parameter names.
    names = {}
    try:
        param_names = list(inspect.signature(func).parameters.keys())
    except (ValueError, TypeError):
        param_names = [f'arg{i}' for i in range(len(args))]
    for pname, warg in zip(param_names, wrapped_args):
        _collect_keys(warg, pname, names)

    def n(key):
        return names.get(key, f'v{key}')

    stack = []
    out = []
    total = 0

    def fmt_stack():
        return '[' + ', '.join(n(k) for k in stack) + ']'

    for ev in ir:
        tag = ev[0]
        if tag == 'STORE':
            key = ev[1]
            stack.append(key)
            out.append(f"STORE {n(key):<20}              stack={fmt_stack()}")
        elif tag == 'READ':
            key, depth = ev[1], ev[2]
            cost = math.isqrt(depth - 1) + 1
            if key not in stack:
                stack.append(key)
            stack.remove(key)
            stack.append(key)
            out.append(f"  READ {n(key)}@{depth:<3} cost={cost:<3}          stack={fmt_stack()}")
        elif tag == 'OP':
            _, opname, keys, depths, out_key = ev
            total += sum(math.isqrt(d - 1) + 1 for d in depths)
            sym = _OP_SYMBOLS.get(opname, opname)
            # Derive a name for the result.
            if out_key is not None:
                if len(keys) == 2:
                    names[out_key] = f"({n(keys[0])}{sym}{n(keys[1])})"
                elif len(keys) == 1:
                    names[out_key] = f"{sym}({n(keys[0])})"
                else:
                    names[out_key] = f"{opname}({', '.join(n(k) for k in keys)})"

    out.append(f"# total cost = {total}")
    result = "\n".join(out)
    print(result)
    return result


def bytedmd(func, args, bytes_per_element=1):
    trace, _ = traced_eval(func, args)
    return trace_to_bytedmd(trace, bytes_per_element)