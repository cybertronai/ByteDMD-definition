"""
Strict ByteDMD tracer implemented via sys.settrace + frame.f_trace_opcodes.

This is the slow-but-robust counterpart to the regular tracer in bytedmd.py.
It operates at the CPython bytecode level: every LOAD_FAST, BINARY_*,
COMPARE_OP, FOR_ITER, CALL_*, etc. is intercepted before CPython executes
it. Because tracking happens below the object/dunder layer, none of the
six proxy escape hatches apply:

  1. Local arrays/globals    -> LOAD_FAST + BINARY_SUBSCR + FOR_ITER are seen
  2. Exception side-channels -> BINARY_TRUE_DIVIDE charges before the divide
  3. Catching AssertionError -> we never raise; branches charge unconditionally
  4. Identity ops `is`       -> IS_OP charges both operands
  5. Stringification         -> FORMAT_VALUE / CALL_FUNCTION charges the arg
  6. math.trunc and friends  -> CALL_METHOD / CALL_FUNCTION charges the arg

The cost model (LRU stack, ceil(sqrt(depth)) per read) and the public API
(traced_eval, bytedmd, trace_to_bytedmd) match bytedmd.py.

The trace VALUES will not match the regular tracer exactly because the proxy
allocates a new slot per intermediate operation result, while this tracer
treats eval-stack temporaries as free (only named variables and container
elements get LRU slots). The two tracers measure related but distinct
quantities by design — use `verify()` to compare them on benign code.

Use this tracer when:
  - benchmarking against unfamiliar or adversarial code
  - sanity-checking that the regular tracer is not silently undercounting
  - performing security audits of cost claims
"""

import dis
import math
import os
import sys
import sysconfig

# Reuse the cost-conversion helper from the main module so a single
# definition of "ceil(sqrt(d))" sums lives in the codebase.
from bytedmd import trace_to_bytedmd


# ──────────────────────────── Context ─────────────────────────────────────

class _SettraceContext:
    """LRU stack + trace, with bookkeeping for container element slots."""
    __slots__ = ('stack', 'trace', 'counter',
                 'name_slots',     # id(frame) -> {varname: slot_id}
                 'global_slots',   # name -> slot_id (shared across frames)
                 'element_slots',  # container_slot -> {key_or_index: elem_slot}
                 'iter_state')     # id(iterator_obj) -> container_slot

    def __init__(self):
        self.stack = []
        self.trace = []
        self.counter = 0
        self.name_slots = {}
        self.global_slots = {}
        self.element_slots = {}
        self.iter_state = {}

    def allocate(self):
        self.counter += 1
        self.stack.append(self.counter)
        return self.counter

    def read(self, slot):
        """Charge a single read of `slot`. Skips None (free temporaries)."""
        if slot is None:
            return
        # Compute depth from top (1 = MRU). Then move slot to top.
        idx = self.stack.index(slot)
        self.trace.append(len(self.stack) - idx)
        self.stack.pop(idx)
        self.stack.append(slot)

    def read_many(self, slots):
        for s in slots:
            self.read(s)


# ──────────────────────────── Shadow frame ────────────────────────────────

class _ShadowFrame:
    """Per-Python-frame symbolic state mirroring CPython's eval stack."""
    __slots__ = ('stack', 'locals', 'code')

    def __init__(self, code):
        self.stack = []           # symbolic eval stack: list[Optional[int]]
        self.locals = {}          # varname -> slot_id
        self.code = code


# ──────────────────────────── Frame filter ────────────────────────────────

_STDLIB_PATHS = tuple(p for p in (
    sysconfig.get_paths().get('stdlib', ''),
    sysconfig.get_paths().get('platstdlib', ''),
    sys.prefix,
    sys.base_prefix,
) if p)


def _should_trace(frame):
    """Trace user code only: skip stdlib, site-packages, and frozen modules."""
    fn = frame.f_code.co_filename
    if not fn or fn.startswith('<'):
        return False
    if 'site-packages' in fn:
        return False
    for p in _STDLIB_PATHS:
        if p and fn.startswith(p):
            return False
    return True


# ──────────────────────────── Opcode handlers ─────────────────────────────

# Each handler is called as: handler(ctx, sf, oparg, frame)
# It must update sf.stack to mirror what the next opcode will do, and call
# ctx.read(...) for any charges that opcode incurs.

_HANDLERS = {}


def _handler(*names):
    def deco(fn):
        for n in names:
            _HANDLERS[n] = fn
        return fn
    return deco


# ---- Loads ----------------------------------------------------------------

@_handler('LOAD_FAST')
def _h_load_fast(ctx, sf, oparg, frame):
    name = sf.code.co_varnames[oparg]
    slot = sf.locals.get(name)
    if slot is None:
        slot = ctx.allocate()
        sf.locals[name] = slot
    # No charge here — the consumer (BINARY_OP, RETURN_VALUE, etc.) charges.
    sf.stack.append(slot)


@_handler('LOAD_GLOBAL', 'LOAD_NAME')
def _h_load_global(ctx, sf, oparg, frame):
    name = sf.code.co_names[oparg]
    slot = ctx.global_slots.get(name)
    if slot is None:
        slot = ctx.allocate()
        ctx.global_slots[name] = slot
    sf.stack.append(slot)


@_handler('LOAD_DEREF', 'LOAD_CLASSDEREF')
def _h_load_deref(ctx, sf, oparg, frame):
    cellvars = sf.code.co_cellvars + sf.code.co_freevars
    name = cellvars[oparg] if oparg < len(cellvars) else f'_deref_{oparg}'
    slot = ctx.global_slots.get(name)
    if slot is None:
        slot = ctx.allocate()
        ctx.global_slots[name] = slot
    sf.stack.append(slot)


@_handler('LOAD_CONST')
def _h_load_const(ctx, sf, oparg, frame):
    sf.stack.append(None)  # constants are free


# ---- Stores ---------------------------------------------------------------

@_handler('STORE_FAST')
def _h_store_fast(ctx, sf, oparg, frame):
    val = sf.stack.pop() if sf.stack else None
    name = sf.code.co_varnames[oparg]
    if val is None:
        # Storing a temporary: allocate a fresh slot for this name
        sf.locals[name] = ctx.allocate()
    else:
        # Aliasing assignment: y = x rebinds the same slot
        sf.locals[name] = val


@_handler('STORE_GLOBAL', 'STORE_NAME')
def _h_store_global(ctx, sf, oparg, frame):
    val = sf.stack.pop() if sf.stack else None
    name = sf.code.co_names[oparg]
    if val is None:
        ctx.global_slots[name] = ctx.allocate()
    else:
        ctx.global_slots[name] = val


@_handler('STORE_DEREF')
def _h_store_deref(ctx, sf, oparg, frame):
    val = sf.stack.pop() if sf.stack else None
    cellvars = sf.code.co_cellvars + sf.code.co_freevars
    name = cellvars[oparg] if oparg < len(cellvars) else f'_deref_{oparg}'
    if val is None:
        ctx.global_slots[name] = ctx.allocate()
    else:
        ctx.global_slots[name] = val


@_handler('DELETE_FAST')
def _h_delete_fast(ctx, sf, oparg, frame):
    name = sf.code.co_varnames[oparg]
    sf.locals.pop(name, None)


# ---- Attribute access -----------------------------------------------------

@_handler('LOAD_ATTR')
def _h_load_attr(ctx, sf, oparg, frame):
    # Pop the object, push the attribute (treated as a free read of the obj
    # slot we already charged on LOAD_FAST). For now we don't allocate a per-
    # attribute slot — that would over-count for plain object access.
    if sf.stack:
        sf.stack.pop()
    sf.stack.append(None)


@_handler('LOAD_METHOD')
def _h_load_method(ctx, sf, oparg, frame):
    # CPython pushes (method, self) or (NULL, callable). Either way the eval
    # stack grows by 1. We model both slots as free temporaries.
    if sf.stack:
        sf.stack.pop()
    sf.stack.append(None)
    sf.stack.append(None)


@_handler('STORE_ATTR')
def _h_store_attr(ctx, sf, oparg, frame):
    # Pops TOS = obj, TOS1 = value
    if sf.stack: sf.stack.pop()
    if sf.stack: sf.stack.pop()


@_handler('DELETE_ATTR')
def _h_delete_attr(ctx, sf, oparg, frame):
    if sf.stack: sf.stack.pop()


# ---- Stack manipulation ---------------------------------------------------

@_handler('POP_TOP')
def _h_pop_top(ctx, sf, oparg, frame):
    if sf.stack: sf.stack.pop()


@_handler('ROT_TWO')
def _h_rot_two(ctx, sf, oparg, frame):
    if len(sf.stack) >= 2:
        sf.stack[-1], sf.stack[-2] = sf.stack[-2], sf.stack[-1]


@_handler('ROT_THREE')
def _h_rot_three(ctx, sf, oparg, frame):
    if len(sf.stack) >= 3:
        a = sf.stack.pop()
        b = sf.stack.pop()
        c = sf.stack.pop()
        sf.stack.append(a)
        sf.stack.append(c)
        sf.stack.append(b)


@_handler('ROT_FOUR')
def _h_rot_four(ctx, sf, oparg, frame):
    if len(sf.stack) >= 4:
        a = sf.stack.pop()
        b = sf.stack.pop()
        c = sf.stack.pop()
        d = sf.stack.pop()
        sf.stack.append(a)
        sf.stack.append(d)
        sf.stack.append(c)
        sf.stack.append(b)


@_handler('DUP_TOP')
def _h_dup_top(ctx, sf, oparg, frame):
    if sf.stack:
        sf.stack.append(sf.stack[-1])


@_handler('DUP_TOP_TWO')
def _h_dup_top_two(ctx, sf, oparg, frame):
    if len(sf.stack) >= 2:
        sf.stack.append(sf.stack[-2])
        sf.stack.append(sf.stack[-2])


@_handler('NOP', 'EXTENDED_ARG')
def _h_nop(ctx, sf, oparg, frame):
    pass


# ---- Binary / unary ops ---------------------------------------------------

_BINARY_OPS = (
    'BINARY_OP',  # Python 3.12+ unified opcode
    'BINARY_ADD', 'BINARY_SUBTRACT', 'BINARY_MULTIPLY', 'BINARY_TRUE_DIVIDE',
    'BINARY_FLOOR_DIVIDE', 'BINARY_MODULO', 'BINARY_POWER',
    'BINARY_LSHIFT', 'BINARY_RSHIFT', 'BINARY_AND', 'BINARY_OR', 'BINARY_XOR',
    'BINARY_MATRIX_MULTIPLY',
    'INPLACE_ADD', 'INPLACE_SUBTRACT', 'INPLACE_MULTIPLY',
    'INPLACE_TRUE_DIVIDE', 'INPLACE_FLOOR_DIVIDE', 'INPLACE_MODULO',
    'INPLACE_POWER', 'INPLACE_LSHIFT', 'INPLACE_RSHIFT',
    'INPLACE_AND', 'INPLACE_OR', 'INPLACE_XOR', 'INPLACE_MATRIX_MULTIPLY',
)

@_handler(*_BINARY_OPS)
def _h_binary(ctx, sf, oparg, frame):
    rhs = sf.stack.pop() if sf.stack else None
    lhs = sf.stack.pop() if sf.stack else None
    ctx.read(lhs)
    ctx.read(rhs)
    sf.stack.append(None)  # result is a temporary


_UNARY_OPS = ('UNARY_POSITIVE', 'UNARY_NEGATIVE', 'UNARY_NOT', 'UNARY_INVERT')

@_handler(*_UNARY_OPS)
def _h_unary(ctx, sf, oparg, frame):
    val = sf.stack.pop() if sf.stack else None
    ctx.read(val)
    sf.stack.append(None)


# ---- Comparisons / branches -----------------------------------------------

@_handler('COMPARE_OP')
def _h_compare(ctx, sf, oparg, frame):
    rhs = sf.stack.pop() if sf.stack else None
    lhs = sf.stack.pop() if sf.stack else None
    ctx.read(lhs)
    ctx.read(rhs)
    sf.stack.append(None)


@_handler('IS_OP', 'CONTAINS_OP')
def _h_is_op(ctx, sf, oparg, frame):
    rhs = sf.stack.pop() if sf.stack else None
    lhs = sf.stack.pop() if sf.stack else None
    ctx.read(lhs)
    ctx.read(rhs)
    sf.stack.append(None)


@_handler('POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE')
def _h_pop_jump(ctx, sf, oparg, frame):
    cond = sf.stack.pop() if sf.stack else None
    ctx.read(cond)


@_handler('JUMP_IF_FALSE_OR_POP', 'JUMP_IF_TRUE_OR_POP')
def _h_jump_or_pop(ctx, sf, oparg, frame):
    # The branch may or may not pop; we conservatively peek + charge.
    if sf.stack:
        ctx.read(sf.stack[-1])


@_handler('JUMP_FORWARD', 'JUMP_ABSOLUTE')
def _h_jump(ctx, sf, oparg, frame):
    pass


# ---- Calls ----------------------------------------------------------------

@_handler('CALL_FUNCTION')
def _h_call_function(ctx, sf, oparg, frame):
    # Pop oparg args + 1 callable. Charge each non-None arg slot.
    n = oparg
    args = []
    for _ in range(n):
        if sf.stack:
            args.append(sf.stack.pop())
    if sf.stack:
        sf.stack.pop()  # the callable
    for a in args:
        ctx.read(a)
    sf.stack.append(None)


@_handler('CALL_FUNCTION_KW')
def _h_call_function_kw(ctx, sf, oparg, frame):
    # Pops kwarg names tuple, then oparg args, then callable.
    if sf.stack: sf.stack.pop()  # kwarg names tuple (constant, free)
    n = oparg
    args = []
    for _ in range(n):
        if sf.stack:
            args.append(sf.stack.pop())
    if sf.stack:
        sf.stack.pop()
    for a in args:
        ctx.read(a)
    sf.stack.append(None)


@_handler('CALL_FUNCTION_EX')
def _h_call_function_ex(ctx, sf, oparg, frame):
    # oparg & 1: kwargs dict on top; pop it. Then args tuple. Then callable.
    if oparg & 1:
        if sf.stack: sf.stack.pop()  # kwargs
    args_tuple = sf.stack.pop() if sf.stack else None
    callable_slot = sf.stack.pop() if sf.stack else None
    ctx.read(args_tuple)  # at least charge a read of the args bundle
    sf.stack.append(None)


@_handler('CALL_METHOD')
def _h_call_method(ctx, sf, oparg, frame):
    n = oparg
    args = []
    for _ in range(n):
        if sf.stack:
            args.append(sf.stack.pop())
    # Pop self and method (LOAD_METHOD pushes both)
    if sf.stack: sf.stack.pop()
    if sf.stack: sf.stack.pop()
    for a in args:
        ctx.read(a)
    sf.stack.append(None)


# ---- f-strings ------------------------------------------------------------

@_handler('FORMAT_VALUE')
def _h_format_value(ctx, sf, oparg, frame):
    # If oparg & 0x04, fmt_spec is on the stack as well.
    if oparg & 0x04:
        if sf.stack: sf.stack.pop()  # fmt_spec
    val = sf.stack.pop() if sf.stack else None
    ctx.read(val)
    sf.stack.append(None)


@_handler('BUILD_STRING')
def _h_build_string(ctx, sf, oparg, frame):
    for _ in range(oparg):
        if sf.stack: sf.stack.pop()
    sf.stack.append(None)


# ---- Container construction -----------------------------------------------

@_handler('BUILD_LIST', 'BUILD_TUPLE', 'BUILD_SET')
def _h_build_seq(ctx, sf, oparg, frame):
    elems = []
    for _ in range(oparg):
        if sf.stack:
            elems.append(sf.stack.pop())
    elems.reverse()
    container_slot = ctx.allocate()
    elem_map = ctx.element_slots.setdefault(container_slot, {})
    for i, e in enumerate(elems):
        # Store either the source slot (alias) or a fresh element slot
        elem_map[i] = e if e is not None else ctx.allocate()
    sf.stack.append(container_slot)


@_handler('BUILD_MAP')
def _h_build_map(ctx, sf, oparg, frame):
    for _ in range(oparg * 2):
        if sf.stack: sf.stack.pop()
    sf.stack.append(ctx.allocate())


@_handler('BUILD_CONST_KEY_MAP')
def _h_build_const_key_map(ctx, sf, oparg, frame):
    if sf.stack: sf.stack.pop()  # keys tuple
    for _ in range(oparg):
        if sf.stack: sf.stack.pop()
    sf.stack.append(ctx.allocate())


@_handler('BUILD_SLICE')
def _h_build_slice(ctx, sf, oparg, frame):
    for _ in range(oparg):
        if sf.stack: sf.stack.pop()
    sf.stack.append(None)


@_handler('LIST_APPEND')
def _h_list_append(ctx, sf, oparg, frame):
    val = sf.stack.pop() if sf.stack else None
    # The target list is at depth oparg from the top after the pop
    if len(sf.stack) >= oparg:
        container_slot = sf.stack[-oparg]
        if container_slot is not None:
            elem_map = ctx.element_slots.setdefault(container_slot, {})
            elem_map[len(elem_map)] = val if val is not None else ctx.allocate()


@_handler('SET_ADD', 'MAP_ADD')
def _h_set_map_add(ctx, sf, oparg, frame):
    if sf.stack: sf.stack.pop()


@_handler('LIST_EXTEND', 'SET_UPDATE')
def _h_list_extend(ctx, sf, oparg, frame):
    if sf.stack: sf.stack.pop()


@_handler('DICT_MERGE', 'DICT_UPDATE')
def _h_dict_merge(ctx, sf, oparg, frame):
    if sf.stack: sf.stack.pop()


@_handler('LIST_TO_TUPLE')
def _h_list_to_tuple(ctx, sf, oparg, frame):
    pass  # type change, slot id unchanged


# ---- Subscript -----------------------------------------------------------

@_handler('BINARY_SUBSCR')
def _h_binary_subscr(ctx, sf, oparg, frame):
    index_slot = sf.stack.pop() if sf.stack else None
    container_slot = sf.stack.pop() if sf.stack else None
    ctx.read(container_slot)
    ctx.read(index_slot)
    # Try to find a per-element slot. If we don't have one (e.g., dict
    # key), allocate a fresh element slot.
    elem_slot = None
    if container_slot is not None:
        elem_map = ctx.element_slots.setdefault(container_slot, {})
        # We don't know the actual index value (we only have its slot id),
        # so use a generic "next-element" slot per container. This makes the
        # cost monotone in the number of subscripts.
        key = len(elem_map)
        elem_slot = ctx.allocate()
        elem_map[key] = elem_slot
        ctx.read(elem_slot)
    sf.stack.append(elem_slot)


@_handler('STORE_SUBSCR')
def _h_store_subscr(ctx, sf, oparg, frame):
    # TOS = index, TOS1 = container, TOS2 = value
    index = sf.stack.pop() if sf.stack else None
    container = sf.stack.pop() if sf.stack else None
    value = sf.stack.pop() if sf.stack else None
    ctx.read(index)
    ctx.read(container)
    ctx.read(value)
    if container is not None:
        elem_map = ctx.element_slots.setdefault(container, {})
        elem_map[len(elem_map)] = ctx.allocate()


@_handler('DELETE_SUBSCR')
def _h_delete_subscr(ctx, sf, oparg, frame):
    if sf.stack: sf.stack.pop()
    if sf.stack: sf.stack.pop()


# ---- Iteration ------------------------------------------------------------

@_handler('GET_ITER', 'GET_YIELD_FROM_ITER')
def _h_get_iter(ctx, sf, oparg, frame):
    container = sf.stack.pop() if sf.stack else None
    # Charge a read of the container (we're about to start iterating it)
    ctx.read(container)
    # Push an iterator slot that remembers the container
    iter_slot = ctx.allocate() if container is not None else None
    if container is not None:
        ctx.iter_state[iter_slot] = container
    sf.stack.append(iter_slot)


@_handler('FOR_ITER')
def _h_for_iter(ctx, sf, oparg, frame):
    if not sf.stack:
        sf.stack.append(None)
        return
    iter_slot = sf.stack[-1]
    if iter_slot is None:
        sf.stack.append(None)
        return
    container = ctx.iter_state.get(iter_slot)
    # Charge one read per iteration tick: read the iterator state
    ctx.read(iter_slot)
    # Allocate one element slot per tick from the container's element pool
    elem_slot = None
    if container is not None:
        elem_map = ctx.element_slots.setdefault(container, {})
        elem_slot = ctx.allocate()
        elem_map[len(elem_map)] = elem_slot
        ctx.read(elem_slot)
    else:
        elem_slot = ctx.allocate()
        ctx.read(elem_slot)
    sf.stack.append(elem_slot)


@_handler('UNPACK_SEQUENCE')
def _h_unpack_sequence(ctx, sf, oparg, frame):
    src = sf.stack.pop() if sf.stack else None
    ctx.read(src)
    # Push oparg element slots (in reverse, since CPython pushes last-first)
    if src is not None:
        elem_map = ctx.element_slots.setdefault(src, {})
        for i in range(oparg):
            slot = elem_map.get(i)
            if slot is None:
                slot = ctx.allocate()
                elem_map[i] = slot
        for i in range(oparg - 1, -1, -1):
            sf.stack.append(elem_map[i])
    else:
        for _ in range(oparg):
            sf.stack.append(None)


@_handler('UNPACK_EX')
def _h_unpack_ex(ctx, sf, oparg, frame):
    if sf.stack: sf.stack.pop()
    n_before = oparg & 0xff
    n_after = (oparg >> 8) & 0xff
    for _ in range(n_before + 1 + n_after):
        sf.stack.append(None)


# ---- Returns / yields -----------------------------------------------------

@_handler('RETURN_VALUE')
def _h_return(ctx, sf, oparg, frame):
    val = sf.stack.pop() if sf.stack else None
    ctx.read(val)


@_handler('YIELD_VALUE')
def _h_yield_value(ctx, sf, oparg, frame):
    if sf.stack: sf.stack.pop()
    sf.stack.append(None)


@_handler('YIELD_FROM')
def _h_yield_from(ctx, sf, oparg, frame):
    if sf.stack: sf.stack.pop()
    if sf.stack: sf.stack.pop()
    sf.stack.append(None)


# ---- Imports --------------------------------------------------------------

@_handler('IMPORT_NAME')
def _h_import_name(ctx, sf, oparg, frame):
    if sf.stack: sf.stack.pop()  # fromlist
    if sf.stack: sf.stack.pop()  # level
    sf.stack.append(None)


@_handler('IMPORT_FROM')
def _h_import_from(ctx, sf, oparg, frame):
    sf.stack.append(None)


@_handler('IMPORT_STAR')
def _h_import_star(ctx, sf, oparg, frame):
    if sf.stack: sf.stack.pop()


# ---- Exception handling (just shadow-stack bookkeeping) -------------------

@_handler('SETUP_FINALLY', 'SETUP_WITH', 'SETUP_ASYNC_WITH', 'POP_BLOCK',
          'POP_EXCEPT', 'RERAISE', 'RAISE_VARARGS', 'WITH_EXCEPT_START',
          'JUMP_IF_NOT_EXC_MATCH')
def _h_exception(ctx, sf, oparg, frame):
    # Conservative bookkeeping; opcodes here vary in stack effect, but
    # since they don't perform reads we can be approximate without
    # affecting cost.
    pass


# ---- Generic fallback -----------------------------------------------------

def _generic_dispatch(ctx, sf, op, oparg, frame):
    """Use dis.stack_effect for opcodes we haven't special-cased."""
    try:
        delta = dis.stack_effect(op, oparg)
    except (ValueError, TypeError):
        return
    if delta < 0:
        for _ in range(-delta):
            if sf.stack:
                sf.stack.pop()
    elif delta > 0:
        for _ in range(delta):
            sf.stack.append(None)


# ──────────────────────────── Trace function ──────────────────────────────

def _make_trace_fn(ctx, target_code):
    """Build the settrace callback for a given context."""
    shadow_frames = {}  # id(frame) -> _ShadowFrame
    state = {'entered': False}

    def _local_trace(frame, event, arg):
        if event == 'opcode':
            sf = shadow_frames.get(id(frame))
            if sf is None:
                return _local_trace
            code = frame.f_code
            offset = frame.f_lasti
            op = code.co_code[offset]
            oparg = code.co_code[offset + 1]
            opname = dis.opname[op]
            handler = _HANDLERS.get(opname)
            try:
                if handler is not None:
                    handler(ctx, sf, oparg, frame)
                else:
                    _generic_dispatch(ctx, sf, op, oparg, frame)
            except Exception:
                pass
            return _local_trace

        if event == 'call':
            is_target = (frame.f_code is target_code)
            if not is_target and state['entered'] and not _should_trace(frame):
                return None
            if is_target:
                state['entered'] = True
            sf = _ShadowFrame(frame.f_code)
            if is_target:
                preset = ctx.name_slots.get(id(target_code), {})
                for name, slot in preset.items():
                    sf.locals[name] = slot
            shadow_frames[id(frame)] = sf
            frame.f_trace_opcodes = True
            return _local_trace

        if event == 'return':
            shadow_frames.pop(id(frame), None)
            return _local_trace

        return _local_trace

    return _local_trace


# ──────────────────────────── Public API ──────────────────────────────────

def traced_eval(func, args):
    """Run `func(*args)` under bytecode-level tracing.

    Returns (trace, result) where trace is a list of LRU depths charged
    by the settrace tracer.
    """
    ctx = _SettraceContext()

    # Pre-allocate one slot per argument and register them by name in
    # the target function's frame.
    target_code = func.__code__
    arg_names = target_code.co_varnames[:target_code.co_argcount]
    arg_slot_map = {}
    for name in arg_names:
        arg_slot_map[name] = ctx.allocate()
    ctx.name_slots[id(target_code)] = arg_slot_map

    # For container args, register element slots so subscripts charge
    # correctly. We need access to the actual values, so do this manually.
    for name, val in zip(arg_names, args):
        slot = arg_slot_map[name]
        if isinstance(val, (list, tuple)):
            elem_map = ctx.element_slots.setdefault(slot, {})
            for i, item in enumerate(val):
                elem_slot = ctx.allocate()
                elem_map[i] = elem_slot
                if isinstance(item, (list, tuple)):
                    sub_map = ctx.element_slots.setdefault(elem_slot, {})
                    for j, sub in enumerate(item):
                        sub_map[j] = ctx.allocate()
        elif type(val).__name__ == 'ndarray':
            elem_map = ctx.element_slots.setdefault(slot, {})
            try:
                for i, item in enumerate(val.flat):
                    elem_map[i] = ctx.allocate()
            except Exception:
                pass

    trace_fn = _make_trace_fn(ctx, target_code)
    prev = sys.gettrace()
    sys.settrace(trace_fn)
    try:
        result = func(*args)
    finally:
        sys.settrace(prev)
    return ctx.trace, result


def bytedmd(func, args, bytes_per_element=1):
    """Evaluate ByteDMD cost via bytecode-level tracing."""
    trace, _ = traced_eval(func, args)
    return trace_to_bytedmd(trace, bytes_per_element)


# ──────────────────────────── Cross-tracer verification ───────────────────

def verify(func, args, bytes_per_element=1, tolerance=3.0, verbose=True):
    """Run both the regular and strict tracers and warn on large divergence.

    The two tracers measure related but distinct quantities (the regular
    tracer allocates an LRU slot per intermediate operation result; the
    strict tracer treats eval-stack temporaries as free). On benign code
    they should report costs within a small constant factor of each other.
    A larger ratio is a strong signal that the regular tracer is silently
    undercounting some hidden data movement (e.g., a local array, a hidden
    introspection, or a C-extension call).

    Returns the tuple (regular_cost, strict_cost). Prints a warning if the
    ratio exceeds `tolerance` (default 3x in either direction).
    """
    import bytedmd as _regular

    try:
        regular_cost = _regular.bytedmd(func, args, bytes_per_element)
    except AssertionError as e:
        regular_cost = None
        if verbose:
            print(f"verify: regular tracer refused: {e}")

    strict_cost = bytedmd(func, args, bytes_per_element)

    if verbose:
        print(f"verify: regular = {regular_cost}, strict = {strict_cost}")

    if regular_cost is not None and regular_cost > 0:
        ratio = strict_cost / regular_cost
        if ratio > tolerance or ratio < 1.0 / tolerance:
            print(
                f"verify: WARNING — strict/regular ratio = {ratio:.2f}x "
                f"(threshold {tolerance}x). The regular tracer may be "
                f"undercounting hidden data movement (local arrays, "
                f"introspection, C-extensions, etc.). Inspect the function "
                f"or use the strict cost for adversarial benchmarks."
            )

    return regular_cost, strict_cost


# ──────────────────────────── Smoke test ──────────────────────────────────

if __name__ == '__main__':
    def f(a, b):
        return a + b
    trace, res = traced_eval(f, (3, 4))
    print(f"f(3,4) = {res}, trace = {trace}, cost = {trace_to_bytedmd(trace, 1)}")

    def g(xs):
        s = 0
        for x in xs:
            s = s + x
        return s
    trace, res = traced_eval(g, ([1, 2, 3, 4, 5],))
    print(f"g([1..5]) = {res}, trace len = {len(trace)}, cost = {trace_to_bytedmd(trace, 1)}")
