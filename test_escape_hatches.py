#!/usr/bin/env python3
"""
Escape hatch demonstrations for the strict (bytecode-level) ByteDMD tracer.

The regular tracer (bytedmd.py) operates at the dunder-method layer.
Adversarial code can perform untracked work by avoiding dunders entirely
(local arrays, identity ops, f-strings, math.trunc, etc.).

The strict tracer (bytedmd_strict.py) operates at the CPython bytecode
layer via sys.settrace + frame.f_trace_opcodes. It charges reads on every
LOAD_FAST, BINARY_OP, COMPARE_OP, FOR_ITER, CALL_*, etc. — operations that
are completely invisible to the proxy. None of the 6 escape hatches can
avoid being charged.

Each test below runs both tracers on the same adversarial function and
asserts that the strict tracer reports a meaningfully higher cost.
"""
import math
import bytedmd
import bytedmd_strict


def _both(func, args):
    """Run both tracers, return (proxy_cost, settrace_cost).

    Returns -1 for proxy_cost if it raises AssertionError (strict refusal).
    """
    try:
        proxy_cost = bytedmd.bytedmd(func, args)
    except AssertionError:
        proxy_cost = -1
    settrace_cost = bytedmd_strict.bytedmd(func, args)
    return proxy_cost, settrace_cost


# ─────────────────────── ESCAPE HATCH 1 ──────────────────────────
# Local arrays / globals: the proxy only wraps function arguments.

def test_hatch1_local_array_summing():
    """1000-element local array summed in a loop. Proxy sees ~1 read."""
    def f(a):
        local_array = [10] * 100
        s = 0
        for x in local_array:
            s = s + x
        return a + s

    proxy_cost, st_cost = _both(f, (5,))
    print(f"\n[H1a] local array N=100: proxy={proxy_cost}, settrace={st_cost}")
    # The settrace tracer must see significantly more work (>= 100 reads)
    assert st_cost > 100, f"settrace cost {st_cost} should be >100"
    assert st_cost >= 100 * max(proxy_cost, 1), \
        f"settrace {st_cost} should be >= 100x proxy {proxy_cost}"


def test_hatch1_global_lookup_in_loop():
    """Loop reading from a module-level global table."""
    GLOBAL_TABLE = list(range(50))

    def f(a):
        s = 0
        for i in range(50):
            s = s + GLOBAL_TABLE[i]
        return a + s

    proxy_cost, st_cost = _both(f, (5,))
    print(f"[H1b] global table loop: proxy={proxy_cost}, settrace={st_cost}")
    assert st_cost > 50


# ─────────────────────── ESCAPE HATCH 2 ──────────────────────────
# Exception side-channels: trigger ZeroDivisionError to leak the value
# without invoking __bool__.

def test_hatch2_exception_side_channel():
    """
    Probe `a` via `1/(a-i)` which raises ZeroDivisionError when i==a.
    Then run a "secret" untracked loop based on the leaked value.
    The settrace tracer charges every BINARY_TRUE_DIVIDE in the probe loop
    AND every iteration of the secret loop.
    """
    def f(a):
        val = None
        for i in range(20):
            try:
                _ = 1 / (a - i)
            except ZeroDivisionError:
                val = i
                break
        result = a
        if val is not None:
            for _ in range(10):
                result = result + 1
        return result

    proxy_cost, st_cost = _both(f, (5,))
    print(f"[H2] exception probe: proxy={proxy_cost}, settrace={st_cost}")
    assert st_cost > max(proxy_cost, 1)


# ─────────────────────── ESCAPE HATCH 3 ──────────────────────────
# Catching tracer assertions: try/except AssertionError.
# The settrace tracer never raises, so this fails.

def test_hatch3_catching_assertion():
    """Even with try/except AssertionError, the secret loop is charged."""
    def f(a):
        try:
            if a > 0:
                pass
        except AssertionError:
            return a * 100
        s = 0
        for _ in range(20):
            s = s + 1
        return a + s

    proxy_cost, st_cost = _both(f, (5,))
    print(f"[H3] catch AssertionError: proxy={proxy_cost}, settrace={st_cost}")
    assert st_cost > max(proxy_cost, 1)


# ─────────────────────── ESCAPE HATCH 4 ──────────────────────────
# Identity / introspection ops: `is`, `id()`, `type()` evaluate at C level.

def test_hatch4_identity_op():
    """`a is b` produces a native bool that the proxy can't see."""
    def f(a, b):
        same = (a is b)
        if same:
            s = 0
            for _ in range(10):
                s = s + 1
            return a + b + s
        return a + b

    proxy_cost, st_cost = _both(f, (5, 5))
    print(f"[H4a] identity op: proxy={proxy_cost}, settrace={st_cost}")
    assert st_cost > max(proxy_cost, 1)


def test_hatch4_type_introspection():
    """`type(a) is int` leaks via type(), then triggers a secret loop."""
    def f(a):
        if type(a) is int:
            s = 0
            for _ in range(15):
                s = s + 1
            return a + s
        return a

    proxy_cost, st_cost = _both(f, (5,))
    print(f"[H4b] type introspection: proxy={proxy_cost}, settrace={st_cost}")
    assert st_cost > max(proxy_cost, 1)


# ─────────────────────── ESCAPE HATCH 5 ──────────────────────────
# Stringification: f-strings, str(), repr() return native strings.

def test_hatch5_f_string():
    """f-string converts the proxy to a string we can branch on."""
    def f(a):
        s = f"{a}"
        if len(s) > 0:
            t = 0
            for _ in range(15):
                t = t + 1
            return a + t
        return a

    proxy_cost, st_cost = _both(f, (5,))
    print(f"[H5] f-string: proxy={proxy_cost}, settrace={st_cost}")
    assert st_cost > max(proxy_cost, 1)


# ─────────────────────── ESCAPE HATCH 6 ──────────────────────────
# Math coercions: math.trunc/ceil/floor and round() use parallel dunders.

def test_hatch6_math_trunc():
    """math.trunc(a) extracts a raw int, used in a secret loop."""
    def f(a):
        raw = math.trunc(a)
        s = 0
        for _ in range(raw):
            s = s + 1
        return a + s

    proxy_cost, st_cost = _both(f, (10.0,))
    print(f"[H6] math.trunc: proxy={proxy_cost}, settrace={st_cost}")
    assert st_cost > max(proxy_cost, 1)


# ─────────────────────── Summary ──────────────────────────

def test_print_summary_table():
    """Print a side-by-side cost comparison for all escape hatches."""
    print()
    print("=" * 80)
    print(f"{'Escape Hatch':<45} {'Proxy':>15} {'Settrace':>15}")
    print("-" * 80)

    GLOBAL_TBL = list(range(50))

    cases = [
        ("1a. Local array (100 elem) summed",
         lambda a: __import__('functools').reduce(lambda s, x: s + x, [10] * 100, a)),
        ("2. Exception side channel",
         lambda a: _exception_probe(a)),
        ("3. Catching tracer assertion",
         lambda a: _catch_assertion(a)),
        ("4a. Identity operator (is)",
         lambda a: _identity_op(a, a)),
        ("4b. Type introspection",
         lambda a: _type_introspection(a)),
        ("5. f-string leak",
         lambda a: _fstring_leak(a)),
        ("6. math.trunc coercion",
         lambda a: _math_trunc(a)),
    ]
    inputs = [(5,), (5,), (5,), (5,), (5,), (5,), (10.0,)]

    for (name, fn), args in zip(cases, inputs):
        try:
            p = bytedmd.bytedmd(fn, args)
        except Exception as e:
            p = f"err"
        try:
            s = bytedmd_strict.bytedmd(fn, args)
        except Exception as e:
            s = f"err"
        print(f"{name:<45} {str(p):>15} {str(s):>15}")
    print("=" * 80)


def _exception_probe(a):
    val = None
    for i in range(20):
        try:
            _ = 1 / (a - i)
        except ZeroDivisionError:
            val = i
            break
    result = a
    if val is not None:
        for _ in range(10):
            result = result + 1
    return result


def _catch_assertion(a):
    try:
        if a > 0:
            pass
    except AssertionError:
        return a * 100
    s = 0
    for _ in range(20):
        s = s + 1
    return a + s


def _identity_op(a, b):
    same = (a is b)
    if same:
        s = 0
        for _ in range(10):
            s = s + 1
        return a + b + s
    return a + b


def _type_introspection(a):
    if type(a) is int:
        s = 0
        for _ in range(15):
            s = s + 1
        return a + s
    return a


def _fstring_leak(a):
    s = f"{a}"
    if len(s) > 0:
        t = 0
        for _ in range(15):
            t = t + 1
        return a + t
    return a


def _math_trunc(a):
    raw = math.trunc(a)
    s = 0
    for _ in range(raw):
        s = s + 1
    return a + s


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
