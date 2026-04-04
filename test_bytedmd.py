#!/usr/bin/env python3
import numpy as np
from bytedmd import bytedmd, traced_eval, trace_to_cost


def my_add(a, b, c, d):
    return b + c


def test_my_add():
    cost = bytedmd(my_add, (1, 2, 3, 4))
    assert cost == 4

    # trace counts depth in terms of number of elements, not bytes
    trace, _ = traced_eval(my_add, (1, 2, 3, 4))
    assert trace == [3, 2]

    assert trace_to_cost(trace, bytes_per_element=1) == 4
    assert trace_to_cost(trace, bytes_per_element=2) == 10

    assert bytedmd(my_add, (1, 2, 3, 4), bytes_per_element=2) == 10


def my_composite_func(a, b, c, d):
    e = b + c
    f = a + d
    return e > f


def test_my_composite_func():
    trace, result = traced_eval(my_composite_func, (1, 2, 3, 4))
    assert trace == [3, 2, 5, 4, 4, 1]
    cost = bytedmd(my_composite_func, (1, 2, 3, 4))
    assert cost == 12

def test_dot_product():
    def dot(a, b):
        return sum(i1 * i2 for i1, i2 in zip(a, b))
        
    a, b = [0, 1], [2, 3]
    trace, result = traced_eval(dot, (a, b))

    assert trace == [4, 2, 1, 6, 5, 4, 1]
    assert result == 3
    assert bytedmd(dot, (a, b)) == 14


def test_branching_and_comparisons_trace():
    def my_relu(a):
        if a > 0:
            return a * 2
        return a
        
    # Branch taken: traces reading `a` twice (`a > 0` and `a * 2`)
    trace_pos, _ = traced_eval(my_relu, (5,))
    assert trace_pos == [1, 1]
    
    # Branch skipped: traces reading `a` once (`a > 0`)
    trace_neg, _ = traced_eval(my_relu, (-5,))
    assert trace_neg == [1]


def test_divmod_tuple_allocation_trace():
    """
    6. Tests operations natively returning multiple tracked values.
    divmod(a, b) evaluates to a tuple (q, r), sequentially triggering 
    multiple allocations on the LRU stack.
    """
    def my_divmod(a, b):
        q, r = divmod(a, b)
        return q + r + a
        
    trace, result = traced_eval(my_divmod, (10, 3))
    assert trace == [2, 1, 2, 1, 1, 5]


### Python gotchas
# Because we implement ByteDMD in Python by wrapping Python objects, our Python framework deviates from idealized description in the README
# for certain cases, illustrated by tests below.

def test_gotcha_constant_ops():
    """
    Limitation of Python model: constants are not tracked
    """
    def f(a):
        return 10 - a * 10
    
    trace, _ = traced_eval(f, (5,))
    assert trace == [1, 1]


def test_gotcha_pure_memory_movement_is_free_trace():
    """
    Limitation of Python model: pure list index without computation does not trigger math magic methods,
    hence generating no read trace.
    """
    def transpose(A):
        n = len(A)
        return [[A[j][i] for j in range(n)] for i in range(n)]
    
    A = [[1, 2], [3, 4]]
    trace, result = traced_eval(transpose, (A,))
    
    assert trace == []
    assert result == [[1, 3], [2, 4]]

def test_gotcha_implicit_boolean_bypass_trace():
    """
    WEAKNESS 1: Implicit truthiness evaluation bypasses the trace.
    Python evaluates an object's truthiness using `__bool__`. Because `_TrackedValue` 
    does not override it, Python treats ALL wrapped values as `True` (since they are objects).
    This silently executes the wrong branch AND completely bypasses the read trace.
    """
    def implicit_branch(a):
        if a:  # Wrongly evaluates to True for a=0, generates NO trace!
            return a + 10
        return a

    def explicit_branch(a):
        if a != 0:  # Correctly evaluates to False for a=0, generates trace!
            return a + 10
        return a

    trace_implicit, result_implicit = traced_eval(implicit_branch, (0,))
    # Misses the condition check entirely, only tracking the `a + 10` execution!
    assert trace_implicit == [1]
    assert result_implicit == 10

    trace_explicit, result_explicit = traced_eval(explicit_branch, (0,))
    # Correctly traces the explicit `a != 0` check
    assert trace_explicit == [1]
    assert result_explicit == 0

def test_gotcha_short_circuit_logic_bypass_trace():
    """
    WEAKNESS 2: Python's `and` / `or` keywords do not invoke magic methods.
    They rely on implicit truthiness, completely circumventing read tracing and
    returning incorrect mathematical results.
    """
    def logical_and(a, b):
        return a and b
        
    trace, result = traced_eval(logical_and, (0, 5))
    assert trace == []
    assert result == 5

def test_gotcha_comparison_untracking_trace():
    """
    WEAKNESS 3: Native Booleans escape the LRU Stack.
    To allow standard Python control flow (`if a > b:`), comparison operators 
    intentionally evaluate directly to native Python booleans. If an algorithm 
    subsequently uses these booleans mathematically, they are completely untracked.
    """
    def compare_and_use(a, b):
        c = a > b  # 'c' escapes the tracker and becomes a raw Python boolean (True)
        return c + a # 'c' is mathematically used, but its read generates no cost!
        
    trace, result = traced_eval(compare_and_use, (5, 3))
    
    # Trace logic:
    # 1. `a > b` triggers trace [2, 1] (a is depth 2, b is depth 1)
    # 2. `c + a` reads 'a' at depth 2 ('c' generates NO trace because it is a raw boolean)
    assert trace == [2, 1, 2]
    
    # Result is 6 (True + 5)
    assert result == 6

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
