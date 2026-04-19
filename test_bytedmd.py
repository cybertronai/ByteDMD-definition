#!/usr/bin/env uv run python
import numpy as np
from bytedmd import bytedmd, traced_eval, trace_to_bytedmd


def my_add(a, b, c):
    return (a + b) + c


def test_my_add():
    cost = bytedmd(my_add, (1, 2, 3))
    assert cost == 7

    # Two-stack model: arg stack [a@1, b@2, c@3], geom empty.
    # a+b: reads a@arg=1, b@arg=2 (cost 1+2=3); promote & die; push t.
    # t+c: reads t@geom=1, c@arg=3 (cost 1+2=3).
    # Output epilogue: read the final sum at geom=1 (cost 1). Total: 7.
    trace, _ = traced_eval(my_add, (1, 2, 3))
    assert trace == [1, 2, 1, 3, 1]

    assert trace_to_bytedmd(trace, bytes_per_element=1) == 7
    assert trace_to_bytedmd(trace, bytes_per_element=2) == 19

    assert bytedmd(my_add, (1, 2, 3), bytes_per_element=2) == 19


def my_composite_func(a, b, c, d):
    e = b + c
    f = a + d
    return e > f


def test_repeated_operand_is_charged_twice():
    """README says a+a should charge two reads against the same pre-instruction stack."""
    # 2 reads + 1 epilogue read of the output sum = [1, 1, 1].
    trace, _ = traced_eval(lambda a: a + a, (5,))
    assert trace == [1, 1, 1]


def test_my_composite_func():
    # Arg stack [a@1, b@2, c@3, d@4].
    # b+c: arg 2+3. a+d: arg 1+4. cmp: geom 2+1. epilogue: geom 1.
    trace, result = traced_eval(my_composite_func, (1, 2, 3, 4))
    assert trace == [2, 3, 1, 4, 2, 1, 1]
    cost = bytedmd(my_composite_func, (1, 2, 3, 4))
    assert cost == 11


def test_dot_product():
    def dot(a, b):
        return a[0]*b[0] + a[1]*b[1]

    a, b = [0, 1], [2, 3]
    trace, result = traced_eval(dot, (a, b))

    # Arg stack [a[0]@1, a[1]@2, b[0]@3, b[1]@4].
    # a[0]*b[0]: arg 1+3. a[1]*b[1]: arg 2+4. p0+p1: geom 2+1.
    # Output epilogue: one more read at geom=1.
    assert trace == [1, 3, 2, 4, 2, 1, 1]
    assert result == 3
    assert bytedmd(dot, (a, b)) == 11


def test_branching_and_comparisons_trace():
    def my_relu(a):
        if a > 0:
            return a * 2
        return a

    # Branch taken: a > 0 reads a, __bool__ reads result, then a * 2
    # reads a. Output epilogue reads the final product at geom=1.
    trace_pos, _ = traced_eval(my_relu, (5,))
    assert trace_pos == [1, 1, 1, 1]

    # Branch skipped: a > 0 reads a, __bool__ reads result; function
    # returns `a` (now on geom after promotion). Epilogue reads it.
    trace_neg, _ = traced_eval(my_relu, (-5,))
    assert trace_neg == [1, 1, 1]


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
    assert trace == [1, 2, 2, 1, 1, 2, 1]


def test_implicit_boolean_is_traced():
    """
    `if a:` now correctly calls __bool__, recording a read and evaluating truthiness properly.
    Both implicit and explicit branches produce the same result for a=0.
    """
    def implicit_branch(a):
        if a:
            return a + 10
        return a

    def explicit_branch(a):
        if a != 0:
            return a + 10
        return a

    trace_implicit, result_implicit = traced_eval(implicit_branch, (0,))
    # __bool__ reads a, promotes it; else branch returns a; epilogue
    # reads the (now-promoted) a from geom.
    assert trace_implicit == [1, 1]
    assert result_implicit == 0

    trace_explicit, result_explicit = traced_eval(explicit_branch, (0,))
    # a != 0 reads a, __bool__ reads the comparison result, else
    # branch returns a; epilogue reads a from geom.
    assert trace_explicit == [1, 1, 1]
    assert result_explicit == 0


def test_index_protocol_works():
    trace, result = traced_eval(lambda n: [i for i in range(n)], (3,))
    assert trace == [1]
    assert result == [0, 1, 2]

    trace, result = traced_eval(lambda xs, i: xs[i], ([10, 20, 30], 1))
    # Arg stack: xs[0]@1, xs[1]@2, xs[2]@3, i@4. Reading i for indexing
    # costs 4; returned value xs[1] is a pristine arg, so the epilogue
    # prices it at its static arg depth 2.
    assert trace == [4, 2]
    assert result == 20


def test_not_is_traced():
    """
    `not a` now invokes __bool__, generating a read trace and returning the correct result.
    """
    trace, result = traced_eval(lambda a: not a, (0,))
    assert trace == [1]
    assert result is True


def _matvec(A, x):
    n = len(x)
    y = [None] * n
    for i in range(n):
        s = A[i][0] * x[0]
        for j in range(1, n):
            s = s + A[i][j] * x[j]
        y[i] = s
    return y


def _vecmat(A, x):
    n = len(x)
    y = [None] * n
    for j in range(n):
        s = x[0] * A[0][j]
        for i in range(1, n):
            s = s + x[i] * A[i][j]
        y[j] = s
    return y


def _ceil_sqrt(x):
    """ceil(sqrt(x)) via integer arithmetic."""
    import math
    return math.isqrt(x - 1) + 1 if x > 0 else 0


def test_matvec_costs():
    """Under the two-stack model, matvec and vecmat are once again
    symmetric: both algorithms touch the same (A cell, x cell) pairs
    in the same multiplicity, and the argument stack positions are
    fixed regardless of traversal order — so the total priced trace
    is identical. The output epilogue reads each of the n y[i]
    entries once more, at its final geom depth."""
    expected = {2: 26, 3: 71, 4: 147, 5: 253, 6: 394, 7: 574, 8: 839}
    for n in [2, 3, 4, 5, 6, 7, 8]:
        A = np.ones((n, n))
        x = np.ones(n)
        mv = bytedmd(_matvec, (A, x))
        vm = bytedmd(_vecmat, (A, x))
        assert mv == expected[n], f"matvec N={n}: got {mv}, expected {expected[n]}"
        assert vm == expected[n], f"vecmat N={n}: got {vm}, expected {expected[n]}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
