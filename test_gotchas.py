#!/usr/bin/env python3
"""
Python gotchas: Because we implement ByteDMD in Python by wrapping Python objects,
our Python framework deviates from idealized description in the README
for certain cases, illustrated by tests below.

Future release of ByteDMD metric may fix this to be more faithful to the README.md
"""
from bytedmd import traced_eval


def test_gotcha_constant_ops():
    """
    Limitation of Python model: constants are not tracked.
    Reads charged: a * 10 (a at arg depth 1) + 10 - tmp (tmp at geom
    depth 1) + output epilogue (final result at geom depth 1).
    """
    def f(a):
        return 10 - a * 10

    trace, _ = traced_eval(f, (5,))
    assert trace == [1, 1, 1]


def test_gotcha_pure_memory_movement_is_free():
    """
    Limitation of Python model: pure list index without computation does not trigger
    math magic methods, hence generating no read trace for the body.
    The output epilogue, however, still reads every returned cell once
    — and for transpose(A) those are the four pristine arg cells of
    A, priced at their static arg depths 1, 3, 2, 4 (transposed order).
    """
    def transpose(A):
        n = len(A)
        return [[A[j][i] for j in range(n)] for i in range(n)]

    A = [[1, 2], [3, 4]]
    trace, result = traced_eval(transpose, (A,))

    assert trace == [1, 3, 2, 4]
    assert result == [[1, 3], [2, 4]]


def test_short_circuit_gotcha():
    """
    Python short-circuit means only one operand may be traced.

    `a and b` with a=0: a.__bool__() reads a (promotes it to geom).
    Since the expression short-circuits to a, b is never touched.
    Output epilogue reads the promoted a at geom depth 1.

    `a or b` with a=0: a.__bool__() reads a; expression short-circuits
    to b. b was never promoted, so the epilogue prices it at its
    static arg depth 2.
    """
    def logical_and(a, b):
        return a and b

    trace, result = traced_eval(logical_and, (0, 5))
    assert trace == [1, 1]
    assert result == 0

    trace, result = traced_eval(lambda a, b: a or b, (0, 5))
    assert trace == [1, 2]
    assert result == 5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
