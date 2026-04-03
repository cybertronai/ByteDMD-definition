#!/usr/bin/env python3
import numpy as np
from bytedmd import bytedmd, traced_eval


def my_add(a, b, c, d):
    return b + c


def test_my_add_int8():
    a, b, c, d = np.int8(1), np.int8(2), np.int8(3), np.int8(4)
    cost, result = bytedmd(my_add, (a, b, c, d))
    assert cost == 4
    assert result == np.int8(5)


# Example of using a mixture of one byte and two byte data types.
def test_my_add_hybrid_trace():
    a, b, c, d = np.int8(1), np.int16(2), np.int16(3), np.int8(4)
    trace, result = traced_eval(my_add, (a, b, c, d))
    assert trace == [5, 4, 3, 2]
    assert result == np.int16(5)


def my_composite_func(a, b, c, d):
    e = b + c
    f = a + d
    return e > f


def test_my_composite_func():
    a, b, c, d = np.int8(1), np.int16(2), np.int16(3), np.int8(4)
    trace, result = traced_eval(my_composite_func, (a, b, c, d))
    assert trace == [5, 4, 3, 2, 8, 7, 5, 4, 1]
    cost, result = bytedmd(my_composite_func, (a, b, c, d))
    assert cost == 21


# --- For-loop array-based functions (runtime tracing) ---

def matvec4(A, x):
    """4x4 matrix-vector multiply y = A @ x."""
    n = len(x)
    y = [None] * n
    for i in range(n):
        s = A[i][0] * x[0]
        for j in range(1, n):
            s = s + A[i][j] * x[j]
        y[i] = s
    return y


def vecmat4(A, x):
    """4x4 vector-matrix multiply y = x^T @ A."""
    n = len(x)
    y = [None] * n
    for j in range(n):
        s = x[0] * A[0][j]
        for i in range(1, n):
            s = s + x[i] * A[i][j]
        y[j] = s
    return y


def matmul4(A, B):
    """4x4 matrix multiply C = A @ B, naive i-j-k loop order."""
    n = len(A)
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = A[i][0] * B[0][j]
            for k in range(1, n):
                s = s + A[i][k] * B[k][j]
            C[i][j] = s
    return C


def matmul4_tiled(A, B):
    """4x4 matrix multiply C = A @ B, tiled with 2x2 blocks."""
    n = len(A)
    t = 2
    C = [[None] * n for _ in range(n)]
    for bi in range(0, n, t):
        for bj in range(0, n, t):
            for bk in range(0, n, t):
                for i in range(bi, bi + t):
                    for j in range(bj, bj + t):
                        for k in range(bk, bk + t):
                            if C[i][j] is None:
                                C[i][j] = A[i][k] * B[k][j]
                            else:
                                C[i][j] = C[i][j] + A[i][k] * B[k][j]
    return C


def test_matmul4():
    A = np.ones((4, 4), dtype=np.int8)
    B = np.ones((4, 4), dtype=np.int8)
    cost, result = bytedmd(matmul4, (A, B))
    assert cost == 948


def test_matmul4_tiled():
    A = np.ones((4, 4), dtype=np.int8)
    B = np.ones((4, 4), dtype=np.int8)
    cost, result = bytedmd(matmul4_tiled, (A, B))
    assert cost == 947


def test_matvec4():
    A = np.ones((4, 4), dtype=np.int8)
    x = np.ones(4, dtype=np.int8)
    cost, result = bytedmd(matvec4, (A, x))
    assert cost == 194


def test_vecmat4():
    A = np.ones((4, 4), dtype=np.int8)
    x = np.ones(4, dtype=np.int8)
    cost, result = bytedmd(vecmat4, (A, x))
    assert cost == 191


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
