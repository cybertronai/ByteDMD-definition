import math
import numpy as np
import pytest
from bytedmd import measureDMD, measureDMDSquared, byteReadTrace


def myAdd(a, b, c, d):
    return b + c


def test_myAdd_int8():
    a, b, c, d = np.int8(1), np.int8(2), np.int8(3), np.int8(4)
    cost, result = measureDMD(myAdd, a, b, c, d)
    assert cost == math.sqrt(3) + math.sqrt(2)
    assert result == np.int8(5)


def test_myAdd_int8_dmd2():
    a, b, c, d = np.int8(1), np.int8(2), np.int8(3), np.int8(4)
    cost, result = measureDMDSquared(myAdd, a, b, c, d)
    assert cost == 3 + 2


def test_myAdd_hybrid_trace():
    a, b, c, d = np.int8(1), np.int16(2), np.int16(3), np.int8(4)
    trace, result = byteReadTrace(myAdd, a, b, c, d)
    assert trace == [5, 4, 3, 2]
    assert result == np.int16(5)


def myFunc2(a, b, c, d):
    e = b + c
    f = a + d
    return e > f


def test_myFunc2():
    a, b, c, d = np.int8(1), np.int16(2), np.int16(3), np.int8(4)
    trace, result = byteReadTrace(myFunc2, a, b, c, d)
    assert trace == [5, 4, 3, 2, 8, 7, 5, 4, 1]
    cost, result = measureDMD(myFunc2, a, b, c, d)
    expected = (math.sqrt(5) + math.sqrt(4) + math.sqrt(3) + math.sqrt(2)
                + math.sqrt(8) + math.sqrt(7)
                + math.sqrt(5) + math.sqrt(4) + math.sqrt(1))
    assert cost == expected


def test_myFunc2_dmd2():
    a, b, c, d = np.int8(1), np.int16(2), np.int16(3), np.int8(4)
    cost, result = measureDMDSquared(myFunc2, a, b, c, d)
    assert cost == 5 + 4 + 3 + 2 + 8 + 7 + 5 + 4 + 1


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


def test_matmul4_dmd2():
    A = np.ones((4, 4), dtype=np.int8)
    B = np.ones((4, 4), dtype=np.int8)
    cost, result = measureDMDSquared(matmul4, A, B)
    assert cost == 5166


def test_matvec4():
    A = np.ones((4, 4), dtype=np.int8)
    x = np.ones(4, dtype=np.int8)
    cost, result = measureDMD(matvec4, A, x)
    assert cost == pytest.approx(183.841928, abs=1e-4)


def test_matvec4_dmd2():
    A = np.ones((4, 4), dtype=np.int8)
    x = np.ones(4, dtype=np.int8)
    cost, result = measureDMDSquared(matvec4, A, x)
    assert cost == 792


def test_vecmat4():
    A = np.ones((4, 4), dtype=np.int8)
    x = np.ones(4, dtype=np.int8)
    cost, result = measureDMD(vecmat4, A, x)
    assert cost == pytest.approx(181.741063, abs=1e-4)


def test_vecmat4_dmd2():
    A = np.ones((4, 4), dtype=np.int8)
    x = np.ones(4, dtype=np.int8)
    cost, result = measureDMDSquared(vecmat4, A, x)
    assert cost == 768


def test_matmul4_tiled_dmd2():
    A = np.ones((4, 4), dtype=np.int8)
    B = np.ones((4, 4), dtype=np.int8)
    cost, result = measureDMDSquared(matmul4_tiled, A, B)
    assert cost == 5062


def test_matmul4_tiled_cheaper_than_naive():
    A = np.ones((4, 4), dtype=np.int8)
    B = np.ones((4, 4), dtype=np.int8)
    cost_tiled, _ = measureDMDSquared(matmul4_tiled, A, B)
    cost_naive, _ = measureDMDSquared(matmul4, A, B)
    assert cost_tiled < cost_naive
