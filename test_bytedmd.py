#!/usr/bin/env python3
import numpy as np
from bytedmd import bytedmd, traced_eval, trace_to_cost


def my_add(a, b, c, d):
    return b + c


def test_my_add():
    cost = bytedmd(my_add, (1, 2, 3, 4))
    assert cost == 4

def my_add2(a, b, c):
    return (a + b) + c


def test_my_add2():
    cost = bytedmd(my_add2, (1, 2, 3))
    assert cost == 7


def test_my_add2_16bit():
    cost = bytedmd(my_add2, (1, 2, 3), bytes_per_element=2)
    assert cost == 19

    trace, _ = traced_eval(my_add2, (1, 2, 3))
    # trace counts depth in terms of number of elements, not bytes
    assert trace == [3, 2, 1, 4]
    assert trace_to_cost(trace, bytes_per_element=2) == 19


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
    A = np.ones((4, 4))
    B = np.ones((4, 4))
    cost = bytedmd(matmul4, (A, B))
    assert cost == 948


def test_matmul4_tiled():
    A = np.ones((4, 4))
    B = np.ones((4, 4))
    cost = bytedmd(matmul4_tiled, (A, B))
    assert cost == 947


def test_matvec4():
    A = np.ones((4, 4))
    x = np.ones(4)
    cost = bytedmd(matvec4, (A, x))
    assert cost == 194


def test_vecmat4():
    A = np.ones((4, 4))
    x = np.ones(4)
    cost = bytedmd(vecmat4, (A, x))
    assert cost == 191


# --- Strassen matrix multiplication ---

def _check_square_same_size(A, B):
    n = len(A)
    return n


def _split(M):
    """Split a matrix into four quadrants."""
    n = len(M)
    h = n // 2
    M11 = [[M[i][j] for j in range(h)] for i in range(h)]
    M12 = [[M[i][j] for j in range(h, n)] for i in range(h)]
    M21 = [[M[i][j] for j in range(h)] for i in range(h, n)]
    M22 = [[M[i][j] for j in range(h, n)] for i in range(h, n)]
    return M11, M12, M21, M22


def _join(C11, C12, C21, C22):
    """Join four quadrants into a single matrix."""
    h = len(C11)
    n = 2 * h
    return [[C11[i][j] if j < h else C12[i][j - h] for j in range(n)] for i in range(h)] + \
           [[C21[i][j] if j < h else C22[i][j - h] for j in range(n)] for i in range(h)]


def _add(A, B):
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def _sub(A, B):
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


def _matmul_ikj(A, B):
    """Naive matrix multiply with i-k-j loop order."""
    n = len(A)
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = A[i][0] * B[0][j]
            for k in range(1, n):
                s = s + A[i][k] * B[k][j]
            C[i][j] = s
    return C


def _matmul_strassen(A, B, leaf):
    n = len(A)
    if n <= leaf:
        return _matmul_ikj(A, B)

    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)

    M1 = _matmul_strassen(_add(A11, A22), _add(B11, B22), leaf)
    M2 = _matmul_strassen(_add(A21, A22), B11, leaf)
    M3 = _matmul_strassen(A11, _sub(B12, B22), leaf)
    M4 = _matmul_strassen(A22, _sub(B21, B11), leaf)
    M5 = _matmul_strassen(_add(A11, A12), B22, leaf)
    M6 = _matmul_strassen(_sub(A21, A11), _add(B11, B12), leaf)
    M7 = _matmul_strassen(_sub(A12, A22), _add(B21, B22), leaf)

    C11 = _add(_sub(_add(M1, M4), M5), M7)
    C12 = _add(M3, M5)
    C21 = _add(M2, M4)
    C22 = _add(_sub(_add(M1, M3), M2), M6)

    return _join(C11, C12, C21, C22)


def matmul_strassen(A, B, leaf=4):
    n = _check_square_same_size(A, B)
    if n & (n - 1):
        raise ValueError("Strassen requires power-of-two size")
    return _matmul_strassen(A, B, leaf)


def test_matmul4_strassen():
    A = np.ones((4, 4))
    B = np.ones((4, 4))
    cost = bytedmd(matmul_strassen, (A, B))
    assert cost == 948  # leaf=4, same as naive for 4x4


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
