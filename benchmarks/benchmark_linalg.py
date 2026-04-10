#!/usr/bin/env python3
"""Measure ByteDMD costs for linear algebra algorithms on 4x4 matrices."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from bytedmd import bytedmd


# --- Matrix-vector and vector-matrix ---

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


# --- Matrix multiply variants ---

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


def _zeros(n):
    return [[None] * n for _ in range(n)]


def matmul4_ikj(A, B):
    """4x4 matrix multiply C = A @ B, i-k-j loop order."""
    n = len(A)
    C = _zeros(n)
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            aik = Ai[k]
            Bk = B[k]
            for j in range(n):
                if Ci[j] is None:
                    Ci[j] = aik * Bk[j]
                else:
                    Ci[j] = Ci[j] + aik * Bk[j]
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


def matmul4_snake_j(A, B):
    n = len(A)
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        js = range(n) if i % 2 == 0 else range(n - 1, -1, -1)
        for j in js:
            s = A[i][0] * B[0][j]
            for k in range(1, n):
                s = s + A[i][k] * B[k][j]
            C[i][j] = s
    return C


def matmul4_tsp(A, B):
    n = 4
    C = [[None] * n for _ in range(n)]
    optimal_schedule = [
        (0, 2), (0, 1), (0, 0), (0, 3),
        (1, 3), (1, 2), (1, 1), (1, 0),
        (2, 0), (2, 2), (2, 1), (2, 3),
        (3, 3), (3, 2), (3, 1), (3, 0)
    ]
    for i, j in optimal_schedule:
        s = A[i][0] * B[0][j]
        for k in range(1, n):
            s = s + A[i][k] * B[k][j]
        C[i][j] = s
    return C


# --- Winograd ---

def _matmul_2x2_winograd(A, B):
    a11, a12 = A[0]
    a21, a22 = A[1]
    b11, b12 = B[0]
    b21, b22 = B[1]
    v1 = b22 - b12
    v2 = v1 + b11
    v3 = v2 - b21
    v4 = b12 - b11
    u1 = a11 - a21
    u2 = a21 + a22
    u3 = u1 - a22
    u4 = u3 + a12
    p1 = a11 * b11
    p2 = a12 * b21
    p3 = a22 * v3
    p4 = u1 * v1
    p5 = u2 * v4
    p6 = u4 * b22
    p7 = u3 * v2
    t1 = p1 - p7
    t2 = t1 + p5
    return [
        [p1 + p2, t2 + p6],
        [t1 - p3 + p4, t2 + p4],
    ]


def _join_2x2_blocks_into_4x4(C11, C12, C21, C22):
    out = []
    for i in range(2):
        out.append(C11[i] + C12[i])
    for i in range(2):
        out.append(C21[i] + C22[i])
    return out


def _split_4x4_into_2x2_blocks(M):
    return (
        [row[:2] for row in M[:2]],
        [row[2:] for row in M[:2]],
        [row[:2] for row in M[2:]],
        [row[2:] for row in M[2:]],
    )


def matmul_4x4_winograd(A, B):
    A11, A12, A21, A22 = _split_4x4_into_2x2_blocks(A)
    B11, B12, B21, B22 = _split_4x4_into_2x2_blocks(B)
    V1 = _sub(B22, B12)
    V2 = _add(V1, B11)
    V3 = _sub(V2, B21)
    V4 = _sub(B12, B11)
    U1 = _sub(A11, A21)
    U2 = _add(A21, A22)
    U3 = _sub(U1, A22)
    U4 = _add(U3, A12)
    P1 = _matmul_2x2_winograd(A11, B11)
    P2 = _matmul_2x2_winograd(A12, B21)
    P3 = _matmul_2x2_winograd(A22, V3)
    P4 = _matmul_2x2_winograd(U1, V1)
    P5 = _matmul_2x2_winograd(U2, V4)
    P6 = _matmul_2x2_winograd(U4, B22)
    P7 = _matmul_2x2_winograd(U3, V2)
    T1 = _sub(P1, P7)
    T2 = _add(T1, P5)
    C11 = _add(P1, P2)
    C12 = _add(T2, P6)
    C21 = _add(_sub(T1, P3), P4)
    C22 = _add(T2, P4)
    return _join_2x2_blocks_into_4x4(C11, C12, C21, C22)


# --- Strassen ---

def _check_square_same_size(A, B):
    return len(A)

def _split(M):
    n = len(M)
    h = n // 2
    M11 = [[M[i][j] for j in range(h)] for i in range(h)]
    M12 = [[M[i][j] for j in range(h, n)] for i in range(h)]
    M21 = [[M[i][j] for j in range(h)] for i in range(h, n)]
    M22 = [[M[i][j] for j in range(h, n)] for i in range(h, n)]
    return M11, M12, M21, M22

def _join(C11, C12, C21, C22):
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

def matmul_strassen(A, B, leaf=1):
    n = _check_square_same_size(A, B)
    if n & (n - 1):
        raise ValueError("Strassen requires power-of-two size")
    return _matmul_strassen(A, B, leaf)


# --- Measurements ---

def measure(name, operation, func, args, expected):
    cost = bytedmd(func, args)
    assert cost == expected, f"{name}: expected {expected}, got {cost}"
    return name, operation, cost


if __name__ == '__main__':
    A = np.ones((4, 4))
    B = np.ones((4, 4))
    x = np.ones(4)

    results = [
        measure("matvec (i-j)", "y = A @ x", matvec4, (A, x), 187),
        measure("vecmat (j-i)", "y = x^T @ A", vecmat4, (A, x), 181),
        measure("matmul (i-j-k)", "C = A @ B", matmul4, (A, B), 830),
        measure("matmul (i-k-j)", "C = A @ B", matmul4_ikj, (A, B), 865),
        measure("matmul (snake-j)", "C = A @ B", matmul4_snake_j, (A, B), 797),
        measure("matmul (2x2 tiled)", "C = A @ B", matmul4_tiled, (A, B), 835),
        measure("matmul (TSP)", "C = A @ B", matmul4_tsp, (A, B), 803),
        measure("Strassen (leaf=1)", "C = A @ B", matmul_strassen, (A, B), 2013),
        measure("Winograd", "C = A @ B", matmul_4x4_winograd, (A, B), 1885),
    ]

    print(f"{'Algorithm':<25} {'Operation':<15} {'ByteDMD Cost':>12}")
    print("-" * 55)
    for name, op, cost in results:
        print(f"{name:<25} {op:<15} {cost:>12}")
