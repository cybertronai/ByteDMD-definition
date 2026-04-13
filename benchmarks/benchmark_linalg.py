#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""Measure ByteDMD costs for linear algebra algorithms across multiple sizes.

Prints one table per method with ByteDMD costs at N = 2, 4, 8, 16.

Run directly (uv resolves deps automatically, no venv activation needed):

    ./benchmarks/benchmark_linalg.py
    uv run benchmarks/benchmark_linalg.py
"""
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
    # Mathematica's exact left-associative parsing: ((M1 - M2) + M3) + M6.
    # This reads M2 while it's still hot at the top of the LRU stack (M2
    # was just used by C21 above), avoiding the deeper re-read that the
    # alternative (((M1+M3)-M2)+M6) grouping would trigger.
    C22 = _add(_add(_sub(M1, M2), M3), M6)
    return _join(C11, C12, C21, C22)

def matmul_strassen(A, B, leaf=1):
    n = _check_square_same_size(A, B)
    if n & (n - 1):
        raise ValueError("Strassen requires power-of-two size")
    return _matmul_strassen(A, B, leaf)


# --- Vanilla recursive matmul (8-way divide-and-conquer, not Strassen) ---

def _matmul_vanilla_recursive(A, B):
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)
    C11 = _add(_matmul_vanilla_recursive(A11, B11),
               _matmul_vanilla_recursive(A12, B21))
    C12 = _add(_matmul_vanilla_recursive(A11, B12),
               _matmul_vanilla_recursive(A12, B22))
    C21 = _add(_matmul_vanilla_recursive(A21, B11),
               _matmul_vanilla_recursive(A22, B21))
    C22 = _add(_matmul_vanilla_recursive(A21, B12),
               _matmul_vanilla_recursive(A22, B22))
    return _join(C11, C12, C21, C22)


def matmul_vanilla_recursive(A, B):
    """Vanilla 8-way recursive matmul with temporaries, leaf=1.
    Same FLOP count as naive, but different data movement pattern."""
    n = len(A)
    if n & (n - 1):
        raise ValueError("Vanilla recursive matmul requires power-of-two size")
    return _matmul_vanilla_recursive(A, B)


# --- Self-attention (inspired by Noam Shazeer's shape-suffix example) ---
#
# Single-layer, single-head causal-free self-attention on an N-token
# sequence with d_model = d_k = N. Shapes:
#   X: [L, D]         input activations  (L=N, D=N)
#   Wq, Wk, Wv: [D, K]  projection weights (K=N)
#   Wo: [K, D]          output projection
# Returns out: [L, D].
#
# layer_norm and softmax are replaced by row-normalisation (shift-then-
# divide-by-sum) so the tracer sees the full data-movement pattern of
# real attention without needing exp/rsqrt (which escape tracking).

def self_attention(X, Wq, Wk, Wv, Wo):
    L = len(X)
    D = len(X[0])
    K = len(Wq[0])

    def matmul_LxD_DxK(A, W):
        # [L,D] @ [D,K] -> [L,K]
        out = [[None] * K for _ in range(L)]
        for l in range(L):
            for k in range(K):
                s = A[l][0] * W[0][k]
                for d in range(1, D):
                    s = s + A[l][d] * W[d][k]
                out[l][k] = s
        return out

    # Q, Kmat, V = X @ Wq, X @ Wk, X @ Wv
    Q = matmul_LxD_DxK(X, Wq)
    Kmat = matmul_LxD_DxK(X, Wk)
    V = matmul_LxD_DxK(X, Wv)

    # logits[i][j] = sum_k Q[i][k] * Kmat[j][k]  (Q @ K^T)
    inv_sqrt_k = K ** -0.5
    logits = [[None] * L for _ in range(L)]
    for i in range(L):
        for j in range(L):
            s = Q[i][0] * Kmat[j][0]
            for k in range(1, K):
                s = s + Q[i][k] * Kmat[j][k]
            logits[i][j] = s * inv_sqrt_k

    # Softmax replacement: shift each logit by +1 then row-normalise.
    for i in range(L):
        for j in range(L):
            logits[i][j] = logits[i][j] + 1.0
    weights = [[None] * L for _ in range(L)]
    for i in range(L):
        total = logits[i][0]
        for j in range(1, L):
            total = total + logits[i][j]
        for j in range(L):
            weights[i][j] = logits[i][j] / total

    # wtd[i][k] = sum_j weights[i][j] * V[j][k]   (weights @ V)
    wtd = [[None] * K for _ in range(L)]
    for i in range(L):
        for k in range(K):
            s = weights[i][0] * V[0][k]
            for j in range(1, L):
                s = s + weights[i][j] * V[j][k]
            wtd[i][k] = s

    # out = wtd @ Wo, shape [L, D]
    out = [[None] * D for _ in range(L)]
    for l in range(L):
        for d in range(D):
            s = wtd[l][0] * Wo[0][d]
            for k in range(1, K):
                s = s + wtd[l][k] * Wo[k][d]
            out[l][d] = s
    return out


# --- Flash Attention 2 (inspired by Noam Shazeer shape-suffix example) ---
#
# Same signature as self_attention: (X, Wq, Wk, Wv, Wo) with L=D=K=N and
# single batch / single head. The difference is that the attention kernel
# itself is block-tiled with Br = Bc = N/2 (two blocks per dimension at
# every size), and the softmax is computed online:
#   - Q tile kept hot in the inner loop
#   - inner loop sweeps K and V tiles
#   - running output accumulator + running row-sum
#   - final normalization after the inner loop finishes
#
# The attention matrix [L,L] is never materialized; only [Br,Bc] blocks
# are materialised, so V elements stay close to MRU instead of being
# buried under L*L intermediates.
#
# Causal masking is omitted (all-ones inputs don't trigger it anyway) and
# the numerical-stability max tracking is skipped — both would require
# comparison/exp operations that escape the tracer. The resulting trace
# faithfully captures Flash Attention's data-movement pattern.
#
# Implementation kept for reference/re-enabling via METHODS, but not
# part of the default output table.

def flash_attention(X, Wq, Wk, Wv, Wo):
    L = len(X)
    D = len(X[0])
    K = len(Wq[0])
    Br = max(1, L // 2)
    Bc = max(1, L // 2)

    def matmul_LxD_DxK(A, W):
        out = [[None] * K for _ in range(L)]
        for l in range(L):
            for k in range(K):
                s = A[l][0] * W[0][k]
                for d in range(1, D):
                    s = s + A[l][d] * W[d][k]
                out[l][k] = s
        return out

    # Q, Km, V projections (same as standard self-attention)
    Q = matmul_LxD_DxK(X, Wq)
    Km = matmul_LxD_DxK(X, Wk)
    V = matmul_LxD_DxK(X, Wv)

    inv_sqrt_k = K ** -0.5

    # Global output accumulator (filled block-by-block)
    O = [[None] * K for _ in range(L)]

    # Outer loop: sweep query blocks
    for bi in range(0, L, Br):
        bi_end = bi + Br if bi + Br <= L else L
        R = bi_end - bi

        # Per-row running denominator for this query block
        row_sum = [None] * R
        # Per-row running output [R, K]
        O_block = [[None] * K for _ in range(R)]
        first_block = True

        # Inner loop: sweep key/value blocks (for each query tile)
        for bj in range(0, L, Bc):
            bj_end = bj + Bc if bj + Bc <= L else L
            C = bj_end - bj

            # Compute block logits [R, C] = Q[bi:bi+R] @ Km[bj:bj+C]^T * scale
            logits = [[None] * C for _ in range(R)]
            for r in range(R):
                for c in range(C):
                    s = Q[bi + r][0] * Km[bj + c][0]
                    for k in range(1, K):
                        s = s + Q[bi + r][k] * Km[bj + c][k]
                    logits[r][c] = s * inv_sqrt_k

            # Shift to keep logits positive — placeholder for softmax max
            for r in range(R):
                for c in range(C):
                    logits[r][c] = logits[r][c] + 1.0

            # Block-wise row sums (partial denominator)
            block_sum = [None] * R
            for r in range(R):
                s = logits[r][0]
                for c in range(1, C):
                    s = s + logits[r][c]
                block_sum[r] = s

            # Block-wise weighted values: logits @ V[bj:bj+C], shape [R, K]
            wtd = [[None] * K for _ in range(R)]
            for r in range(R):
                for k in range(K):
                    s = logits[r][0] * V[bj + 0][k]
                    for c in range(1, C):
                        s = s + logits[r][c] * V[bj + c][k]
                    wtd[r][k] = s

            # Online merge into running accumulator
            if first_block:
                for r in range(R):
                    row_sum[r] = block_sum[r]
                    for k in range(K):
                        O_block[r][k] = wtd[r][k]
                first_block = False
            else:
                for r in range(R):
                    row_sum[r] = row_sum[r] + block_sum[r]
                    for k in range(K):
                        O_block[r][k] = O_block[r][k] + wtd[r][k]

        # Final per-row normalization for this query block
        for r in range(R):
            for k in range(K):
                O[bi + r][k] = O_block[r][k] / row_sum[r]

    # Output projection: [L, K] @ [K, D] -> [L, D]
    out = [[None] * D for _ in range(L)]
    for l in range(L):
        for d in range(D):
            s = O[l][0] * Wo[0][d]
            for k in range(1, K):
                s = s + O[l][k] * Wo[k][d]
            out[l][d] = s
    return out


# --- Measurements ---

SIZES = [2, 4, 8, 16]


# (column header, kind, function)
#   kind = 'matvec'    runs fn(A, x)
#   kind = 'matmul'    runs fn(A, B)
#   kind = 'attention' runs fn(X, Wq, Wk, Wv, Wo)  (1 layer, 1 head, L=D=K=N)
#
# self_attention and flash_attention are defined above and can be
# re-enabled by appending them to this list, but are not part of the
# default output.
METHODS = [
    ("matvec\n(y=A@x)",         'matvec',    matvec4),
    ("vecmat\n(y=xᵀ@A)",         'matvec',    vecmat4),
    ("naive matmul\n(i-j-k)",      'matmul',    matmul4),
    ("vanilla rec\n(8-way D&C)",   'matmul',    matmul_vanilla_recursive),
    ("Strassen\n(7-way D&C)",      'matmul',    matmul_strassen),
]


def _cost(kind, fn, n):
    if kind == 'matvec':
        return bytedmd(fn, (np.ones((n, n)), np.ones(n)))
    if kind == 'attention':
        X  = np.ones((n, n))
        Wq = np.ones((n, n))
        Wk = np.ones((n, n))
        Wv = np.ones((n, n))
        Wo = np.ones((n, n))
        return bytedmd(fn, (X, Wq, Wk, Wv, Wo))
    return bytedmd(fn, (np.ones((n, n)), np.ones((n, n))))


if __name__ == '__main__':
    print(f"# ByteDMD cost by method and size\n")

    # Compute the table data first
    rows = [[_cost(kind, fn, n) for (_, kind, fn) in METHODS] for n in SIZES]

    # Each header has two stacked lines — split them for multi-row markdown-ish output
    headers_top    = [h.split('\n')[0] for (h, _, _) in METHODS]
    headers_bottom = [h.split('\n')[1] for (h, _, _) in METHODS]

    # Column width = max of header and widest number in that column
    widths = []
    for col, (top, bot) in enumerate(zip(headers_top, headers_bottom)):
        w = max(len(top), len(bot), max(len(f"{r[col]:,}") for r in rows))
        widths.append(w)
    n_col_w = max(len("N"), max(len(str(s)) for s in SIZES))

    def fmt_row(cells, first):
        parts = [f"{first:>{n_col_w}}"] + [f"{c:>{w}}" for c, w in zip(cells, widths)]
        return "| " + " | ".join(parts) + " |"

    print(fmt_row(headers_top, ""))
    print(fmt_row(headers_bottom, "N"))
    sep_cells = ["-" * w for w in widths]
    print("|" + "-" * (n_col_w + 2) + "|" + "|".join("-" * (w + 2) for w in widths) + "|")

    for n, row in zip(SIZES, rows):
        print(fmt_row([f"{c:,}" for c in row], n))
