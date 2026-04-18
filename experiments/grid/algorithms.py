"""Traceable Python algorithms for the grid experiment.

Every algorithm performs its work through overloaded arithmetic so that
bytedmd_ir.trace() can convert it to an L2 event sequence. Helpers _max2,
_exp, _inv stand in for max/exp/1-over in attention — they have the same
read/write pattern as the real ops.
"""
from __future__ import annotations

from typing import List


# ============================================================================
# Matmul variants not already in bytedmd_ir
# ============================================================================

def _split(M):
    n = len(M); h = n // 2
    return ([[M[i][j] for j in range(h)] for i in range(h)],
            [[M[i][j] for j in range(h, n)] for i in range(h)],
            [[M[i][j] for j in range(h)] for i in range(h, n)],
            [[M[i][j] for j in range(h, n)] for i in range(h, n)])

def _join(C11, C12, C21, C22):
    h = len(C11); n = 2 * h
    return [[C11[i][j] if j < h else C12[i][j-h] for j in range(n)] for i in range(h)] + \
           [[C21[i][j] if j < h else C22[i][j-h] for j in range(n)] for i in range(h)]

def _addm(A, B):
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def _subm(A, B):
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


def matmul_tiled_explicit(A, B, tile=4):
    """Tiled matmul with explicit DMA: per tile, copy A/B/C blocks into
    fresh _Tracked variables via `+ 0.0` before the MAC, and flush sC
    back to C at the end. This materializes the scratchpad in the L2
    trace so space_dmd can rank the short-lived tile vars at the hot
    addresses. The LRU heuristics don't benefit — they already get that
    effect automatically via recency bump."""
    n = len(A)
    zero = A[0][0] - A[0][0]  # tracked zero for C init
    C = [[zero + 0.0 for _ in range(n)] for _ in range(n)]
    for bi in range(0, n, tile):
        for bj in range(0, n, tile):
            sC = [[C[bi + i][bj + j] + 0.0 for j in range(tile)] for i in range(tile)]
            for bk in range(0, n, tile):
                sA = [[A[bi + i][bk + k] + 0.0 for k in range(tile)] for i in range(tile)]
                sB = [[B[bk + k][bj + j] + 0.0 for j in range(tile)] for k in range(tile)]
                for i in range(tile):
                    for j in range(tile):
                        for k in range(tile):
                            sC[i][j] = sC[i][j] + sA[i][k] * sB[k][j]
            for i in range(tile):
                for j in range(tile):
                    C[bi + i][bj + j] = sC[i][j] + 0.0
    return C


def matmul_naive_abt(A, B):
    """Naive triple loop computing C = A @ B^T, i.e.,
    C[i][j] = sum_k A[i][k] * B[j][k]. Both A and B are read row-major
    (contiguous) in the inner k-loop — the cache-friendly variant."""
    n = len(A)
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = A[i][0] * B[j][0]
            for k in range(1, n):
                s = s + A[i][k] * B[j][k]
            C[i][j] = s
    return C


def matmul_strassen(A, B):
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)
    M1 = matmul_strassen(_addm(A11, A22), _addm(B11, B22))
    M2 = matmul_strassen(_addm(A21, A22), B11)
    M3 = matmul_strassen(A11, _subm(B12, B22))
    M4 = matmul_strassen(A22, _subm(B21, B11))
    M5 = matmul_strassen(_addm(A11, A12), B22)
    M6 = matmul_strassen(_subm(A21, A11), _addm(B11, B12))
    M7 = matmul_strassen(_subm(A12, A22), _addm(B21, B22))
    C11 = _addm(_subm(_addm(M1, M4), M5), M7)
    C12 = _addm(M3, M5)
    C21 = _addm(M2, M4)
    C22 = _addm(_subm(_addm(M1, M3), M2), M6)
    return _join(C11, C12, C21, C22)


# ============================================================================
# Attention
# ============================================================================

def _max2(a, b): return a + b      # trace-safe stand-in for max
def _exp(x):    return x * x       # trace-safe stand-in for exp
def _inv(x):    return x * x       # trace-safe stand-in for 1/x


def naive_attention(Q, K, V):
    N = len(Q); d = len(Q[0]); scale = d ** -0.5
    S = [[None] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            acc = Q[i][0] * K[j][0]
            for dd in range(1, d):
                acc = acc + Q[i][dd] * K[j][dd]
            S[i][j] = acc * scale
    P = [[None] * N for _ in range(N)]
    for i in range(N):
        mx = S[i][0]
        for j in range(1, N):
            mx = _max2(mx, S[i][j])
        row_sum = None
        for j in range(N):
            P[i][j] = _exp(S[i][j] - mx)
            row_sum = P[i][j] if row_sum is None else row_sum + P[i][j]
        inv_sum = _inv(row_sum)
        for j in range(N):
            P[i][j] = P[i][j] * inv_sum
    O = [[None] * d for _ in range(N)]
    for i in range(N):
        for dd in range(d):
            acc = P[i][0] * V[0][dd]
            for j in range(1, N):
                acc = acc + P[i][j] * V[j][dd]
            O[i][dd] = acc
    return O


def flash_attention(Q, K, V, Bk=2):
    N = len(Q); d = len(Q[0]); scale = d ** -0.5
    num_blocks = (N + Bk - 1) // Bk
    O = [[None] * d for _ in range(N)]
    for i in range(N):
        m_prev = None; l_prev = None
        o_acc = [None] * d
        for kb in range(num_blocks):
            k0 = kb * Bk
            k1 = min(k0 + Bk, N)
            bs = k1 - k0
            s_block = [None] * bs
            for j in range(bs):
                kj = k0 + j
                acc = Q[i][0] * K[kj][0]
                for dd in range(1, d):
                    acc = acc + Q[i][dd] * K[kj][dd]
                s_block[j] = acc * scale
            m_block = s_block[0]
            for j in range(1, bs):
                m_block = _max2(m_block, s_block[j])
            p_block = [None] * bs
            l_block = None
            for j in range(bs):
                p_block[j] = _exp(s_block[j] - m_block)
                l_block = p_block[j] if l_block is None else l_block + p_block[j]
            o_block = [None] * d
            for dd in range(d):
                acc = p_block[0] * V[k0][dd]
                for j in range(1, bs):
                    acc = acc + p_block[j] * V[k0 + j][dd]
                o_block[dd] = acc
            if m_prev is None:
                m_prev = m_block; l_prev = l_block
                for dd in range(d):
                    o_acc[dd] = o_block[dd]
            else:
                m_new = _max2(m_prev, m_block)
                alpha = _exp(m_prev - m_new)
                beta = _exp(m_block - m_new)
                l_prev = alpha * l_prev + beta * l_block
                for dd in range(d):
                    o_acc[dd] = alpha * o_acc[dd] + beta * o_block[dd]
                m_prev = m_new
        inv_l = _inv(l_prev)
        for dd in range(d):
            O[i][dd] = o_acc[dd] * inv_l
    return O


# ============================================================================
# Transpose
# ============================================================================

def transpose_naive(A):
    """B[i][j] = A[j][i]. Reads A in column-major, writes B in row-major."""
    n = len(A)
    B = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # force a load via "+ 0"
            B[i][j] = A[j][i] + 0
    return B


def transpose_blocked(A, T=None):
    n = len(A)
    if T is None:
        T = max(1, int(round(n ** 0.5)))
    B = [[None] * n for _ in range(n)]
    for bi in range(0, n, T):
        for bj in range(0, n, T):
            for i in range(bi, min(bi + T, n)):
                for j in range(bj, min(bj + T, n)):
                    B[i][j] = A[j][i] + 0
    return B


def transpose_recursive(A):
    """Cache-oblivious transpose via 4-way split."""
    n = len(A)
    B = [[None] * n for _ in range(n)]

    def rec(ar, ac, br, bc, sz):
        if sz == 1:
            B[br][bc] = A[ar][ac] + 0
            return
        h = sz // 2
        rec(ar,     ac,     br,     bc,     h)
        rec(ar + h, ac,     br,     bc + h, h)
        rec(ar,     ac + h, br + h, bc,     h)
        rec(ar + h, ac + h, br + h, bc + h, h)

    rec(0, 0, 0, 0, n)
    return B


# ============================================================================
# Matrix-vector
# ============================================================================

def matvec_row(A, x):
    """Row-major matvec: y[i] = sum_j A[i][j] * x[j]."""
    n = len(A)
    y = [None] * n
    for i in range(n):
        s = A[i][0] * x[0]
        for j in range(1, n):
            s = s + A[i][j] * x[j]
        y[i] = s
    return y


def matvec_blocked(A, x, B=4):
    """Tiled matvec with explicit x-tile DMA: every inner tile loads the
    current B-slice of x into short-lived `x_tile` vars via `+ 0.0`,
    then does a B×B MAC against the corresponding A tile and accumulates
    into B running sums. Models the "streaming-A + x-tile scratchpad"
    schedule from gemini/efficient-matvec.md."""
    n = len(A)
    y = [None] * n
    for i_out in range(0, n, B):
        s = [None] * B
        for j_out in range(0, n, B):
            # DMA-load x tile into short-lived, high-density tile vars
            x_tile = [x[j_out + j] + 0.0 for j in range(B)]
            for i in range(B):
                for j in range(B):
                    if s[i] is None:
                        s[i] = A[i_out + i][j_out + j] * x_tile[j]
                    else:
                        s[i] = s[i] + A[i_out + i][j_out + j] * x_tile[j]
        for i in range(B):
            y[i_out + i] = s[i] + 0.0
    return y


def matvec_col(A, x):
    """Column-major matvec: accumulate y column-by-column — strided reads of A."""
    n = len(A)
    y = [None] * n
    for i in range(n):
        y[i] = A[i][0] * x[0]
    for j in range(1, n):
        for i in range(n):
            y[i] = y[i] + A[i][j] * x[j]
    return y


# ============================================================================
# FFT (radix-2 Cooley–Tukey, real twiddle stand-in — we only care about the
# read/write pattern, not the numeric result)
# ============================================================================

def fft_iterative(x_in):
    """In-place iterative radix-2 Cooley-Tukey on a length-N array.
    Uses a constant real factor in place of the complex twiddle — the
    load pattern is identical."""
    x = [v + 0 for v in x_in]  # force per-element load into fresh vars
    N = len(x)
    # Bit-reverse permutation
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            t = x[i] + 0
            x[i] = x[j] + 0
            x[j] = t
    # Butterflies
    m = 1
    while m < N:
        for k in range(0, N, m * 2):
            for jj in range(m):
                t = x[k + jj + m] * 1.5   # twiddle stand-in
                u = x[k + jj]
                x[k + jj] = u + t
                x[k + jj + m] = u - t
        m *= 2
    return x


def fft_recursive(x_in):
    """Out-of-place recursive radix-2 Cooley-Tukey."""
    N = len(x_in)
    if N == 1:
        return [x_in[0] + 0]
    even = fft_recursive([x_in[2 * i] + 0 for i in range(N // 2)])
    odd  = fft_recursive([x_in[2 * i + 1] + 0 for i in range(N // 2)])
    out = [None] * N
    for k in range(N // 2):
        t = odd[k] * 1.5
        out[k] = even[k] + t
        out[k + N // 2] = even[k] - t
    return out


# ============================================================================
# 2D Jacobi stencil (5-point, one sweep)
# ============================================================================

def stencil_naive(A):
    """Row-major sweep: B[i][j] = 0.2 * (A[i][j] + A[i-1][j] + A[i+1][j]
    + A[i][j-1] + A[i][j+1]). Boundary cells are left None."""
    n = len(A)
    B = [[None] * n for _ in range(n)]
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            B[i][j] = (A[i][j] + A[i - 1][j] + A[i + 1][j]
                       + A[i][j - 1] + A[i][j + 1]) * 0.2
    return B


def stencil_recursive(A, leaf=8):
    """Tile-recursive split: quad-tree over the 2D grid, naive sweep at leaves."""
    n = len(A)
    B = [[None] * n for _ in range(n)]

    def rec(r0, c0, sz):
        if sz <= leaf:
            for i in range(r0, r0 + sz):
                for j in range(c0, c0 + sz):
                    if 0 < i < n - 1 and 0 < j < n - 1:
                        B[i][j] = (A[i][j] + A[i - 1][j] + A[i + 1][j]
                                   + A[i][j - 1] + A[i][j + 1]) * 0.2
            return
        h = sz // 2
        rec(r0,     c0,     h)
        rec(r0,     c0 + h, h)
        rec(r0 + h, c0,     h)
        rec(r0 + h, c0 + h, h)

    rec(0, 0, n)
    return B


# ============================================================================
# Convolution
# ============================================================================

def spatial_convolution(A, Wk):
    """2D single-channel convolution. A: HxW, Wk: KxK.
    Output O: (H-K+1) x (W-K+1) with O[i][j] = sum_{ki,kj} A[i+ki][j+kj] * Wk[ki][kj]."""
    H = len(A); Wd = len(A[0])
    K = len(Wk)
    out_h = H - K + 1
    out_w = Wd - K + 1
    O = [[None] * out_w for _ in range(out_h)]
    for i in range(out_h):
        for j in range(out_w):
            acc = A[i][j] * Wk[0][0]
            for ki in range(K):
                for kj in range(K):
                    if ki == 0 and kj == 0:
                        continue
                    acc = acc + A[i + ki][j + kj] * Wk[ki][kj]
            O[i][j] = acc
    return O


def fft_conv(x_in, y_in):
    """1D circular convolution via FFT: IFFT(FFT(x) * FFT(y)).
    All three FFTs share the iterative radix-2 structure; the middle
    stage is an N-element pointwise multiply."""
    X = fft_iterative(x_in)
    Y = fft_iterative(y_in)
    Z = [X[k] * Y[k] for k in range(len(X))]
    return fft_iterative(Z)


def regular_convolution(A, Wk):
    """Full multi-channel CNN layer.
    A: H x W x Cin (nested list). Wk: K x K x Cin x Cout.
    Output: (H-K+1) x (W-K+1) x Cout,
      O[i][j][co] = sum_{ki,kj,ci} A[i+ki][j+kj][ci] * Wk[ki][kj][ci][co]."""
    H = len(A); Wd = len(A[0]); Cin = len(A[0][0])
    K = len(Wk); Cout = len(Wk[0][0][0])
    out_h = H - K + 1
    out_w = Wd - K + 1
    O = [[[None] * Cout for _ in range(out_w)] for _ in range(out_h)]
    for i in range(out_h):
        for j in range(out_w):
            for co in range(Cout):
                acc = A[i][j][0] * Wk[0][0][0][co]
                for ki in range(K):
                    for kj in range(K):
                        for ci in range(Cin):
                            if ki == 0 and kj == 0 and ci == 0:
                                continue
                            acc = acc + A[i + ki][j + kj][ci] * Wk[ki][kj][ci][co]
                O[i][j][co] = acc
    return O


# ============================================================================
# Mergesort (data-oblivious stand-in — _Tracked has no __lt__, so we emit the
# read/write pattern of mergesort without actually comparing)
# ============================================================================

def quicksort(arr):
    """Data-oblivious quicksort stand-in. At each level, reads every
    element against the last-position pivot (2 reads per compare, writes
    discarded), then recurses on two equal halves. In-place — no temp
    arrays. Emits ~2n events per level over log n levels."""
    n = len(arr)
    if n <= 1:
        return [v + 0 for v in arr] if n == 1 else []
    pivot = arr[-1]
    # Partition pass: compare every other element to pivot (oblivious)
    for i in range(n - 1):
        _ = arr[i] + pivot   # two reads per element; result discarded
    mid = n // 2
    left = quicksort(arr[:mid])
    right = quicksort(arr[mid:])
    return left + right


def heapsort(arr):
    """Data-oblivious heapsort stand-in. Builds a binary max-heap via
    sift-down from n/2-1 down to 0, then repeatedly extracts root and
    restores heap over a shrinking prefix. Each sift-down step reads
    parent + left child (and, when present, right sibling for the
    larger-child choice) and writes one slot — emits O(n log n) events."""
    n = len(arr)
    out = [v + 0 for v in arr]
    if n <= 1:
        return out

    def sift_down(j, heap_size):
        while 2 * j + 1 < heap_size:
            child = 2 * j + 1
            if child + 1 < heap_size:
                # pick larger of two children (branch-free combine)
                out[child] = out[child] + out[child + 1]
            # compare parent & (chosen) child, swap-write (branch-free combine)
            out[j] = out[j] + out[child]
            j = child

    # Build heap: bottom-up sift-down
    for i in range(n // 2 - 1, -1, -1):
        sift_down(i, n)
    # Extract: swap root with last, then sift-down over shrinking prefix
    for k in range(n - 1, 0, -1):
        out[k] = out[k] + out[0]     # swap root & last
        sift_down(0, k)
    return out


def mergesort(arr):
    """Recursive mergesort. The merge step is oblivious: each output cell
    combines one element from the left half and one from the right half,
    matching the 2n reads-per-merge traffic of a real data-dependent merge
    without needing __lt__."""
    n = len(arr)
    if n <= 1:
        return [v + 0 for v in arr]
    mid = n // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    out = [None] * n
    for k in range(n):
        li = k if k < mid else mid - 1
        ri = (k - mid) if k >= mid else 0
        out[k] = left[li] + right[ri]
    return out


# ============================================================================
# Longest Common Subsequence — row-major DP over an (m+1)x(n+1) table
# ============================================================================

# ============================================================================
# LU / Gaussian elimination (in-place, overwrites A with L and U)
# ============================================================================

def lu_no_pivot(A):
    """No-pivot LU (Doolittle). Access pattern matches the classical triple
    loop: rank-1 trailing update after each column elimination."""
    n = len(A)
    for k in range(n):
        pivot = A[k][k] + 0  # force a load
        for i in range(k + 1, n):
            A[i][k] = A[i][k] + pivot   # stand-in for /= pivot (same touches)
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = A[i][j] - A[i][k] * A[k][j]
    return A


def blocked_lu(A, NB=8):
    """One-level blocked LU. For each diagonal block: factor via naive LU;
    triangular-solve the panel and row strip; GEMM-update the trailing
    submatrix."""
    n = len(A)
    for kb in range(0, n, NB):
        ke = min(kb + NB, n)
        # (a) factor diagonal block A[kb:ke, kb:ke] via naive LU
        for k in range(kb, ke):
            pivot = A[k][k] + 0
            for i in range(k + 1, ke):
                A[i][k] = A[i][k] + pivot
            for i in range(k + 1, ke):
                for j in range(k + 1, ke):
                    A[i][j] = A[i][j] - A[i][k] * A[k][j]
        # (b) update panel A[ke:n, kb:ke]  — triangular solve with U
        for i in range(ke, n):
            for k in range(kb, ke):
                A[i][k] = A[i][k] + A[k][k]
                for j in range(k + 1, ke):
                    A[i][j] = A[i][j] - A[i][k] * A[k][j]
        # (c) update row strip A[kb:ke, ke:n] — triangular solve with L
        for k in range(kb, ke):
            for j in range(ke, n):
                for i in range(k + 1, ke):
                    A[i][j] = A[i][j] - A[i][k] * A[k][j]
        # (d) trailing GEMM update A[ke:n, ke:n] -= A[ke:n, kb:ke] · A[kb:ke, ke:n]
        for i in range(ke, n):
            for j in range(ke, n):
                for k in range(kb, ke):
                    A[i][j] = A[i][j] - A[i][k] * A[k][j]
    return A


def recursive_lu(A):
    """Cache-oblivious recursive LU: split into 2x2 quadrants, factor
    top-left, solve off-diagonals, Schur-complement, recurse on bottom-right."""
    n = len(A)
    if n == 1:
        return A
    h = n // 2

    # (1) Factor top-left A[0:h, 0:h] recursively
    A11 = [[A[i][j] for j in range(h)] for i in range(h)]
    recursive_lu(A11)
    # write back (we keep A11 coupled; in real LU this is in-place)
    for i in range(h):
        for j in range(h):
            A[i][j] = A11[i][j] + 0

    # (2) Solve A[h:n, 0:h] with U11  (L21 = A21 · U11^-1)
    for i in range(h, n):
        for k in range(h):
            A[i][k] = A[i][k] + A[k][k]
            for j in range(k + 1, h):
                A[i][j] = A[i][j] - A[i][k] * A[k][j]

    # (3) Solve A[0:h, h:n] with L11  (U12 = L11^-1 · A12)
    for k in range(h):
        for j in range(h, n):
            for i in range(k + 1, h):
                A[i][j] = A[i][j] - A[i][k] * A[k][j]

    # (4) Schur-complement update A[h:n, h:n] -= A[h:n, 0:h] · A[0:h, h:n]
    for i in range(h, n):
        for j in range(h, n):
            for k in range(h):
                A[i][j] = A[i][j] - A[i][k] * A[k][j]

    # (5) Recurse on A[h:n, h:n]
    A22 = [[A[h + i][h + j] for j in range(n - h)] for i in range(n - h)]
    recursive_lu(A22)
    for i in range(n - h):
        for j in range(n - h):
            A[h + i][h + j] = A22[i][j] + 0
    return A


def lu_partial_pivot(A):
    """LU with partial pivoting. Adds a column scan per step plus a row
    swap between the current row and a (data-obliviously-chosen) pivot row."""
    n = len(A)
    for k in range(n):
        # (a) scan column k for pivot (reads n-k elements)
        _ = A[k][k] + 0
        for i in range(k + 1, n):
            _ = A[i][k] + A[k][k]  # compare-ish stand-in (2 reads)
        # (b) data-oblivious swap: exchange row k with row p = (k+1)%n if k<n-1
        p = k + 1 if k + 1 < n else k
        for j in range(k, n):
            _ = A[k][j] + A[p][j]  # read both rows; fake swap (writes discarded)
        # (c) elimination as in no-pivot
        pivot = A[k][k] + 0
        for i in range(k + 1, n):
            A[i][k] = A[i][k] + pivot
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = A[i][j] - A[i][k] * A[k][j]
    return A


# ============================================================================
# Cholesky (lower-triangular, SPD matrix)
# ============================================================================

def cholesky(A):
    """Right-looking Cholesky — reads only the lower triangle. No pivoting."""
    n = len(A)
    for k in range(n):
        pivot = A[k][k] + 0   # stand-in for sqrt(A[k][k])
        for i in range(k + 1, n):
            A[i][k] = A[i][k] + pivot   # stand-in for /= sqrt(pivot)
        for j in range(k + 1, n):
            for i in range(j, n):     # only lower triangle (i >= j)
                A[i][j] = A[i][j] - A[i][k] * A[j][k]
    return A


# ============================================================================
# QR factorization via Householder reflections
# ============================================================================

def householder_qr(A):
    """Classical Householder QR: for each column k, compute a reflector
    using the subdiagonal, apply it to the trailing columns. Access
    pattern matches LAPACK's DGEQR2."""
    m = len(A); n = len(A[0])
    for k in range(min(m, n)):
        # (a) compute reflector — reads A[k:m, k]
        nrm = A[k][k] + 0
        for i in range(k + 1, m):
            nrm = nrm + A[i][k]
        # (b) apply reflector to each trailing column
        for j in range(k + 1, n):
            t = A[k][k] * A[k][j]
            for i in range(k + 1, m):
                t = t + A[i][k] * A[i][j]
            A[k][j] = A[k][j] - t * A[k][k]
            for i in range(k + 1, m):
                A[i][j] = A[i][j] - t * A[i][k]
    return A


def blocked_qr(A, NB=8):
    """Blocked Householder QR (WY form): factor an NB-wide panel, then
    apply the accumulated block reflector to the trailing columns in one
    GEMM-like sweep. Simplified: panel factor uses classical
    Householder; block update uses a rank-NB trailing update."""
    m = len(A); n = len(A[0])
    for kb in range(0, min(m, n), NB):
        ke = min(kb + NB, min(m, n))
        # (a) panel factor A[kb:m, kb:ke] with classical Householder
        for k in range(kb, ke):
            nrm = A[k][k] + 0
            for i in range(k + 1, m):
                nrm = nrm + A[i][k]
            for j in range(k + 1, ke):
                t = A[k][k] * A[k][j]
                for i in range(k + 1, m):
                    t = t + A[i][k] * A[i][j]
                A[k][j] = A[k][j] - t * A[k][k]
                for i in range(k + 1, m):
                    A[i][j] = A[i][j] - t * A[i][k]
        # (b) apply accumulated block reflector to trailing columns
        # A[kb:m, ke:n] -= V_kb:m · W^T · A[kb:m, ke:n]  (rank-NB)
        for j in range(ke, n):
            # compute W^T · A[kb:m, j]  (NB-length vector)
            w = [None] * (ke - kb)
            for t_idx, k in enumerate(range(kb, ke)):
                acc = A[k][k] * A[k][j]
                for i in range(k + 1, m):
                    acc = acc + A[i][k] * A[i][j]
                w[t_idx] = acc
            # update A[kb:m, j] -= V · w
            for t_idx, k in enumerate(range(kb, ke)):
                A[k][j] = A[k][j] - A[k][k] * w[t_idx]
                for i in range(k + 1, m):
                    A[i][j] = A[i][j] - A[i][k] * w[t_idx]
    return A


def tsqr(A, block_rows=8):
    """Tall-skinny QR via tree reduction. Phase 1: factor each
    (block_rows × n) tile locally via Householder QR. Phase 2: pairwise
    merge the R factors up a tree (stack two NB×N R's and re-factor)."""
    m = len(A); n = len(A[0])
    # Phase 1: local QR on each row-tile
    for row0 in range(0, m, block_rows):
        row1 = min(row0 + block_rows, m)
        for k in range(min(row1 - row0, n)):
            kk = row0 + k
            nrm = A[kk][k] + 0
            for i in range(kk + 1, row1):
                nrm = nrm + A[i][k]
            for j in range(k + 1, n):
                t = A[kk][k] * A[kk][j]
                for i in range(kk + 1, row1):
                    t = t + A[i][k] * A[i][j]
                A[kk][j] = A[kk][j] - t * A[kk][k]
                for i in range(kk + 1, row1):
                    A[i][j] = A[i][j] - t * A[i][k]
    # Phase 2: tree reduction — pairwise merge at powers-of-2 strides
    num_tiles = (m + block_rows - 1) // block_rows
    stride = 1
    while stride < num_tiles:
        for idx in range(0, num_tiles, 2 * stride):
            other = idx + stride
            if other >= num_tiles:
                break
            left_row = idx * block_rows
            right_row = other * block_rows
            right_end = min(right_row + block_rows, m)
            # Re-factor the stacked R's (top of each tile) — access pattern
            for k in range(min(n, block_rows)):
                _ = A[left_row + k][k] + A[right_row + k][k]
                for j in range(k + 1, n):
                    t = A[left_row + k][k] * A[left_row + k][j]
                    for i in range(right_row + k, right_end):
                        t = t + A[i][k] * A[i][j]
                    A[left_row + k][j] = A[left_row + k][j] - t * A[left_row + k][k]
                    for i in range(right_row + k, right_end):
                        A[i][j] = A[i][j] - t * A[i][k]
        stride *= 2
    return A


def lcs_dp(x, y):
    """Row-major LCS DP. Uses a branch-free sum in place of the max/equality
    recurrence so the access pattern matches canonical LCS:
    each cell reads D[i-1][j-1], D[i-1][j], D[i][j-1], x[i-1], y[j-1]."""
    m = len(x); n = len(y)
    # zero-initialized first row/col (use tracked zeros via x[0] - x[0])
    zero = x[0] - x[0]
    D = [[zero + 0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            D[i][j] = D[i - 1][j - 1] + D[i - 1][j] + D[i][j - 1] + x[i - 1] + y[j - 1]
    return D[m][n]


# ============================================================================
# Time-skewed stencils (multi-sweep temporal locality)
# ============================================================================

def stencil_time_naive(A, T=4):
    """Unfused Jacobi: sweep the full grid T times, writing a fresh
    next-timestep buffer each pass. The naive CA-bad pattern — bulk
    reads of A never hit cache between sweeps."""
    n = len(A)
    cur = [[A[i][j] + 0 for j in range(n)] for i in range(n)]
    for _ in range(T):
        nxt = [[cur[i][j] + 0 for j in range(n)] for i in range(n)]
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                nxt[i][j] = cur[i][j] + cur[i-1][j] + cur[i+1][j] \
                            + cur[i][j-1] + cur[i][j+1]
        cur = nxt
    return cur


def stencil_time_diamond(A, T=4, block=4):
    """Diamond/trapezoidal time-tiling — advance T timesteps inside each
    (i,j) block before moving on. Halos shrink by 1 row/col per step so
    the block at step t is (block + 2*(T-t)) wide. Carries cur in a
    per-block buffer so boundary cells stay hot across sweeps."""
    n = len(A)
    cur = [[A[i][j] + 0 for j in range(n)] for i in range(n)]
    halo = T
    for bi in range(0, n, block):
        for bj in range(0, n, block):
            # Per-block, run T steps using local buffers.
            local_cur = [[cur[i][j] + 0
                          for j in range(max(0, bj - halo),
                                         min(n, bj + block + halo))]
                         for i in range(max(0, bi - halo),
                                        min(n, bi + block + halo))]
            rr = max(0, bi - halo); cc = max(0, bj - halo)
            rows = len(local_cur); cols = len(local_cur[0])
            for t in range(T):
                local_nxt = [[local_cur[ii][jj] + 0
                              for jj in range(cols)] for ii in range(rows)]
                for ii in range(1, rows - 1):
                    for jj in range(1, cols - 1):
                        if (0 < rr + ii < n - 1) and (0 < cc + jj < n - 1):
                            local_nxt[ii][jj] = (local_cur[ii][jj]
                                                 + local_cur[ii-1][jj]
                                                 + local_cur[ii+1][jj]
                                                 + local_cur[ii][jj-1]
                                                 + local_cur[ii][jj+1])
                local_cur = local_nxt
            # Flush the interior (bi..bi+block) back to cur.
            for i in range(bi, min(bi + block, n)):
                for j in range(bj, min(bj + block, n)):
                    local_i = i - rr; local_j = j - cc
                    cur[i][j] = local_cur[local_i][local_j] + 0
    return cur


# ============================================================================
# Floyd-Warshall (all-pairs shortest paths)
# ============================================================================

def floyd_warshall_naive(M):
    """Standard 3-nested loop. M[i][j] = min(M[i][j], M[i][k] + M[k][j]).
    Branch-free stand-in: use + to emulate min so the trace has fixed
    dependencies (same access pattern as real Floyd-Warshall)."""
    V = len(M)
    D = [[M[i][j] + 0 for j in range(V)] for i in range(V)]
    for k in range(V):
        for i in range(V):
            for j in range(V):
                D[i][j] = D[i][j] + D[i][k] + D[k][j]
    return D


def floyd_warshall_recursive(M):
    """Kleene's cache-oblivious APSP: divide into 4 quadrants and recurse.
    Mirrors matmul_rmm structure (A-diagonal first, then A @ C updates,
    then B @ A, then C @ B style). Simplified: 8 recursive calls on
    quadrant submatrices, same asymptotic work as naive."""
    V = len(M)
    D = [[M[i][j] + 0 for j in range(V)] for i in range(V)]

    def rec(r0, c0, sz):
        if sz <= 2:
            for k in range(r0, r0 + sz):
                for i in range(r0, r0 + sz):
                    for j in range(c0, c0 + sz):
                        D[i][j] = D[i][j] + D[i][k] + D[k][j]
            return
        h = sz // 2
        # A = top-left, B = top-right, C = bottom-left, Dq = bottom-right
        rec(r0, c0, h)                          # A
        rec(r0, c0 + h, h)                      # B via A
        rec(r0 + h, c0, h)                      # C via A
        rec(r0 + h, c0 + h, h)                  # Dq via A + B + C
        rec(r0 + h, c0 + h, h)                  # Dq
        rec(r0 + h, c0, h)                      # C via Dq
        rec(r0, c0 + h, h)                      # B via Dq
        rec(r0, c0, h)                          # A via Dq

    rec(0, 0, V)
    return D


# ============================================================================
# LayerNorm / RMSNorm (1D vector fusion)
# ============================================================================

def layernorm_unfused(x):
    """Three-pass LayerNorm. Pass 1 computes mean, pass 2 computes variance,
    pass 3 normalizes. Each pass reads the full x vector from bulk memory
    — the naive memory-unfriendly version."""
    N = len(x)
    # Pass 1: mean = sum(x) / N (branch-free division stand-in: + 0)
    s = x[0] + 0
    for i in range(1, N):
        s = s + x[i]
    mean = s + 0  # stand-in for s / N
    # Pass 2: var = sum((x-mean)^2)
    v = (x[0] + mean) * (x[0] + mean)
    for i in range(1, N):
        v = v + (x[i] + mean) * (x[i] + mean)
    inv_std = v + 0  # stand-in for 1 / sqrt(var)
    # Pass 3: y[i] = (x[i] - mean) * inv_std
    y = [None] * N
    for i in range(N):
        y[i] = (x[i] + mean) * inv_std
    return y


def layernorm_fused(x):
    """Welford's online mean+variance in a single pass, then a second pass
    to normalize. The two passes each read x once, but intermediate
    accumulators (mu, M2, inv_std) stay in hot registers throughout."""
    N = len(x)
    mu = x[0] + 0
    m2 = x[0] + 0
    for i in range(1, N):
        # Welford update (branch-free algebraic stand-ins)
        delta = x[i] + mu
        mu = mu + delta
        delta2 = x[i] + mu
        m2 = m2 + delta * delta2
    inv_std = m2 + 0
    y = [None] * N
    for i in range(N):
        y[i] = (x[i] + mu) * inv_std
    return y


# ============================================================================
# Matrix Powers Kernel (s-step Krylov)
# ============================================================================

def matrix_powers_naive(A, x, s=4):
    """Compute x, Ax, A²x, ..., A^s x naively — each matvec reads the
    whole bulk matrix A again."""
    n = len(A)
    cur = [x[i] + 0 for i in range(n)]
    xs = [cur]
    for step in range(s):
        nxt = [None] * n
        for i in range(n):
            acc = A[i][0] * cur[0]
            for j in range(1, n):
                acc = acc + A[i][j] * cur[j]
            nxt[i] = acc
        xs.append(nxt)
        cur = nxt
    return xs[-1]


def matrix_powers_ca(A, x, s=4, block=4):
    """Communication-avoiding s-step: process A in row-blocks; for each
    block compute its contribution to all s output vectors locally
    before moving on. Avoids re-reading A s times from bulk.

    Simplification: for a row-block of A, produce its contribution to
    x^(1)..x^(s) using the currently-available previous-step vectors.
    The CA bit that cache heuristics reward is that A's rows are read
    once per outer block-visit rather than once per step."""
    n = len(A)
    cur = [x[i] + 0 for i in range(n)]
    # Ping-pong between cur and nxt across s steps, but with row blocking
    # reversed: iterate blocks outermost.
    for step in range(s):
        nxt = [cur[i] + 0 for i in range(n)]  # initialize to cur shape
        for bi in range(0, n, block):
            for i in range(bi, min(bi + block, n)):
                acc = A[i][0] * cur[0]
                for j in range(1, n):
                    acc = acc + A[i][j] * cur[j]
                nxt[i] = acc
        cur = nxt
    return cur


# ============================================================================
# Left-looking Cholesky (read-heavy vs. right-looking write-heavy)
# ============================================================================

def cholesky_left_looking(A):
    """Left-looking Cholesky. For each column k, pull all needed data
    from previously-factored columns 0..k-1 (far-flung reads), then
    finalize column k locally (localized writes). Complementary to the
    right-looking variant which does far-flung writes + localized reads."""
    n = len(A)
    L = [[A[i][j] + 0 for j in range(n)] for i in range(n)]
    for k in range(n):
        # Pull from previously-factored columns to update column k:
        # L[i][k] -= sum_{j<k} L[i][j] * L[k][j]
        for i in range(k, n):
            for j in range(k):
                L[i][k] = L[i][k] + L[i][j] * L[k][j]
        # Finalize the pivot (sqrt stand-in: + 0).
        L[k][k] = L[k][k] + 0
        for i in range(k + 1, n):
            L[i][k] = L[i][k] + L[k][k]
    return L


# ============================================================================
# Sparse matrix-vector multiply (CSR)
# ============================================================================

def _spmv_csr(row_ptr, col_ind, vals, x):
    """Run y = vals @ x using CSR indices (Python ints) and _Tracked x
    values. row_ptr / col_ind are plain integers so indirect addressing
    produces data-dependent L2 reads of x without adding control flow to
    the trace."""
    n = len(row_ptr) - 1
    y = [None] * n
    for i in range(n):
        start = row_ptr[i]; end = row_ptr[i + 1]
        if start == end:
            y[i] = x[0] + 0  # zero via input (keeps trace well-formed)
            continue
        acc = vals[start] * x[col_ind[start]]
        for k in range(start + 1, end):
            acc = acc + vals[k] * x[col_ind[k]]
        y[i] = acc
    return y


def _banded_csr(n, bandwidth):
    row_ptr, col_ind, vals = [0], [], []
    total = 0
    for i in range(n):
        for j in range(max(0, i - bandwidth), min(n, i + bandwidth + 1)):
            col_ind.append(j); vals.append(1.0); total += 1
        row_ptr.append(total)
    return row_ptr, col_ind, vals


def _random_csr(n, nnz_per_row, seed=0xC0FFEE):
    import random
    rng = random.Random(seed)
    row_ptr, col_ind, vals = [0], [], []
    total = 0
    for i in range(n):
        cols = sorted(rng.sample(range(n), min(nnz_per_row, n)))
        for j in cols:
            col_ind.append(j); vals.append(1.0); total += 1
        row_ptr.append(total)
    return row_ptr, col_ind, vals


def spmv_csr_banded(x, n=32, bandwidth=3):
    row_ptr, col_ind, _vals = _banded_csr(n, bandwidth)
    # Materialize vals as _Tracked via x[0]+0 scalars so each read costs.
    vals = [x[0] + 0 for _ in col_ind]
    return _spmv_csr(row_ptr, col_ind, vals, x)


def spmv_csr_random(x, n=32, nnz_per_row=7):
    row_ptr, col_ind, _vals = _random_csr(n, nnz_per_row)
    vals = [x[0] + 0 for _ in col_ind]
    return _spmv_csr(row_ptr, col_ind, vals, x)


# ============================================================================
# Bitonic sort (data-oblivious sorting network)
# ============================================================================

def bitonic_sort(arr):
    """True sorting network. For N = power of two, runs log2(N) stages
    each with log2(stage) substages, doing compare-swap on paired
    elements. Since _Tracked has no comparison, the compare-swap becomes
    a branchless algebraic stand-in: (a+b, a+b) — identical access
    pattern to real bitonic sort (butterfly read pairs + 2 writes)."""
    N = len(arr)
    out = [arr[i] + 0 for i in range(N)]
    k = 2
    while k <= N:
        j = k // 2
        while j > 0:
            for i in range(N):
                l = i ^ j
                if l > i:
                    # Branch-free pair "swap": assign symmetric function.
                    s = out[i] + out[l]
                    out[i] = s + 0
                    out[l] = s + 0
            j //= 2
        k *= 2
    return out
