"""Manual (hand-placed) implementations for each algorithm in the grid.

Each function returns the total cost under the 2D cache model:
  cost = sum over memory touches of ceil(sqrt(addr)).

Convention (matching bytedmd_ir.cost): every binop reads its inputs; stores
are free. Scalar accumulators and temporaries are allocated at low addresses
so they sit near the center of the Manhattan disc.
"""
from __future__ import annotations

import math


class Allocator:
    """Bump-pointer allocator with push/pop stack discipline.

    touch(addr) charges ceil(sqrt(addr)) — modelling one memory access
    at that address in the continuous-Manhattan cache.
    """
    __slots__ = ("cost", "ptr", "peak")

    def __init__(self) -> None:
        self.cost = 0
        self.ptr = 1
        self.peak = 1

    def alloc(self, size: int) -> int:
        addr = self.ptr
        self.ptr += size
        if self.ptr > self.peak:
            self.peak = self.ptr
        return addr

    def push(self) -> int:
        return self.ptr

    def pop(self, p: int) -> None:
        self.ptr = p

    def touch(self, addr: int) -> None:
        self.cost += math.isqrt(max(0, addr - 1)) + 1


# ============================================================================
# Matmul family
# ============================================================================

def manual_naive_matmul(n: int) -> int:
    """Hand-placed naive triple loop. Scratch slots s, t at addrs 1,2."""
    a = Allocator()
    s = a.alloc(1); t = a.alloc(1)
    A = a.alloc(n * n); B = a.alloc(n * n); C = a.alloc(n * n)
    for i in range(n):
        for j in range(n):
            # s = A[i][0] * B[0][j]
            a.touch(A + i * n + 0); a.touch(B + 0 * n + j)
            for k in range(1, n):
                # t = A[i][k] * B[k][j]
                a.touch(A + i * n + k); a.touch(B + k * n + j)
                # s = s + t
                a.touch(s); a.touch(t)
            # C[i][j] = s
            a.touch(s)
    return a.cost


def manual_tiled_matmul(n: int, T: int | None = None) -> int:
    """One-level blocked matmul. Scratchpad holds sA, sB, sC (T*T each)
    at the lowest addresses, reloaded per (bi,bj,bk) tile."""
    if T is None:
        T = max(1, int(round(n ** 0.5)))
    a = Allocator()
    # Scratchpad first — hottest addresses
    sA = a.alloc(T * T); sB = a.alloc(T * T); sC = a.alloc(T * T)
    tmp = a.alloc(1)
    A = a.alloc(n * n); B = a.alloc(n * n); C = a.alloc(n * n)

    def load_tile(src_base: int, src_stride: int, dst_base: int, bi: int, bj: int) -> None:
        for ii in range(T):
            for jj in range(T):
                if bi + ii < n and bj + jj < n:
                    a.touch(src_base + (bi + ii) * src_stride + (bj + jj))

    def store_tile(dst_base: int, dst_stride: int, src_base: int, bi: int, bj: int) -> None:
        for ii in range(T):
            for jj in range(T):
                if bi + ii < n and bj + jj < n:
                    a.touch(src_base + ii * T + jj)

    for bi in range(0, n, T):
        for bj in range(0, n, T):
            # Load C tile (to accumulate)
            load_tile(C, n, sC, bi, bj)
            for bk in range(0, n, T):
                load_tile(A, n, sA, bi, bk)
                load_tile(B, n, sB, bk, bj)
                for ii in range(min(T, n - bi)):
                    for jj in range(min(T, n - bj)):
                        for kk in range(min(T, n - bk)):
                            # tmp = sA[ii][kk] * sB[kk][jj]
                            a.touch(sA + ii * T + kk); a.touch(sB + kk * T + jj)
                            # sC[ii][jj] = sC[ii][jj] + tmp
                            a.touch(sC + ii * T + jj); a.touch(tmp)
            # Flush sC -> C
            store_tile(C, n, sC, bi, bj)
    return a.cost


def manual_rmm(n: int, T: int = 4) -> int:
    """Recursive (cache-oblivious) matmul with T-tile scratchpad at leaves."""
    a = Allocator()
    sA = a.alloc(T * T); sB = a.alloc(T * T); sC = a.alloc(T * T)
    tmp = a.alloc(1)
    A = a.alloc(n * n); B = a.alloc(n * n); C = a.alloc(n * n)

    def compute_tile(pA: int, sAstr: int, pB: int, sBstr: int, pC: int, sCstr: int) -> None:
        # Load A, B, C tiles
        for ii in range(T):
            for jj in range(T):
                a.touch(pA + ii * sAstr + jj)
        for ii in range(T):
            for jj in range(T):
                a.touch(pB + ii * sBstr + jj)
        for ii in range(T):
            for jj in range(T):
                a.touch(pC + ii * sCstr + jj)
        # MAC into scratchpad sC
        for ii in range(T):
            for jj in range(T):
                for kk in range(T):
                    a.touch(sA + ii * T + kk); a.touch(sB + kk * T + jj)
                    a.touch(sC + ii * T + jj); a.touch(tmp)
        # Flush sC back
        for ii in range(T):
            for jj in range(T):
                a.touch(sC + ii * T + jj)

    def recurse(rA: int, cA: int, rB: int, cB: int, rC: int, cC: int, sz: int) -> None:
        if sz <= T:
            compute_tile(A + rA * n + cA, n, B + rB * n + cB, n, C + rC * n + cC, n)
            return
        h = sz // 2
        for dr, dc, erb, ecb, frc, fcc in [
            (0, 0, 0, 0, 0, 0), (0, 0, 0, h, 0, h),
            (h, 0, 0, h, h, h), (h, 0, 0, 0, h, 0),
            (h, h, h, 0, h, 0), (h, h, h, h, h, h),
            (0, h, h, h, 0, h), (0, h, h, 0, 0, 0),
        ]:
            recurse(rA + dr, cA + dc, rB + erb, cB + ecb, rC + frc, cC + fcc, h)

    recurse(0, 0, 0, 0, 0, 0, n)
    return a.cost


def manual_strassen(n: int, T: int = 4) -> int:
    """Strassen with scratchpad at base case + push/pop stack for 7 intermediates."""
    a = Allocator()
    sA = a.alloc(T * T); sB = a.alloc(T * T); sC = a.alloc(T * T)
    tmp = a.alloc(1)
    A = a.alloc(n * n); B = a.alloc(n * n); C = a.alloc(n * n)

    def add_mats(p1: int, s1: int, p2: int, s2: int, h: int) -> None:
        for i in range(h):
            for j in range(h):
                a.touch(p1 + i * s1 + j); a.touch(p2 + i * s2 + j)

    def compute_tile(pA: int, sAstr: int, pB: int, sBstr: int, pC: int, sCstr: int) -> None:
        for ii in range(T):
            for jj in range(T):
                a.touch(pA + ii * sAstr + jj)
        for ii in range(T):
            for jj in range(T):
                a.touch(pB + ii * sBstr + jj)
        for ii in range(T):
            for jj in range(T):
                a.touch(pC + ii * sCstr + jj)
        for ii in range(T):
            for jj in range(T):
                for kk in range(T):
                    a.touch(sA + ii * T + kk); a.touch(sB + kk * T + jj)
                    a.touch(sC + ii * T + jj); a.touch(tmp)
        for ii in range(T):
            for jj in range(T):
                a.touch(sC + ii * T + jj)

    def recurse(pA_: int, sAstr: int, pB_: int, sBstr: int, pC_: int, sCstr: int, sz: int) -> None:
        if sz <= T:
            compute_tile(pA_, sAstr, pB_, sBstr, pC_, sCstr)
            return
        h = sz // 2
        ckpt = a.push()
        SA = a.alloc(h * h); SB = a.alloc(h * h)
        M = [a.alloc(h * h) for _ in range(7)]

        A11, A12 = pA_, pA_ + h
        A21, A22 = pA_ + h * sAstr, pA_ + h * sAstr + h
        B11, B12 = pB_, pB_ + h
        B21, B22 = pB_ + h * sBstr, pB_ + h * sBstr + h

        add_mats(A11, sAstr, A22, sAstr, h); add_mats(B11, sBstr, B22, sBstr, h)
        recurse(SA, h, SB, h, M[0], h, h)

        add_mats(A21, sAstr, A22, sAstr, h)
        recurse(SA, h, B11, sBstr, M[1], h, h)

        add_mats(B12, sBstr, B22, sBstr, h)
        recurse(A11, sAstr, SB, h, M[2], h, h)

        add_mats(B21, sBstr, B11, sBstr, h)
        recurse(A22, sAstr, SB, h, M[3], h, h)

        add_mats(A11, sAstr, A12, sAstr, h)
        recurse(SA, h, B22, sBstr, M[4], h, h)

        add_mats(A21, sAstr, A11, sAstr, h); add_mats(B11, sBstr, B12, sBstr, h)
        recurse(SA, h, SB, h, M[5], h, h)

        add_mats(A12, sAstr, A22, sAstr, h); add_mats(B21, sBstr, B22, sBstr, h)
        recurse(SA, h, SB, h, M[6], h, h)

        # Reads to assemble C quadrants (4 submatrix adds/subs each)
        def read_M(*indices: int) -> None:
            for i in range(h):
                for j in range(h):
                    for idx in indices:
                        a.touch(M[idx] + i * h + j)

        read_M(0, 3, 4, 6)
        read_M(2, 4)
        read_M(1, 3)
        read_M(0, 1, 2, 5)

        a.pop(ckpt)

    recurse(A, n, B, n, C, n, n)
    return a.cost


# ============================================================================
# Attention family
# ============================================================================

def manual_naive_attention(N: int, d: int) -> int:
    """Three stages: S = Q K^T, P = softmax(S), O = P V. Full N×N S materialized."""
    a = Allocator()
    # Hot scratch scalars at low addresses
    s_acc = a.alloc(1); tmp = a.alloc(1)
    row_max = a.alloc(1); row_sum = a.alloc(1); inv_sum = a.alloc(1)
    # Bulk data
    Q = a.alloc(N * d); K = a.alloc(N * d); V = a.alloc(N * d)
    S = a.alloc(N * N)
    O = a.alloc(N * d)

    # Stage 1: S[i][j] = Q[i] . K[j] (scale folded in)
    for i in range(N):
        for j in range(N):
            a.touch(Q + i * d + 0); a.touch(K + j * d + 0)
            for dd in range(1, d):
                a.touch(Q + i * d + dd); a.touch(K + j * d + dd)
                a.touch(s_acc); a.touch(tmp)
            # scale: one mul
            a.touch(s_acc)
            # write S[i][j] — free

    # Stage 2: row-wise softmax (in place on S, becomes P)
    for i in range(N):
        # row_max
        a.touch(S + i * N + 0)
        for j in range(1, N):
            a.touch(S + i * N + j); a.touch(row_max)
        # exp and sum
        for j in range(N):
            a.touch(S + i * N + j); a.touch(row_max)
            # exp -> write S[i][j] (reuse S as P, free write)
            if j == 0:
                pass  # row_sum initialized from P[i][0]
            else:
                a.touch(row_sum); a.touch(S + i * N + j)
        # inv_sum
        a.touch(row_sum)
        # normalize
        for j in range(N):
            a.touch(S + i * N + j); a.touch(inv_sum)

    # Stage 3: O[i][dd] = sum_j P[i][j] * V[j][dd]
    P = S
    for i in range(N):
        for dd in range(d):
            a.touch(P + i * N + 0); a.touch(V + 0 * d + dd)
            for j in range(1, N):
                a.touch(P + i * N + j); a.touch(V + j * d + dd)
                a.touch(s_acc); a.touch(tmp)
            # write O[i][dd] — free; but read s_acc once
            a.touch(s_acc)
    return a.cost


def manual_flash_attention(N: int, d: int, Bk: int) -> int:
    """Flash attention: stream K/V in blocks; never materialize full N×N S."""
    a = Allocator()
    # Hot scalars
    m_i = a.alloc(1); l_i = a.alloc(1)
    m_block = a.alloc(1); l_block = a.alloc(1)
    m_new = a.alloc(1); alpha = a.alloc(1); beta = a.alloc(1)
    inv_l = a.alloc(1); tmp = a.alloc(1)
    # Small block scratchpads
    s_block = a.alloc(Bk); p_block = a.alloc(Bk)
    o_acc = a.alloc(d)
    # Bulk data
    Q = a.alloc(N * d); K = a.alloc(N * d); V = a.alloc(N * d)
    O = a.alloc(N * d)

    num_blocks = (N + Bk - 1) // Bk

    for i in range(N):
        for kb in range(num_blocks):
            k0 = kb * Bk
            bs = min(Bk, N - k0)
            # s_block[j] = Q[i] . K[k0+j]
            for j in range(bs):
                a.touch(Q + i * d + 0); a.touch(K + (k0 + j) * d + 0)
                for dd in range(1, d):
                    a.touch(Q + i * d + dd); a.touch(K + (k0 + j) * d + dd)
                    a.touch(s_block + j); a.touch(tmp)
                a.touch(s_block + j)  # scale
            # m_block = max(s_block)
            a.touch(s_block + 0)
            for j in range(1, bs):
                a.touch(s_block + j); a.touch(m_block)
            # p_block[j] = exp(s - m_block); l_block = sum(p)
            for j in range(bs):
                a.touch(s_block + j); a.touch(m_block)
                if j > 0:
                    a.touch(p_block + j); a.touch(l_block)
            # online merge
            if kb == 0:
                a.touch(m_block); a.touch(l_block)
                for dd in range(d):
                    a.touch(p_block + 0); a.touch(V + k0 * d + dd)
                    for j in range(1, bs):
                        a.touch(p_block + j); a.touch(V + (k0 + j) * d + dd)
                        a.touch(o_acc + dd); a.touch(tmp)
            else:
                a.touch(m_i); a.touch(m_block)          # m_new = max
                a.touch(m_i); a.touch(m_new)            # alpha = exp(m_i - m_new)
                a.touch(m_block); a.touch(m_new)        # beta  = exp(m_block - m_new)
                a.touch(alpha); a.touch(l_i)
                a.touch(beta); a.touch(l_block)
                a.touch(tmp); a.touch(tmp)              # l_i = alpha*l_i + beta*l_block
                for dd in range(d):
                    a.touch(alpha); a.touch(o_acc + dd)  # rescale o_acc
                    for j in range(bs):
                        a.touch(p_block + j); a.touch(V + (k0 + j) * d + dd)
                        a.touch(beta); a.touch(tmp); a.touch(o_acc + dd)
        # Final: O[i][dd] = o_acc[dd] / l_i
        a.touch(l_i)
        for dd in range(d):
            a.touch(o_acc + dd); a.touch(inv_l)
    return a.cost


# ============================================================================
# Transpose family
# ============================================================================

def manual_transpose_naive(n: int) -> int:
    """Read A column-major (B[i][j] = A[j][i]) — one read per cell."""
    a = Allocator()
    A = a.alloc(n * n); B = a.alloc(n * n)
    for i in range(n):
        for j in range(n):
            a.touch(A + j * n + i)
            # write B[i][j] — free
    return a.cost


def manual_transpose_blocked(n: int, T: int | None = None) -> int:
    """Blocked iteration order over A — no scratchpad (redundant in the fixed-
    address Manhattan model). Reads each A cell once in block order."""
    if T is None:
        T = max(1, int(round(n ** 0.5)))
    a = Allocator()
    A = a.alloc(n * n); B = a.alloc(n * n)
    for bi in range(0, n, T):
        for bj in range(0, n, T):
            for ii in range(min(T, n - bi)):
                for jj in range(min(T, n - bj)):
                    a.touch(A + (bj + jj) * n + (bi + ii))
    return a.cost


def manual_transpose_recursive(n: int) -> int:
    """Cache-oblivious transpose: recursively split into 4 quadrants."""
    a = Allocator()
    A = a.alloc(n * n); B = a.alloc(n * n)

    def rec(ar: int, ac: int, br: int, bc: int, sz: int) -> None:
        if sz == 1:
            a.touch(A + ar * n + ac)
            return
        h = sz // 2
        rec(ar,     ac,     br,     bc,     h)
        rec(ar + h, ac,     br,     bc + h, h)
        rec(ar,     ac + h, br + h, bc,     h)
        rec(ar + h, ac + h, br + h, bc + h, h)

    rec(0, 0, 0, 0, n)
    return a.cost


# ============================================================================
# Matrix-vector
# ============================================================================

def manual_matvec_row(n: int) -> int:
    """y[i] = sum_j A[i][j] * x[j], outer loop over i — A read row-major.
    Hot slots first: s, tmp, y, x at low addrs; A is the cold bulk region."""
    a = Allocator()
    s = a.alloc(1); tmp = a.alloc(1)
    y = a.alloc(n); x = a.alloc(n)
    A = a.alloc(n * n)
    for i in range(n):
        a.touch(A + i * n + 0); a.touch(x + 0)
        for j in range(1, n):
            a.touch(A + i * n + j); a.touch(x + j)
            a.touch(s); a.touch(tmp)
        a.touch(s)  # write y[i]
    return a.cost


def manual_matvec_col(n: int) -> int:
    """Outer loop over j: y[i] += A[i][j] * x[j] — A read column-major (strided).
    Hot slots first: tmp, y, x at low addrs; A is the cold bulk region."""
    a = Allocator()
    tmp = a.alloc(1)
    y = a.alloc(n); x = a.alloc(n)
    A = a.alloc(n * n)
    for j in range(n):
        a.touch(x + j)
        for i in range(n):
            a.touch(A + i * n + j)
            if j == 0:
                pass  # y[i] initialized (free)
            else:
                a.touch(y + i); a.touch(tmp)
    return a.cost
