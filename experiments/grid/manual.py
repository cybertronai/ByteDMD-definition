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
    at that address in the continuous-Manhattan cache. When logging=True,
    also records the address sequence into .log for trace visualization.
    """
    __slots__ = ("cost", "ptr", "peak", "log")

    def __init__(self, logging: bool = False) -> None:
        self.cost = 0
        self.ptr = 1
        self.peak = 1
        self.log = [] if logging else None

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
        if self.log is not None:
            self.log.append(addr)


# Module-level override for the allocator used inside manual_* functions.
# Normally None (each function creates its own). generate_traces.py sets this
# to a logging Allocator so the captured .log reflects a single call's trace.
_CURRENT_ALLOC: Allocator | None = None


def set_allocator(a: Allocator | None) -> None:
    """Override the allocator returned by _alloc() until set_allocator(None).
    Used by generate_traces.py to inject a logging allocator."""
    global _CURRENT_ALLOC
    _CURRENT_ALLOC = a


def _alloc() -> Allocator:
    return _CURRENT_ALLOC if _CURRENT_ALLOC is not None else Allocator()


# ============================================================================
# Matmul family
# ============================================================================

def manual_naive_matmul(n: int) -> int:
    """Hand-placed naive triple loop computing C = A @ B^T.
    C[i][j] = sum_k A[i][k] * B[j][k] — both A and B are traversed
    row-major (contiguous) in the inner k-loop. Accumulator s at addr 1
    (read once per (i,j)); 2 reads (A[i][k], B[j][k]) per MAC."""
    a = _alloc()
    s = a.alloc(1)
    A = a.alloc(n * n); B = a.alloc(n * n); C = a.alloc(n * n)
    for i in range(n):
        for j in range(n):
            a.touch(s)  # accumulator init read, once per (i,j)
            for k in range(n):
                a.touch(A + i * n + k)
                a.touch(B + j * n + k)   # row-major B (AB^T access)
            # write C[i][j] = s (free)
    return a.cost


def manual_tiled_matmul(n: int, T: int | None = None) -> int:
    """One-level blocked matmul. Scratchpad holds sA, sB, sC (T*T each) at
    the lowest addresses, reloaded per (bi,bj,bk) tile. MAC convention:
    sC read once per (ii,jj) outside kk-loop; 2 reads per MAC."""
    if T is None:
        T = max(1, int(round(n ** 0.5)))
    a = _alloc()
    sA = a.alloc(T * T); sB = a.alloc(T * T); sC = a.alloc(T * T)
    A = a.alloc(n * n); B = a.alloc(n * n); C = a.alloc(n * n)

    for bi in range(0, n, T):
        for bj in range(0, n, T):
            # Load C tile into sC
            for ii in range(min(T, n - bi)):
                for jj in range(min(T, n - bj)):
                    a.touch(C + (bi + ii) * n + (bj + jj))
            for bk in range(0, n, T):
                # Load A tile
                for ii in range(min(T, n - bi)):
                    for kk in range(min(T, n - bk)):
                        a.touch(A + (bi + ii) * n + (bk + kk))
                # Load B tile
                for kk in range(min(T, n - bk)):
                    for jj in range(min(T, n - bj)):
                        a.touch(B + (bk + kk) * n + (bj + jj))
                # MAC
                for ii in range(min(T, n - bi)):
                    for jj in range(min(T, n - bj)):
                        a.touch(sC + ii * T + jj)
                        for kk in range(min(T, n - bk)):
                            a.touch(sA + ii * T + kk)
                            a.touch(sB + kk * T + jj)
            # Flush sC -> C
            for ii in range(min(T, n - bi)):
                for jj in range(min(T, n - bj)):
                    a.touch(sC + ii * T + jj)
    return a.cost


def manual_rmm(n: int, T: int = 4) -> int:
    """Recursive (cache-oblivious) matmul with T-tile scratchpad at leaves.
    MAC convention: fast_C read once per (i,j) outside k-loop; bulk read of
    fast_C before MAC; on first write of a C-tile, skip pre-load of pC
    (the tile is fresh)."""
    a = _alloc()
    sA = a.alloc(T * T); sB = a.alloc(T * T); sC = a.alloc(T * T)
    A = a.alloc(n * n); B = a.alloc(n * n); C = a.alloc(n * n)
    # is_first is checked against the PREVIOUS compute_tile's C coordinate
    # only — if the C tile loaded in fast_C still matches, skip the pC
    # pre-load. Matches strassen_trace's Scratchpad.sync cache semantic.
    last_C = [None]

    def compute_tile(rA: int, cA: int, rB: int, cB: int, rC: int, cC: int) -> None:
        is_first = (last_C[0] != (rC, cC))
        last_C[0] = (rC, cC)
        # Load A, B tiles
        for ii in range(T):
            for jj in range(T):
                a.touch(A + (rA + ii) * n + cA + jj)
        for ii in range(T):
            for jj in range(T):
                a.touch(B + (rB + ii) * n + cB + jj)
        # Bulk read of fast_C (accumulator init)
        for i in range(T * T):
            a.touch(sC + i)
        # MAC: fast_C[i][j] += sum_k fast_A[i][k] * fast_B[k][j]
        for ii in range(T):
            for jj in range(T):
                a.touch(sC + ii * T + jj)
                for kk in range(T):
                    a.touch(sA + ii * T + kk)
                    a.touch(sB + kk * T + jj)
        # Flush fast_C -> pC; if not first write, accumulate with existing pC
        for ii in range(T):
            for jj in range(T):
                a.touch(sC + ii * T + jj)
                if not is_first:
                    a.touch(C + (rC + ii) * n + cC + jj)

    def recurse(rA: int, cA: int, rB: int, cB: int, rC: int, cC: int, sz: int) -> None:
        if sz <= T:
            compute_tile(rA, cA, rB, cB, rC, cC)
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
    """Standard Strassen with scratchpad at base case + push/pop stack for 7
    materialized M-intermediates (SA, SB, M[0..6] — this is what fused_strassen
    avoids). MAC convention matches manual_rmm / strassen_trace.py."""
    a = _alloc()
    sA = a.alloc(T * T); sB = a.alloc(T * T); sC = a.alloc(T * T)
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
                a.touch(sC + ii * T + jj)
                for kk in range(T):
                    a.touch(sA + ii * T + kk)
                    a.touch(sB + kk * T + jj)
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


def manual_fused_strassen(n: int, T: int = 4) -> int:
    """Zero-Allocation Fused Strassen (ZAFS): single-level outer Strassen,
    no intermediate matrices — sub-additions are fused into the L1 tile
    loads and the 7 M-products are flushed straight into their target C
    quadrants. MAC convention matches strassen_trace.py / Gemini's
    reference: fast_C is read once per (i,j) outside the k-loop."""
    a = _alloc()
    fast_A = a.alloc(T * T); fast_B = a.alloc(T * T); fast_C = a.alloc(T * T)
    A = a.alloc(n * n); B = a.alloc(n * n); C = a.alloc(n * n)

    def compute_fused_tile(ops_A, ops_B, ops_C, r, c, k_off):
        # 1. Fused load A tile into fast_A (sum sign-weighted contributions)
        for i in range(T):
            for j in range(T):
                for _sgn, rb, cb in ops_A:
                    a.touch(A + (rb + r + i) * n + cb + k_off + j)
        # 2. Fused load B tile into fast_B
        for i in range(T):
            for j in range(T):
                for _sgn, rb, cb in ops_B:
                    a.touch(B + (rb + k_off + i) * n + cb + c + j)
        # 3. Bulk read of fast_C (accumulator init)
        for i in range(T * T):
            a.touch(fast_C + i)
        # 4. Tile MAC: fast_C[i][j] += sum_k fast_A[i][k] * fast_B[k][j]
        #    fast_C read once per (i,j) outside k-loop; 2 reads per MAC.
        for i in range(T):
            for j in range(T):
                a.touch(fast_C + i * T + j)
                for k in range(T):
                    a.touch(fast_A + i * T + k)
                    a.touch(fast_B + k * T + j)
        # 5. Fan-out fast_C -> multiple C targets with signs
        for _sgn, rb, cb, is_first in ops_C:
            for i in range(T):
                for j in range(T):
                    a.touch(fast_C + i * T + j)
                    if not is_first:
                        a.touch(C + (rb + r + i) * n + cb + c + j)

    h = n // 2
    q11, q12, q21, q22 = (0, 0), (0, h), (h, 0), (h, h)
    recipes = [
        ([(1, *q11), (1, *q22)], [(1, *q11), (1, *q22)], [(1, *q11, True), (1, *q22, True)]),
        ([(1, *q21), (1, *q22)], [(1, *q11)],            [(1, *q21, True), (-1, *q22, False)]),
        ([(1, *q11)],            [(1, *q12), (-1, *q22)], [(1, *q12, True), (1, *q22, False)]),
        ([(1, *q22)],            [(1, *q21), (-1, *q11)], [(1, *q11, False), (1, *q21, False)]),
        ([(1, *q11), (1, *q12)], [(1, *q22)],             [(-1, *q11, False), (1, *q12, False)]),
        ([(1, *q21), (-1, *q11)], [(1, *q11), (1, *q12)], [(1, *q22, False)]),
        ([(1, *q12), (-1, *q22)], [(1, *q21), (1, *q22)], [(1, *q11, False)]),
    ]
    for A_ops, B_ops, C_ops in recipes:
        for r, c in [(0, 0), (0, T), (T, 0), (T, T)]:
            compute_fused_tile(A_ops, B_ops, C_ops, r, c, k_off=0)
            C_ops_accum = [(sgn, rb, cb, False) for sgn, rb, cb, _ in C_ops]
            compute_fused_tile(A_ops, B_ops, C_ops_accum, r, c, k_off=T)
    return a.cost


# ============================================================================
# Attention family
# ============================================================================

def manual_naive_attention(N: int, d: int) -> int:
    """Three stages: S = Q K^T, P = softmax(S), O = P V. Full N×N S materialized."""
    a = _alloc()
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
    a = _alloc()
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
    a = _alloc()
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
    a = _alloc()
    A = a.alloc(n * n); B = a.alloc(n * n)
    for bi in range(0, n, T):
        for bj in range(0, n, T):
            for ii in range(min(T, n - bi)):
                for jj in range(min(T, n - bj)):
                    a.touch(A + (bj + jj) * n + (bi + ii))
    return a.cost


def manual_transpose_recursive(n: int) -> int:
    """Cache-oblivious transpose: recursively split into 4 quadrants."""
    a = _alloc()
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
    a = _alloc()
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


def manual_matvec_blocked(n: int, B: int = 4) -> int:
    """Blocked matvec with streaming-A and x-tile scratchpad, per
    gemini/efficient-matvec.md. Layout:
      addrs 1..B        : B accumulators s[0..B-1]
      addrs B+1..2B     : x_tile (B-slot scratchpad)
      addr  2B+1        : tmp (A·x product)
      addr  2B+2        : A_stream — single streaming FIFO port
      addrs 2B+3..2B+n+2: x_main (cold bulk for x)
    A is never statically stored in the spatial grid; every read of
    an A element goes to the same A_stream address."""
    a = _alloc()
    s = [a.alloc(1) for _ in range(B)]            # 1..B
    x_tile = a.alloc(B)                            # B+1..2B
    tmp = a.alloc(1)                               # 2B+1
    A_stream = a.alloc(1)                          # 2B+2
    x_main = a.alloc(n)                            # 2B+3..2B+n+2

    for i_out in range(0, n, B):
        for j_out in range(0, n, B):
            # DMA-load x tile from x_main → x_tile (writes free)
            for j in range(B):
                a.touch(x_main + j_out + j)
            # B×B MAC, streaming A through a single port
            for i in range(B):
                for j in range(B):
                    a.touch(A_stream)              # stream next A element
                    a.touch(x_tile + j)            # hot x tile read
                    if j_out != 0 or j != 0:
                        a.touch(s[i])              # read accumulator
                    a.touch(tmp)                   # read A*x product
        # Flush accumulators to y (reads of s, writes of y are free)
        for i in range(B):
            a.touch(s[i])
    return a.cost


def manual_matvec_col(n: int) -> int:
    """Outer loop over j: y[i] += A[i][j] * x[j] — A read column-major (strided).
    Hot slots first: tmp, y, x at low addrs; A is the cold bulk region."""
    a = _alloc()
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


# ============================================================================
# FFT
# ============================================================================

def manual_fft_iterative(N: int) -> int:
    """In-place radix-2 Cooley-Tukey on an N-slot array at low addresses."""
    a = _alloc()
    x = a.alloc(N)
    # Bit-reverse permutation — ~N/2 real swaps, 2 reads each
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a.touch(x + i); a.touch(x + j)
    # Butterflies: log2(N) stages × N/2 butterflies × 2 reads
    # (t = twiddle * x[k+j+m], then u = x[k+j] is reused; writes are free)
    m = 1
    while m < N:
        for k in range(0, N, m * 2):
            for jj in range(m):
                a.touch(x + k + jj + m)  # t = w * x[k+j+m]
                a.touch(x + k + jj)      # u = x[k+j]
        m *= 2
    return a.cost


def manual_fft_recursive(N: int) -> int:
    """Out-of-place recursive radix-2: at each level push/pop fresh
    even/odd temp arrays. Temps live briefly but drive the allocator
    pointer up during recursion."""
    a = _alloc()
    x = a.alloc(N)

    def rec(base: int, sz: int) -> None:
        if sz == 1:
            return
        ckpt = a.push()
        even = a.alloc(sz // 2)
        odd  = a.alloc(sz // 2)
        # Split: read x[2i] and x[2i+1] → write even[i], odd[i]
        for i in range(sz // 2):
            a.touch(base + 2 * i)
            a.touch(base + 2 * i + 1)
        rec(even, sz // 2)
        rec(odd,  sz // 2)
        # Combine: t = w * odd[k]; base[k] = even[k] + t; base[k+sz/2] = even[k] - t
        for k in range(sz // 2):
            a.touch(odd + k)
            a.touch(even + k)
        a.pop(ckpt)

    rec(x, N)
    return a.cost


# ============================================================================
# 2D Jacobi stencil
# ============================================================================

def manual_stencil_naive(n: int) -> int:
    """Row-major single sweep of 5-point Jacobi. 5 reads of A per
    interior cell; writes to B are free."""
    a = _alloc()
    A = a.alloc(n * n); B = a.alloc(n * n)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            a.touch(A + i * n + j)
            a.touch(A + (i - 1) * n + j)
            a.touch(A + (i + 1) * n + j)
            a.touch(A + i * n + j - 1)
            a.touch(A + i * n + j + 1)
    return a.cost


def manual_stencil_recursive(n: int, leaf: int = 8) -> int:
    """Tile-recursive 5-point Jacobi. Same set of reads as naive — in
    the fixed-placement Manhattan model the total cost is identical;
    only access ORDER differs (visible to bytedmd_classic/bytedmd_live)."""
    a = _alloc()
    A = a.alloc(n * n); B = a.alloc(n * n)

    def rec(r0: int, c0: int, sz: int) -> None:
        if sz <= leaf:
            for i in range(r0, r0 + sz):
                for j in range(c0, c0 + sz):
                    if 0 < i < n - 1 and 0 < j < n - 1:
                        a.touch(A + i * n + j)
                        a.touch(A + (i - 1) * n + j)
                        a.touch(A + (i + 1) * n + j)
                        a.touch(A + i * n + j - 1)
                        a.touch(A + i * n + j + 1)
            return
        h = sz // 2
        rec(r0,     c0,     h)
        rec(r0,     c0 + h, h)
        rec(r0 + h, c0,     h)
        rec(r0 + h, c0 + h, h)

    rec(0, 0, n)
    return a.cost


# ============================================================================
# Convolution
# ============================================================================

def manual_spatial_convolution(H: int, W: int, K: int) -> int:
    """2D single-channel convolution. Accumulator s and the K*K kernel Wk
    live at the lowest addresses (hot); the H*W image is the cold bulk.
    Per output cell: 1 accumulator read + K*K * (image read + kernel read)."""
    a = _alloc()
    s = a.alloc(1)
    Wk = a.alloc(K * K)
    img = a.alloc(H * W)
    out_h = H - K + 1
    out_w = W - K + 1
    for i in range(out_h):
        for j in range(out_w):
            a.touch(s)
            for ki in range(K):
                for kj in range(K):
                    a.touch(img + (i + ki) * W + (j + kj))
                    a.touch(Wk + ki * K + kj)
    return a.cost


def manual_fft_conv(N: int) -> int:
    """Convolution via FFT: two forward FFTs + pointwise multiply + inverse
    FFT. Arrays X, Y, Z allocated at the lowest addresses."""
    a = _alloc()
    X = a.alloc(N); Y = a.alloc(N); Z = a.alloc(N)

    def fft_in_place(base: int) -> None:
        # Bit-reverse
        j = 0
        for i in range(1, N):
            bit = N >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                a.touch(base + i); a.touch(base + j)
        # Butterflies
        m = 1
        while m < N:
            for k in range(0, N, m * 2):
                for jj in range(m):
                    a.touch(base + k + jj + m)
                    a.touch(base + k + jj)
            m *= 2

    fft_in_place(X)
    fft_in_place(Y)
    # Pointwise multiply: Z[k] = X[k] * Y[k]
    for k in range(N):
        a.touch(X + k)
        a.touch(Y + k)
    fft_in_place(Z)  # inverse FFT — same access pattern
    return a.cost


def manual_regular_convolution(H: int, W: int, K: int, Cin: int, Cout: int) -> int:
    """Full multi-channel CNN layer. Layout: s (1) | Wk (K*K*Cin*Cout) |
    img (H*W*Cin). Image channel-inner-most, kernel channel-pair inner-most.
    Per output cell: 1 accumulator read + K*K*Cin*Cout * (image + kernel)
    for each of Cout output channels."""
    a = _alloc()
    s = a.alloc(1)
    Wk = a.alloc(K * K * Cin * Cout)
    img = a.alloc(H * W * Cin)
    out_h = H - K + 1
    out_w = W - K + 1
    for i in range(out_h):
        for j in range(out_w):
            for co in range(Cout):
                a.touch(s)
                for ki in range(K):
                    for kj in range(K):
                        for ci in range(Cin):
                            a.touch(img + ((i + ki) * W + (j + kj)) * Cin + ci)
                            a.touch(Wk + ((ki * K + kj) * Cin + ci) * Cout + co)
    return a.cost


# ============================================================================
# Mergesort (recursive, out-of-place merge with push/pop stack of temps)
# ============================================================================

def manual_quicksort(N: int) -> int:
    """In-place recursive quicksort. Only the input array at addrs 1..N —
    no temp allocations (quicksort partitions in place). At each level,
    scan sz-1 elements against the pivot (2 reads each); recurse on halves."""
    a = _alloc()
    arr = a.alloc(N)

    def rec(base: int, sz: int) -> None:
        if sz <= 1:
            return
        pivot_addr = base + sz - 1
        for i in range(sz - 1):
            a.touch(base + i)
            a.touch(pivot_addr)
        mid = sz // 2
        rec(base, mid)
        rec(base + mid, sz - mid)

    rec(arr, N)
    return a.cost


def manual_heapsort(N: int) -> int:
    """In-place heapsort. Only the input array at addrs 1..N — binary-heap
    index arithmetic is in-place. Two phases: build (sift-down from n/2-1
    down to 0), then extract (N-1 iterations of root-swap + sift-down
    over a shrinking prefix). Each sift-down step reads parent+child(ren)
    at tree-linked addresses."""
    a = _alloc()
    arr = a.alloc(N)

    def sift_down(j: int, heap_size: int) -> None:
        while 2 * j + 1 < heap_size:
            child = 2 * j + 1
            if child + 1 < heap_size:
                a.touch(arr + child)
                a.touch(arr + child + 1)
            a.touch(arr + j)
            a.touch(arr + child)
            j = child

    for i in range(N // 2 - 1, -1, -1):
        sift_down(i, N)
    for k in range(N - 1, 0, -1):
        a.touch(arr + k)
        a.touch(arr + 0)
        sift_down(0, k)
    return a.cost


def manual_mergesort(N: int) -> int:
    """Recursive mergesort on an N-slot array at addr 1..N. Each merge level
    allocates a temp of the merge size (popped after copy-back). Each merge
    does 2*sz reads (both frontiers) and sz reads (copy-back temp → base)."""
    a = _alloc()
    arr = a.alloc(N)

    def rec(base: int, sz: int) -> None:
        if sz <= 1:
            if sz == 1:
                a.touch(base)
            return
        ckpt = a.push()
        half = sz // 2
        rec(base, half)
        rec(base + half, sz - half)
        # Merge: allocate temp, read both frontiers into temp
        temp = a.alloc(sz)
        for k in range(sz):
            a.touch(base + (k if k < half else half - 1))
            a.touch(base + half + (k - half if k >= half else 0))
        # Copy temp back to base
        for k in range(sz):
            a.touch(temp + k)
        a.pop(ckpt)

    rec(arr, N)
    return a.cost


# ============================================================================
# Longest Common Subsequence DP
# ============================================================================

# ============================================================================
# LU / Gaussian elimination
# ============================================================================

def manual_lu_no_pivot(n: int) -> int:
    """In-place no-pivot LU. A at addrs 1..n²; all traffic stays inside A.
    Per step k: read pivot, (n-k-1) column scales, (n-k-1)² rank-1 MACs
    (3 reads each: A[i][j], A[i][k], A[k][j])."""
    a = _alloc()
    A = a.alloc(n * n)
    for k in range(n):
        pivot_addr = A + k * n + k
        a.touch(pivot_addr)                       # read pivot
        for i in range(k + 1, n):
            a.touch(A + i * n + k)                # A[i][k]
            a.touch(pivot_addr)                   # /pivot
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a.touch(A + i * n + j)            # A[i][j]
                a.touch(A + i * n + k)            # A[i][k]
                a.touch(A + k * n + j)            # A[k][j]
    return a.cost


def manual_blocked_lu(n: int, NB: int = 8) -> int:
    """One-level blocked LU with scratchpad for the NB×NB panel/row-strip
    staging. Scratch at addrs 1..3NB²; A in the bulk region."""
    a = _alloc()
    S_diag = a.alloc(NB * NB)     # diagonal block scratchpad
    S_panel = a.alloc(NB * NB)    # below-diagonal panel scratch
    S_row = a.alloc(NB * NB)      # row strip scratch
    A = a.alloc(n * n)

    def panel_lu(base_r: int, base_c: int, sz: int, scratch: int) -> None:
        # Copy A[base_r:+sz, base_c:+sz] into scratch, factor, copy back
        for ii in range(sz):
            for jj in range(sz):
                a.touch(A + (base_r + ii) * n + base_c + jj)
        for k in range(sz):
            pivot_addr = scratch + k * sz + k
            a.touch(pivot_addr)
            for i in range(k + 1, sz):
                a.touch(scratch + i * sz + k)
                a.touch(pivot_addr)
            for i in range(k + 1, sz):
                for j in range(k + 1, sz):
                    a.touch(scratch + i * sz + j)
                    a.touch(scratch + i * sz + k)
                    a.touch(scratch + k * sz + j)
        for ii in range(sz):
            for jj in range(sz):
                a.touch(scratch + ii * sz + jj)   # flush back to A

    for kb in range(0, n, NB):
        ke = min(kb + NB, n)
        sz = ke - kb
        # (a) factor diagonal block via scratchpad
        panel_lu(kb, kb, sz, S_diag)
        # (b) update trailing panel rows — triangular solve (L11 used)
        for i in range(ke, n):
            for k in range(kb, ke):
                a.touch(A + i * n + k)
                a.touch(A + k * n + k)
                for j in range(k + 1, ke):
                    a.touch(A + i * n + j)
                    a.touch(A + i * n + k)
                    a.touch(A + k * n + j)
        # (c) update trailing row strip — triangular solve (U11 used)
        for k in range(kb, ke):
            for j in range(ke, n):
                for i in range(k + 1, ke):
                    a.touch(A + i * n + j)
                    a.touch(A + i * n + k)
                    a.touch(A + k * n + j)
        # (d) GEMM trailing update — rank-NB into scratch then back
        for i in range(ke, n):
            for j in range(ke, n):
                for k in range(kb, ke):
                    a.touch(A + i * n + j)
                    a.touch(A + i * n + k)
                    a.touch(A + k * n + j)
    return a.cost


def manual_recursive_lu(n: int) -> int:
    """Recursive LU. Top-level A at addrs 1..n². Sub-blocks are processed
    in-place — no temp allocation, only address arithmetic."""
    a = _alloc()
    A = a.alloc(n * n)

    def rec(r0: int, c0: int, sz: int) -> None:
        if sz == 1:
            return
        h = sz // 2
        # (1) Factor top-left h×h via nested recursion
        rec(r0, c0, h)
        # (2) Solve off-diag column A[r0+h:r0+sz, c0:c0+h] with U11
        for i in range(r0 + h, r0 + sz):
            for k in range(c0, c0 + h):
                a.touch(A + i * n + k)
                a.touch(A + k * n + k)
                for j in range(k + 1, c0 + h):
                    a.touch(A + i * n + j)
                    a.touch(A + i * n + k)
                    a.touch(A + k * n + j)
        # (3) Solve off-diag row A[r0:r0+h, c0+h:c0+sz] with L11
        for k in range(r0, r0 + h):
            for j in range(c0 + h, c0 + sz):
                for i in range(k + 1, r0 + h):
                    a.touch(A + i * n + j)
                    a.touch(A + i * n + k)
                    a.touch(A + k * n + j)
        # (4) Schur complement A22 -= A21 · A12
        for i in range(r0 + h, r0 + sz):
            for j in range(c0 + h, c0 + sz):
                for k in range(c0, c0 + h):
                    a.touch(A + i * n + j)
                    a.touch(A + i * n + k)
                    a.touch(A + k * n + j)
        # (5) Recurse on A22
        rec(r0 + h, c0 + h, sz - h)

    rec(0, 0, n)
    return a.cost


def manual_lu_partial_pivot(n: int) -> int:
    """LU with partial pivoting. Adds a column scan (n-k reads) per step
    and a row-swap pass touching row k and row p=(k+1) across n-k columns."""
    a = _alloc()
    A = a.alloc(n * n)
    for k in range(n):
        # (a) column scan
        for i in range(k, n):
            a.touch(A + i * n + k)
            a.touch(A + k * n + k)
        # (b) oblivious row swap with row p = k+1 (or k if k+1 >= n)
        p = k + 1 if k + 1 < n else k
        for j in range(k, n):
            a.touch(A + k * n + j)
            a.touch(A + p * n + j)
        # (c) elimination
        pivot_addr = A + k * n + k
        a.touch(pivot_addr)
        for i in range(k + 1, n):
            a.touch(A + i * n + k); a.touch(pivot_addr)
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a.touch(A + i * n + j)
                a.touch(A + i * n + k)
                a.touch(A + k * n + j)
    return a.cost


# ============================================================================
# Cholesky
# ============================================================================

def manual_cholesky(n: int) -> int:
    """Right-looking Cholesky. Lower triangle only — reads span i >= j
    only, so ~half the touches of full LU. A at addrs 1..n²."""
    a = _alloc()
    A = a.alloc(n * n)
    for k in range(n):
        pivot_addr = A + k * n + k
        a.touch(pivot_addr)   # sqrt(A[k][k]) stand-in
        for i in range(k + 1, n):
            a.touch(A + i * n + k)
            a.touch(pivot_addr)
        for j in range(k + 1, n):
            for i in range(j, n):   # lower triangle (i >= j)
                a.touch(A + i * n + j)
                a.touch(A + i * n + k)
                a.touch(A + j * n + k)
    return a.cost


# ============================================================================
# QR — Householder family
# ============================================================================

def manual_householder_qr(m: int, n: int) -> int:
    """Classical Householder QR in place. For each column k: scan
    subdiagonal (m-k reads), then reflect each trailing column (2 reads
    per dot-product entry + 2 reads per update entry)."""
    a = _alloc()
    A = a.alloc(m * n)
    for k in range(min(m, n)):
        # (a) compute reflector norm
        a.touch(A + k * n + k)
        for i in range(k + 1, m):
            a.touch(A + i * n + k)
        # (b) apply reflector to each trailing column
        for j in range(k + 1, n):
            # dot product: sum_i A[i][k] * A[i][j]
            a.touch(A + k * n + k); a.touch(A + k * n + j)
            for i in range(k + 1, m):
                a.touch(A + i * n + k); a.touch(A + i * n + j)
            # rank-1 update: A[i][j] -= t * A[i][k]
            a.touch(A + k * n + j); a.touch(A + k * n + k)
            for i in range(k + 1, m):
                a.touch(A + i * n + j); a.touch(A + i * n + k)
    return a.cost


def manual_blocked_qr(m: int, n: int, NB: int = 8) -> int:
    """Blocked QR (WY form, simplified). Panel factored via classical
    Householder; trailing columns updated in one rank-NB sweep per
    column. W-vector of length NB stored in hot region."""
    a = _alloc()
    w = a.alloc(NB)   # temp NB-vector for rank-NB update
    A = a.alloc(m * n)
    for kb in range(0, min(m, n), NB):
        ke = min(kb + NB, min(m, n))
        # (a) panel factor
        for k in range(kb, ke):
            a.touch(A + k * n + k)
            for i in range(k + 1, m):
                a.touch(A + i * n + k)
            for j in range(k + 1, ke):
                a.touch(A + k * n + k); a.touch(A + k * n + j)
                for i in range(k + 1, m):
                    a.touch(A + i * n + k); a.touch(A + i * n + j)
                a.touch(A + k * n + j); a.touch(A + k * n + k)
                for i in range(k + 1, m):
                    a.touch(A + i * n + j); a.touch(A + i * n + k)
        # (b) block apply to trailing columns
        for j in range(ke, n):
            for t_idx, k in enumerate(range(kb, ke)):
                a.touch(A + k * n + k); a.touch(A + k * n + j)
                for i in range(k + 1, m):
                    a.touch(A + i * n + k); a.touch(A + i * n + j)
                # write w[t_idx] — free
            for t_idx, k in enumerate(range(kb, ke)):
                a.touch(A + k * n + j); a.touch(A + k * n + k); a.touch(w + t_idx)
                for i in range(k + 1, m):
                    a.touch(A + i * n + j); a.touch(A + i * n + k); a.touch(w + t_idx)
    return a.cost


def manual_tsqr(m: int, n: int, block_rows: int = 8) -> int:
    """Tall-skinny QR: local Householder QR per row-tile, then pairwise
    tree-reduction over stacked R factors."""
    a = _alloc()
    A = a.alloc(m * n)
    # Phase 1: local QR per row-tile
    for row0 in range(0, m, block_rows):
        row1 = min(row0 + block_rows, m)
        for k in range(min(row1 - row0, n)):
            kk = row0 + k
            a.touch(A + kk * n + k)
            for i in range(kk + 1, row1):
                a.touch(A + i * n + k)
            for j in range(k + 1, n):
                a.touch(A + kk * n + k); a.touch(A + kk * n + j)
                for i in range(kk + 1, row1):
                    a.touch(A + i * n + k); a.touch(A + i * n + j)
                a.touch(A + kk * n + j); a.touch(A + kk * n + k)
                for i in range(kk + 1, row1):
                    a.touch(A + i * n + j); a.touch(A + i * n + k)
    # Phase 2: tree reduction
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
            for k in range(min(n, block_rows)):
                a.touch(A + (left_row + k) * n + k)
                a.touch(A + (right_row + k) * n + k)
                for j in range(k + 1, n):
                    a.touch(A + (left_row + k) * n + k); a.touch(A + (left_row + k) * n + j)
                    for i in range(right_row + k, right_end):
                        a.touch(A + i * n + k); a.touch(A + i * n + j)
                    a.touch(A + (left_row + k) * n + j); a.touch(A + (left_row + k) * n + k)
                    for i in range(right_row + k, right_end):
                        a.touch(A + i * n + j); a.touch(A + i * n + k)
        stride *= 2
    return a.cost


# ============================================================================
# Longest Common Subsequence DP
# ============================================================================

def manual_lcs_dp(m: int, n: int) -> int:
    """Row-major LCS DP. Cell (i,j) reads D[i-1][j-1], D[i-1][j], D[i][j-1]
    and the two input characters x[i-1], y[j-1]."""
    a = _alloc()
    # Strings at low addrs (repeatedly touched), DP table at higher addrs
    x = a.alloc(m); y = a.alloc(n)
    D = a.alloc((m + 1) * (n + 1))
    stride = n + 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            a.touch(D + (i - 1) * stride + (j - 1))
            a.touch(D + (i - 1) * stride + j)
            a.touch(D + i * stride + (j - 1))
            a.touch(x + i - 1)
            a.touch(y + j - 1)
    return a.cost
