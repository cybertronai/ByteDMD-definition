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
    """Bump-pointer allocator with two independent address spaces.

    Two stacks run in parallel, each with its own bump pointer and its own
    Manhattan-disc origin at address 1:
      - Arg stack (alloc_arg / touch_arg): holds inputs, read-only.
      - Scratch stack (alloc / touch): holds temporaries and outputs.
    Both price reads as ceil(sqrt(addr)). touch() and touch_arg() record
    log entries as (space, addr) pairs where space ∈ {"scratch", "arg"}.

    write(addr) is FREE under the ByteDMD convention (no cost added)
    but when logging=True is recorded into one of two lists:
      .output_writes : writes whose addr falls inside [out_start, out_end)
                       (as declared by set_output_range). Plotted shifted
                       above peak in trace visualizations.
      .writes        : everything else (scratch / temp writes). Plotted
                       at real address.
    Both store (time, addr) pairs, time = len(self.log).
    """
    __slots__ = ("cost", "ptr", "peak", "arg_ptr", "arg_peak",
                 "log", "writes", "output_writes", "out_start", "out_end")

    def __init__(self, logging: bool = False) -> None:
        self.cost = 0
        self.ptr = 1         # scratch bump pointer
        self.peak = 1        # scratch peak
        self.arg_ptr = 1     # arg bump pointer
        self.arg_peak = 1    # arg peak
        self.log = [] if logging else None             # list of (space, addr)
        self.writes = [] if logging else None
        self.output_writes = [] if logging else None
        self.out_start = None
        self.out_end = None

    def alloc(self, size: int) -> int:
        """Allocate `size` cells on the scratch stack."""
        addr = self.ptr
        self.ptr += size
        if self.ptr > self.peak:
            self.peak = self.ptr
        return addr

    def alloc_arg(self, size: int) -> int:
        """Allocate `size` cells on the argument stack (read-only inputs)."""
        addr = self.arg_ptr
        self.arg_ptr += size
        if self.arg_ptr > self.arg_peak:
            self.arg_peak = self.arg_ptr
        return addr

    def push(self) -> int:
        return self.ptr

    def pop(self, p: int) -> None:
        self.ptr = p

    def set_output_range(self, start: int, end: int) -> None:
        """Mark [start, end) as the algorithm's output region (on the
        scratch stack). Writes to addresses in this range get classified
        as output_writes (shifted above peak in the plot); writes outside
        are treated as scratch."""
        self.out_start = start
        self.out_end = end

    def touch(self, addr: int) -> None:
        """Read from the scratch stack at `addr`."""
        self.cost += math.isqrt(max(0, addr - 1)) + 1
        if self.log is not None:
            self.log.append(("scratch", addr))

    def touch_arg(self, addr: int) -> None:
        """Read from the argument stack at `addr`."""
        self.cost += math.isqrt(max(0, addr - 1)) + 1
        if self.log is not None:
            self.log.append(("arg", addr))

    def write(self, addr: int) -> None:
        if self.writes is None:
            return
        t = len(self.log)
        if (self.out_start is not None
                and self.out_start <= addr < self.out_end):
            self.output_writes.append((t, addr))
        else:
            self.writes.append((t, addr))

    def read_output(self) -> None:
        """Epilogue: the program must return its output, so every output
        cell is read once more at the end from the scratch stack. Priced
        like any other scratch read, but tagged in the log as "output"
        so visualizations can distinguish this final pass."""
        if self.out_start is None or self.out_end is None:
            return
        for addr in range(self.out_start, self.out_end):
            self.cost += math.isqrt(max(0, addr - 1)) + 1
            if self.log is not None:
                self.log.append(("output", addr))


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
    """Naive triple loop C[i][j] = Σₖ A[i][k] · B[j][k], with the
    current A-row hoisted into a hot scratchpad.

    Because A[i][*] is reused across all n values of j (for fixed i),
    preloading it into `c_A_row` once per outer i iteration cuts n−1
    redundant arg-stack reads per A cell down to zero. B[j][*] isn't
    cached (it would need to be reloaded for each i, wiping the win).

      s       (addr 1)         — accumulator
      c_A_row (addrs 2..n+1)   — hot A[i][*] row buffer
      C       (addrs n+2..)    — output"""
    a = _alloc()
    A = a.alloc_arg(n * n); B = a.alloc_arg(n * n)
    s = a.alloc(1)
    c_A_row = a.alloc(n)
    C = a.alloc(n * n)
    a.set_output_range(C, C + n * n)
    for i in range(n):
        # Load A[i][*] into c_A_row once per outer i.
        for k in range(n):
            a.touch_arg(A + i * n + k); a.write(c_A_row + k)
        for j in range(n):
            a.touch(s)
            for k in range(n):
                a.touch(c_A_row + k)
                a.touch_arg(B + j * n + k)
            a.write(C + i * n + j)
    a.read_output()
    return a.cost


def manual_tiled_matmul(n: int, T: int | None = None) -> int:
    """One-level blocked matmul. A, B on arg stack (read once per tile-load
    into scratch sA, sB). sA, sB, sC and output C on scratch."""
    if T is None:
        T = max(1, int(round(n ** 0.5)))
    a = _alloc()
    A = a.alloc_arg(n * n); B = a.alloc_arg(n * n)
    sA = a.alloc(T * T); sB = a.alloc(T * T); sC = a.alloc(T * T)
    C = a.alloc(n * n)
    a.set_output_range(C, C + n * n)

    for bi in range(0, n, T):
        for bj in range(0, n, T):
            # Load C tile into sC (read C, write sC)
            for ii in range(min(T, n - bi)):
                for jj in range(min(T, n - bj)):
                    a.touch(C + (bi + ii) * n + (bj + jj))
                    a.write(sC + ii * T + jj)
            for bk in range(0, n, T):
                # Load A tile into sA
                for ii in range(min(T, n - bi)):
                    for kk in range(min(T, n - bk)):
                        a.touch_arg(A + (bi + ii) * n + (bk + kk))
                        a.write(sA + ii * T + kk)
                # Load B tile into sB
                for kk in range(min(T, n - bk)):
                    for jj in range(min(T, n - bj)):
                        a.touch_arg(B + (bk + kk) * n + (bj + jj))
                        a.write(sB + kk * T + jj)
                # MAC: accumulate into sC (sC write per (ii,jj))
                for ii in range(min(T, n - bi)):
                    for jj in range(min(T, n - bj)):
                        a.touch(sC + ii * T + jj)
                        for kk in range(min(T, n - bk)):
                            a.touch(sA + ii * T + kk)
                            a.touch(sB + kk * T + jj)
                        a.write(sC + ii * T + jj)
            # Flush sC -> C (read sC, write C)
            for ii in range(min(T, n - bi)):
                for jj in range(min(T, n - bj)):
                    a.touch(sC + ii * T + jj)
                    a.write(C + (bi + ii) * n + (bj + jj))
    a.read_output()
    return a.cost


def manual_rmm(n: int, T: int = 4) -> int:
    """Recursive (cache-oblivious) matmul with T-tile scratchpad at leaves.
    A, B on arg stack; sA/sB/sC/C on scratch. MAC convention: fast_C
    read once per (i,j) outside k-loop; bulk read of fast_C before MAC;
    on first write of a C-tile, skip pre-load of pC (fresh)."""
    a = _alloc()
    A = a.alloc_arg(n * n); B = a.alloc_arg(n * n)
    sA = a.alloc(T * T); sB = a.alloc(T * T); sC = a.alloc(T * T)
    C = a.alloc(n * n)
    a.set_output_range(C, C + n * n)
    # is_first is checked against the PREVIOUS compute_tile's C coordinate
    # only — if the C tile loaded in fast_C still matches, skip the pC
    # pre-load. Matches strassen_trace's Scratchpad.sync cache semantic.
    last_C = [None]

    def compute_tile(rA: int, cA: int, rB: int, cB: int, rC: int, cC: int) -> None:
        is_first = (last_C[0] != (rC, cC))
        last_C[0] = (rC, cC)
        # Load A, B tiles (read from arg stack, write to scratch)
        for ii in range(T):
            for jj in range(T):
                a.touch_arg(A + (rA + ii) * n + cA + jj)
                a.write(sA + ii * T + jj)
        for ii in range(T):
            for jj in range(T):
                a.touch_arg(B + (rB + ii) * n + cB + jj)
                a.write(sB + ii * T + jj)
        # Bulk read of fast_C (accumulator init)
        for i in range(T * T):
            a.touch(sC + i)
        # MAC
        for ii in range(T):
            for jj in range(T):
                a.touch(sC + ii * T + jj)
                for kk in range(T):
                    a.touch(sA + ii * T + kk)
                    a.touch(sB + kk * T + jj)
                a.write(sC + ii * T + jj)
        # Flush fast_C -> pC
        for ii in range(T):
            for jj in range(T):
                a.touch(sC + ii * T + jj)
                if not is_first:
                    a.touch(C + (rC + ii) * n + cC + jj)
                a.write(C + (rC + ii) * n + cC + jj)

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
    a.read_output()
    return a.cost


def manual_strassen(n: int, T: int = 4) -> int:
    """Standard Strassen. A, B on arg stack (at top); SA/SB/M/sA/sB/sC/C on
    scratch. Recursion threads is_arg flags through since sub-products
    flip pointers back to scratch buffers (M-intermediates)."""
    a = _alloc()
    A = a.alloc_arg(n * n); B = a.alloc_arg(n * n)
    sA = a.alloc(T * T); sB = a.alloc(T * T); sC = a.alloc(T * T)
    C = a.alloc(n * n)
    a.set_output_range(C, C + n * n)

    def _t(addr: int, is_arg: bool) -> None:
        (a.touch_arg if is_arg else a.touch)(addr)

    def add_mats(p1: int, s1: int, is1a: bool,
                 p2: int, s2: int, is2a: bool, h: int) -> None:
        for i in range(h):
            for j in range(h):
                _t(p1 + i * s1 + j, is1a)
                _t(p2 + i * s2 + j, is2a)

    def compute_tile(pA: int, sAstr: int, isAarg: bool,
                     pB: int, sBstr: int, isBarg: bool,
                     pC: int, sCstr: int) -> None:
        for ii in range(T):
            for jj in range(T):
                _t(pA + ii * sAstr + jj, isAarg)
                a.write(sA + ii * T + jj)
        for ii in range(T):
            for jj in range(T):
                _t(pB + ii * sBstr + jj, isBarg)
                a.write(sB + ii * T + jj)
        for ii in range(T):
            for jj in range(T):
                a.touch(pC + ii * sCstr + jj)
                a.write(sC + ii * T + jj)
        for ii in range(T):
            for jj in range(T):
                a.touch(sC + ii * T + jj)
                for kk in range(T):
                    a.touch(sA + ii * T + kk)
                    a.touch(sB + kk * T + jj)
                a.write(sC + ii * T + jj)
        for ii in range(T):
            for jj in range(T):
                a.touch(sC + ii * T + jj)
                a.write(pC + ii * sCstr + jj)

    def recurse(pA_: int, sAstr: int, isAa: bool,
                pB_: int, sBstr: int, isBa: bool,
                pC_: int, sCstr: int, sz: int) -> None:
        if sz <= T:
            compute_tile(pA_, sAstr, isAa, pB_, sBstr, isBa, pC_, sCstr)
            return
        h = sz // 2
        ckpt = a.push()
        SA = a.alloc(h * h); SB = a.alloc(h * h)
        M = [a.alloc(h * h) for _ in range(7)]

        A11, A12 = pA_, pA_ + h
        A21, A22 = pA_ + h * sAstr, pA_ + h * sAstr + h
        B11, B12 = pB_, pB_ + h
        B21, B22 = pB_ + h * sBstr, pB_ + h * sBstr + h

        add_mats(A11, sAstr, isAa, A22, sAstr, isAa, h)
        add_mats(B11, sBstr, isBa, B22, sBstr, isBa, h)
        recurse(SA, h, False, SB, h, False, M[0], h, h)

        add_mats(A21, sAstr, isAa, A22, sAstr, isAa, h)
        recurse(SA, h, False, B11, sBstr, isBa, M[1], h, h)

        add_mats(B12, sBstr, isBa, B22, sBstr, isBa, h)
        recurse(A11, sAstr, isAa, SB, h, False, M[2], h, h)

        add_mats(B21, sBstr, isBa, B11, sBstr, isBa, h)
        recurse(A22, sAstr, isAa, SB, h, False, M[3], h, h)

        add_mats(A11, sAstr, isAa, A12, sAstr, isAa, h)
        recurse(SA, h, False, B22, sBstr, isBa, M[4], h, h)

        add_mats(A21, sAstr, isAa, A11, sAstr, isAa, h)
        add_mats(B11, sBstr, isBa, B12, sBstr, isBa, h)
        recurse(SA, h, False, SB, h, False, M[5], h, h)

        add_mats(A12, sAstr, isAa, A22, sAstr, isAa, h)
        add_mats(B21, sBstr, isBa, B22, sBstr, isBa, h)
        recurse(SA, h, False, SB, h, False, M[6], h, h)

        # Assemble C quadrants. M[*] and pC_ are both scratch.
        def read_M(*indices: int) -> None:
            for i in range(h):
                for j in range(h):
                    for idx in indices:
                        a.touch(M[idx] + i * h + j)

        def write_quadrant(qr: int, qc: int) -> None:
            for i in range(h):
                for j in range(h):
                    a.write(pC_ + (qr + i) * sCstr + (qc + j))

        read_M(0, 3, 4, 6); write_quadrant(0, 0)
        read_M(2, 4);        write_quadrant(0, h)
        read_M(1, 3);        write_quadrant(h, 0)
        read_M(0, 1, 2, 5);  write_quadrant(h, h)

        a.pop(ckpt)

    recurse(A, n, True, B, n, True, C, n, n)
    a.read_output()
    return a.cost


def manual_fused_strassen(n: int, T: int = 4) -> int:
    """Zero-Allocation Fused Strassen (ZAFS): A, B on arg stack; fast
    scratchpads + output C on scratch. Sub-additions are fused into the
    L1 tile loads. fast_C read once per (i,j) outside k-loop."""
    a = _alloc()
    A = a.alloc_arg(n * n); B = a.alloc_arg(n * n)
    fast_A = a.alloc(T * T); fast_B = a.alloc(T * T); fast_C = a.alloc(T * T)
    C = a.alloc(n * n)
    a.set_output_range(C, C + n * n)

    def compute_fused_tile(ops_A, ops_B, ops_C, r, c, k_off):
        # 1. Fused load A tile (arg) into fast_A (scratch)
        for i in range(T):
            for j in range(T):
                for _sgn, rb, cb in ops_A:
                    a.touch_arg(A + (rb + r + i) * n + cb + k_off + j)
                a.write(fast_A + i * T + j)
        # 2. Fused load B tile (arg) into fast_B (scratch)
        for i in range(T):
            for j in range(T):
                for _sgn, rb, cb in ops_B:
                    a.touch_arg(B + (rb + k_off + i) * n + cb + c + j)
                a.write(fast_B + i * T + j)
        # 3. Bulk read of fast_C (accumulator init)
        for i in range(T * T):
            a.touch(fast_C + i)
        # 4. Tile MAC
        for i in range(T):
            for j in range(T):
                a.touch(fast_C + i * T + j)
                for k in range(T):
                    a.touch(fast_A + i * T + k)
                    a.touch(fast_B + k * T + j)
                a.write(fast_C + i * T + j)
        # 5. Fan-out fast_C -> multiple C targets with signs
        for _sgn, rb, cb, is_first in ops_C:
            for i in range(T):
                for j in range(T):
                    a.touch(fast_C + i * T + j)
                    if not is_first:
                        a.touch(C + (rb + r + i) * n + cb + c + j)
                    a.write(C + (rb + r + i) * n + cb + c + j)

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
    a.read_output()
    return a.cost


# ============================================================================
# Attention family
# ============================================================================

def manual_naive_attention(N: int, d: int) -> int:
    """Three stages: S = Q K^T, P = softmax(S), O = P V.
    Q, K, V on arg stack; S (full N×N) and O on scratch; hot scalars too."""
    a = _alloc()
    Q = a.alloc_arg(N * d); K = a.alloc_arg(N * d); V = a.alloc_arg(N * d)
    # Hot scratch scalars at low addresses
    s_acc = a.alloc(1); tmp = a.alloc(1)
    row_max = a.alloc(1); row_sum = a.alloc(1); inv_sum = a.alloc(1)
    S = a.alloc(N * N)
    O = a.alloc(N * d)
    a.set_output_range(O, O + N * d)

    # Stage 1: S[i][j] = Q[i] . K[j] (scale folded in)
    for i in range(N):
        for j in range(N):
            a.touch_arg(Q + i * d + 0); a.touch_arg(K + j * d + 0)
            for dd in range(1, d):
                a.touch_arg(Q + i * d + dd); a.touch_arg(K + j * d + dd)
                a.touch(s_acc); a.touch(tmp)
            a.touch(s_acc)
            a.write(S + i * N + j)  # write S[i][j]

    # Stage 2: row-wise softmax (in place on S, becomes P)
    for i in range(N):
        a.touch(S + i * N + 0)
        for j in range(1, N):
            a.touch(S + i * N + j); a.touch(row_max)
        for j in range(N):
            a.touch(S + i * N + j); a.touch(row_max)
            a.write(S + i * N + j)  # exp -> P[i][j] (reuse S storage)
            if j == 0:
                pass
            else:
                a.touch(row_sum); a.touch(S + i * N + j)
        a.touch(row_sum)
        for j in range(N):
            a.touch(S + i * N + j); a.touch(inv_sum)
            a.write(S + i * N + j)  # normalized P[i][j]

    # Stage 3: O[i][dd] = sum_j P[i][j] * V[j][dd]
    P = S
    for i in range(N):
        for dd in range(d):
            a.touch(P + i * N + 0); a.touch_arg(V + 0 * d + dd)
            for j in range(1, N):
                a.touch(P + i * N + j); a.touch_arg(V + j * d + dd)
                a.touch(s_acc); a.touch(tmp)
            a.touch(s_acc)
            a.write(O + i * d + dd)
    a.read_output()
    return a.cost


def manual_flash_attention(N: int, d: int, Bk: int) -> int:
    """Flash attention: Q, K, V on arg stack; hot scalars, block scratchpads,
    and O on scratch. Never materializes full N×N S."""
    a = _alloc()
    Q = a.alloc_arg(N * d); K = a.alloc_arg(N * d); V = a.alloc_arg(N * d)
    # Hot scalars
    m_i = a.alloc(1); l_i = a.alloc(1)
    m_block = a.alloc(1); l_block = a.alloc(1)
    m_new = a.alloc(1); alpha = a.alloc(1); beta = a.alloc(1)
    inv_l = a.alloc(1); tmp = a.alloc(1)
    # Small block scratchpads
    s_block = a.alloc(Bk); p_block = a.alloc(Bk)
    o_acc = a.alloc(d)
    O = a.alloc(N * d)
    a.set_output_range(O, O + N * d)

    num_blocks = (N + Bk - 1) // Bk

    for i in range(N):
        for kb in range(num_blocks):
            k0 = kb * Bk
            bs = min(Bk, N - k0)
            # s_block[j] = Q[i] . K[k0+j]
            for j in range(bs):
                a.touch_arg(Q + i * d + 0); a.touch_arg(K + (k0 + j) * d + 0)
                for dd in range(1, d):
                    a.touch_arg(Q + i * d + dd); a.touch_arg(K + (k0 + j) * d + dd)
                    a.touch(s_block + j); a.touch(tmp)
                a.touch(s_block + j)
                a.write(s_block + j)
            for j in range(1, bs):
                a.touch(s_block + j); a.touch(m_block)
            for j in range(bs):
                a.touch(s_block + j); a.touch(m_block)
                a.write(p_block + j)
                if j > 0:
                    a.touch(p_block + j); a.touch(l_block)
            if kb == 0:
                a.touch(m_block); a.touch(l_block)
                a.write(m_i); a.write(l_i)
                for dd in range(d):
                    a.touch(p_block + 0); a.touch_arg(V + k0 * d + dd)
                    for j in range(1, bs):
                        a.touch(p_block + j); a.touch_arg(V + (k0 + j) * d + dd)
                        a.touch(o_acc + dd); a.touch(tmp)
                    a.write(o_acc + dd)
            else:
                a.touch(m_i); a.touch(m_block); a.write(m_new)
                a.touch(m_i); a.touch(m_new); a.write(alpha)
                a.touch(m_block); a.touch(m_new); a.write(beta)
                a.touch(alpha); a.touch(l_i)
                a.touch(beta); a.touch(l_block)
                a.touch(tmp); a.touch(tmp); a.write(l_i); a.write(m_i)
                for dd in range(d):
                    a.touch(alpha); a.touch(o_acc + dd)
                    for j in range(bs):
                        a.touch(p_block + j); a.touch_arg(V + (k0 + j) * d + dd)
                        a.touch(beta); a.touch(tmp); a.touch(o_acc + dd)
                    a.write(o_acc + dd)
        # Final: O[i][dd] = o_acc[dd] / l_i
        a.touch(l_i)
        for dd in range(d):
            a.touch(o_acc + dd); a.touch(inv_l)
            a.write(O + i * d + dd)
    a.read_output()
    return a.cost


# ============================================================================
# Transpose family
# ============================================================================

def manual_transpose_naive(n: int) -> int:
    """B[i][j] = A[j][i]. A on arg stack, B on scratch."""
    a = _alloc()
    A = a.alloc_arg(n * n)
    B = a.alloc(n * n)
    a.set_output_range(B, B + n * n)
    for i in range(n):
        for j in range(n):
            a.touch_arg(A + j * n + i)
            a.write(B + i * n + j)
    a.read_output()
    return a.cost


def manual_transpose_blocked(n: int, T: int | None = None) -> int:
    """Blocked iteration order over A (arg stack); B on scratch."""
    if T is None:
        T = max(1, int(round(n ** 0.5)))
    a = _alloc()
    A = a.alloc_arg(n * n)
    B = a.alloc(n * n)
    a.set_output_range(B, B + n * n)
    for bi in range(0, n, T):
        for bj in range(0, n, T):
            for ii in range(min(T, n - bi)):
                for jj in range(min(T, n - bj)):
                    a.touch_arg(A + (bj + jj) * n + (bi + ii))
                    a.write(B + (bi + ii) * n + (bj + jj))
    a.read_output()
    return a.cost


def manual_transpose_recursive(n: int) -> int:
    """Cache-oblivious transpose: recursively split into 4 quadrants.
    A on arg stack, B on scratch."""
    a = _alloc()
    A = a.alloc_arg(n * n)
    B = a.alloc(n * n)
    a.set_output_range(B, B + n * n)

    def rec(ar: int, ac: int, br: int, bc: int, sz: int) -> None:
        if sz == 1:
            a.touch_arg(A + ar * n + ac)
            a.write(B + br * n + bc)
            return
        h = sz // 2
        rec(ar,     ac,     br,     bc,     h)
        rec(ar + h, ac,     br,     bc + h, h)
        rec(ar,     ac + h, br + h, bc,     h)
        rec(ar + h, ac + h, br + h, bc + h, h)

    rec(0, 0, 0, 0, n)
    a.read_output()
    return a.cost


# ============================================================================
# Matrix-vector
# ============================================================================

def manual_matvec_row(n: int) -> int:
    """y[i] = sum_j A[i][j] * x[j] with x preloaded into a hot scratch
    buffer at the very bottom of the stack.

    Because the Python signature is `matvec(A, x)`, `x` sits at the
    *end* of the arg stack (addrs n²+1..n²+n). Each x[j] is re-read
    n times from those high arg addresses in the original. Preloading
    x into `c_X` at addrs n+3..2n+2 cuts every subsequent x access to
    near-top-of-scratch cost.

      s, tmp  (addrs 1-2)        — accumulator + tmp
      c_X     (addrs 3..n+2)     — hot x buffer (one-time preload)
      y       (addrs n+3..2n+2)  — output"""
    a = _alloc()
    A = a.alloc_arg(n * n); x = a.alloc_arg(n)
    s = a.alloc(1); tmp = a.alloc(1)
    c_X = a.alloc(n)
    y = a.alloc(n)
    a.set_output_range(y, y + n)
    # Preload x once from arg into c_X.
    for j in range(n):
        a.touch_arg(x + j); a.write(c_X + j)
    for i in range(n):
        a.touch_arg(A + i * n + 0); a.touch(c_X + 0)
        a.write(s)
        for j in range(1, n):
            a.touch_arg(A + i * n + j); a.touch(c_X + j)
            a.touch(s); a.touch(tmp)
            a.write(s)
        a.touch(s)
        a.write(y + i)
    a.read_output()
    return a.cost


def manual_matvec_blocked(n: int, B: int = 4) -> int:
    """Blocked matvec. A and x_main on arg stack; B accumulators,
    x_tile scratchpad, tmp and y on scratch. x_tile reuses an x-slice
    across B rows — A's per-cell arg-read cost is unchanged from
    matvec_row."""
    a = _alloc()
    A = a.alloc_arg(n * n); x_main = a.alloc_arg(n)
    s = [a.alloc(1) for _ in range(B)]
    x_tile = a.alloc(B)
    tmp = a.alloc(1)
    y = a.alloc(n)
    a.set_output_range(y, y + n)

    for i_out in range(0, n, B):
        for j_out in range(0, n, B):
            for j in range(B):
                a.touch_arg(x_main + j_out + j)
                a.write(x_tile + j)
            for i in range(B):
                for j in range(B):
                    a.touch_arg(A + (i_out + i) * n + (j_out + j))
                    a.touch(x_tile + j)
                    if j_out != 0 or j != 0:
                        a.touch(s[i])
                    a.touch(tmp)
                    a.write(s[i])
        for i in range(B):
            a.touch(s[i])
            a.write(y + i_out + i)
    a.read_output()
    return a.cost


def manual_matvec_col(n: int) -> int:
    """Outer loop over j: y[i] += A[i][j] * x[j]. A, x on arg stack;
    tmp and y on scratch."""
    a = _alloc()
    A = a.alloc_arg(n * n); x = a.alloc_arg(n)
    tmp = a.alloc(1)
    y = a.alloc(n)
    a.set_output_range(y, y + n)
    for j in range(n):
        a.touch_arg(x + j)
        for i in range(n):
            a.touch_arg(A + i * n + j)
            if j == 0:
                a.write(y + i)  # init
            else:
                a.touch(y + i); a.touch(tmp)
                a.write(y + i)
    a.read_output()
    return a.cost


# ============================================================================
# FFT
# ============================================================================

def manual_fft_iterative(N: int) -> int:
    """In-place radix-2 Cooley-Tukey. Input on arg stack is copied once
    to the scratch-side x buffer (the output); all butterflies then run
    on scratch."""
    a = _alloc()
    x_in = a.alloc_arg(N)
    x = a.alloc(N)
    a.set_output_range(x, x + N)
    # Load input from arg stack into scratch.
    for i in range(N):
        a.touch_arg(x_in + i)
        a.write(x + i)
    # Bit-reverse permutation — swaps
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a.touch(x + i); a.touch(x + j)
            a.write(x + i); a.write(x + j)
    # Butterflies: each writes 2 cells of x
    m = 1
    while m < N:
        for k in range(0, N, m * 2):
            for jj in range(m):
                a.touch(x + k + jj + m)
                a.touch(x + k + jj)
                a.write(x + k + jj)
                a.write(x + k + jj + m)
        m *= 2
    a.read_output()
    return a.cost


def manual_fft_recursive(N: int) -> int:
    """In-place recursive radix-2 Cooley-Tukey. Input on arg stack is
    evaluated directly into the target output array x via strided reads,
    avoiding fresh even/odd temps on the scratch stack.

    Leaves route arg-stack cells straight into their bit-reversed slots
    inside a single N-wide scratch buffer; every butterfly combination
    then operates purely in-place. Peak scratch footprint is exactly N,
    and the geometric cost evaluates over addresses 1..N only."""
    a = _alloc()
    x_in = a.alloc_arg(N)
    x = a.alloc(N)
    a.set_output_range(x, x + N)

    def rec(base: int, sz: int, stride: int, offset: int) -> None:
        if sz == 1:
            a.touch_arg(offset)
            a.write(base)
            return
        rec(base, sz // 2, stride * 2, offset)
        rec(base + sz // 2, sz // 2, stride * 2, offset + stride)
        for k in range(sz // 2):
            a.touch(base + sz // 2 + k)   # odd component
            a.touch(base + k)             # even component
            a.write(base + k)             # overwrite even
            a.write(base + sz // 2 + k)   # overwrite odd

    rec(x, N, 1, x_in)
    a.read_output()
    return a.cost


# ============================================================================
# 2D Jacobi stencil
# ============================================================================

def manual_stencil_naive(n: int) -> int:
    """Row-major single sweep of 5-point Jacobi with rolling 3-row
    buffer. Each A cell is read exactly once from the arg stack (via a
    streaming preload of one row at a time); all 5 stencil reads
    thereafter hit the rolling buffer at addresses 1..3n.

      r0, r1, r2 (addrs 1..3n)    — rolling 3-row buffer, rotated via
                                    (i-1)%3, i%3, (i+1)%3 indexing
      B          (addrs 3n+1..)   — output matrix"""
    a = _alloc()
    A = a.alloc_arg(n * n)
    r0 = a.alloc(n)
    r1 = a.alloc(n)
    r2 = a.alloc(n)
    rows = [r0, r1, r2]
    B = a.alloc(n * n)
    a.set_output_range(B, B + n * n)

    # Initial preload: rows 0, 1, 2.
    for row in range(min(3, n)):
        slot = rows[row % 3]
        for j in range(n):
            a.touch_arg(A + row * n + j); a.write(slot + j)

    for i in range(1, n - 1):
        up, cur, down = rows[(i - 1) % 3], rows[i % 3], rows[(i + 1) % 3]
        for j in range(1, n - 1):
            a.touch(cur + j)        # center
            a.touch(up + j)         # north
            a.touch(down + j)       # south
            a.touch(cur + j - 1)    # west
            a.touch(cur + j + 1)    # east
            a.write(B + i * n + j)
        # Stream the next row into the slot we no longer need.
        if i + 2 < n:
            replace = rows[(i - 1) % 3]
            for j in range(n):
                a.touch_arg(A + (i + 2) * n + j); a.write(replace + j)
    a.read_output()
    return a.cost


def manual_stencil_recursive(n: int, leaf: int = 8) -> int:
    """Tile-recursive 5-point Jacobi. A on arg stack, B on scratch.
    Same read set as naive — only access order differs (visible to the
    LRU heuristics)."""
    a = _alloc()
    A = a.alloc_arg(n * n)
    B = a.alloc(n * n)
    a.set_output_range(B, B + n * n)

    def rec(r0: int, c0: int, sz: int) -> None:
        if sz <= leaf:
            for i in range(r0, r0 + sz):
                for j in range(c0, c0 + sz):
                    if 0 < i < n - 1 and 0 < j < n - 1:
                        a.touch_arg(A + i * n + j)
                        a.touch_arg(A + (i - 1) * n + j)
                        a.touch_arg(A + (i + 1) * n + j)
                        a.touch_arg(A + i * n + j - 1)
                        a.touch_arg(A + i * n + j + 1)
                        a.write(B + i * n + j)
            return
        h = sz // 2
        rec(r0,     c0,     h)
        rec(r0,     c0 + h, h)
        rec(r0 + h, c0,     h)
        rec(r0 + h, c0 + h, h)

    rec(0, 0, n)
    a.read_output()
    return a.cost


# ============================================================================
# Convolution
# ============================================================================

def manual_spatial_convolution(H: int, W: int, K: int) -> int:
    """2D single-channel convolution. img and Wk on arg stack;
    accumulator s and output O on scratch."""
    a = _alloc()
    Wk = a.alloc_arg(K * K)
    img = a.alloc_arg(H * W)
    s = a.alloc(1)
    out_h = H - K + 1
    out_w = W - K + 1
    O = a.alloc(out_h * out_w)
    a.set_output_range(O, O + out_h * out_w)
    for i in range(out_h):
        for j in range(out_w):
            a.touch(s)
            for ki in range(K):
                for kj in range(K):
                    a.touch_arg(img + (i + ki) * W + (j + kj))
                    a.touch_arg(Wk + ki * K + kj)
            a.write(O + i * out_w + j)
    a.read_output()
    return a.cost


def manual_fft_conv(N: int) -> int:
    """Convolution via FFT. Inputs X_in, Y_in on arg stack; X, Y (working
    copies) and Z (output) on scratch. Preload X_in→X, Y_in→Y."""
    a = _alloc()
    X_in = a.alloc_arg(N); Y_in = a.alloc_arg(N)
    X = a.alloc(N); Y = a.alloc(N); Z = a.alloc(N)
    a.set_output_range(Z, Z + N)
    for i in range(N):
        a.touch_arg(X_in + i); a.write(X + i)
        a.touch_arg(Y_in + i); a.write(Y + i)

    def fft_in_place(base: int) -> None:
        j = 0
        for i in range(1, N):
            bit = N >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                a.touch(base + i); a.touch(base + j)
                a.write(base + i); a.write(base + j)
        m = 1
        while m < N:
            for k in range(0, N, m * 2):
                for jj in range(m):
                    a.touch(base + k + jj + m)
                    a.touch(base + k + jj)
                    a.write(base + k + jj)
                    a.write(base + k + jj + m)
            m *= 2

    fft_in_place(X)
    fft_in_place(Y)
    for k in range(N):
        a.touch(X + k)
        a.touch(Y + k)
        a.write(Z + k)
    fft_in_place(Z)
    a.read_output()
    return a.cost


def manual_regular_convolution(H: int, W: int, K: int, Cin: int, Cout: int) -> int:
    """Full multi-channel CNN layer. img and Wk on arg stack;
    accumulator s and output O on scratch."""
    a = _alloc()
    Wk = a.alloc_arg(K * K * Cin * Cout)
    img = a.alloc_arg(H * W * Cin)
    s = a.alloc(1)
    out_h = H - K + 1
    out_w = W - K + 1
    O = a.alloc(out_h * out_w * Cout)
    a.set_output_range(O, O + out_h * out_w * Cout)
    for i in range(out_h):
        for j in range(out_w):
            for co in range(Cout):
                a.touch(s)
                for ki in range(K):
                    for kj in range(K):
                        for ci in range(Cin):
                            a.touch_arg(img + ((i + ki) * W + (j + kj)) * Cin + ci)
                            a.touch_arg(Wk + ((ki * K + kj) * Cin + ci) * Cout + co)
                a.write(O + (i * out_w + j) * Cout + co)
    a.read_output()
    return a.cost


# ============================================================================
# Mergesort (recursive, out-of-place merge with push/pop stack of temps)
# ============================================================================

def manual_quicksort(N: int) -> int:
    """In-place recursive quicksort. Input copied from arg stack into the
    scratch array, then partitioned in place."""
    a = _alloc()
    arr_in = a.alloc_arg(N)
    arr = a.alloc(N)
    a.set_output_range(arr, arr + N)
    for i in range(N):
        a.touch_arg(arr_in + i); a.write(arr + i)

    def rec(base: int, sz: int) -> None:
        if sz <= 1:
            return
        pivot_addr = base + sz - 1
        for i in range(sz - 1):
            a.touch(base + i)
            a.touch(pivot_addr)
            a.write(base + i)  # swap-write during partition
        mid = sz // 2
        rec(base, mid)
        rec(base + mid, sz - mid)

    rec(arr, N)
    a.read_output()
    return a.cost


def manual_heapsort(N: int) -> int:
    """In-place heapsort. Input copied from arg stack to scratch, then
    binary-heap index arithmetic runs in place on scratch."""
    a = _alloc()
    arr_in = a.alloc_arg(N)
    arr = a.alloc(N)
    a.set_output_range(arr, arr + N)
    for i in range(N):
        a.touch_arg(arr_in + i); a.write(arr + i)

    def sift_down(j: int, heap_size: int) -> None:
        while 2 * j + 1 < heap_size:
            child = 2 * j + 1
            if child + 1 < heap_size:
                a.touch(arr + child)
                a.touch(arr + child + 1)
                a.write(arr + child)
            a.touch(arr + j)
            a.touch(arr + child)
            a.write(arr + j)
            a.write(arr + child)
            j = child

    for i in range(N // 2 - 1, -1, -1):
        sift_down(i, N)
    for k in range(N - 1, 0, -1):
        a.touch(arr + k)
        a.touch(arr + 0)
        a.write(arr + k)
        a.write(arr + 0)
        sift_down(0, k)
    a.read_output()
    return a.cost


def manual_mergesort(N: int) -> int:
    """Perfect in-place oblivious mergesort with L1 scratchpad and
    register hoisting (gemini/optimize-mergesort.md).

    By strictly tracking the lifetimes of read variables, the merge
    operates entirely in place — the c_A / c_B scalar registers hold
    left[half-1] and right[0] (the two boundary values that the
    oblivious merge repeatedly re-reads) before the k-sweep, so the
    in-place write of left[k]→arr[k] never conflicts with the still-
    needed boundary reads.

    Scratchpad layout:
      c_A  (addr 1)       — left[half-1] boundary cache
      c_B  (addr 2)       — right[0]     boundary cache
      S    (addr 3..10)   — L1 scratchpad for subtrees of size ≤ 8
      arr  (addr 11..N+10) — main output array

    Subtrees up to S_size use the L1 scratchpad; at the first level
    where a half equals S_size, we compute the left half in S, copy
    it to arr, then compute the right half in S and merge directly
    into arr. Above that level the merge runs fully in-place on arr.
    """
    a = _alloc()
    arr_in = a.alloc_arg(N)
    c_A = a.alloc(1)
    c_B = a.alloc(1)
    S_size = 8
    S = a.alloc(S_size)
    arr = a.alloc(N)
    a.set_output_range(arr, arr + N)

    def rec(base: int, sz: int, dest: str) -> None:
        if sz == 1:
            a.touch_arg(arr_in + base)
            if dest == 'S':
                a.write(S + (base % S_size))
            else:
                a.write(arr + base)
            return

        half = sz // 2

        if sz <= S_size and dest == 'S':
            rec(base, half, 'S')
            rec(base + half, sz - half, 'S')
            a.touch(S + ((base + half - 1) % S_size)); a.write(c_A)
            a.touch(S + ((base + half) % S_size));     a.write(c_B)
            for k in range(sz):
                li = k if k < half else half - 1
                ri = k - half if k >= half else 0
                if li == half - 1:
                    a.touch(c_A)
                else:
                    a.touch(S + ((base + li) % S_size))
                if ri == 0:
                    a.touch(c_B)
                else:
                    a.touch(S + ((base + half + ri) % S_size))
                a.write(S + ((base + k) % S_size))
        elif half == S_size:
            rec(base, half, 'S')
            for i in range(half):
                a.touch(S + ((base + i) % S_size))
                a.write(arr + base + i)
            rec(base + half, half, 'S')
            a.touch(arr + base + half - 1); a.write(c_A)
            a.touch(S + ((base + half) % S_size)); a.write(c_B)
            for k in range(sz):
                li = k if k < half else half - 1
                ri = k - half if k >= half else 0
                if li == half - 1:
                    a.touch(c_A)
                else:
                    a.touch(arr + base + li)
                if ri == 0:
                    a.touch(c_B)
                else:
                    a.touch(S + ((base + half + ri) % S_size))
                a.write(arr + base + k)
        else:
            rec(base, half, 'arr')
            rec(base + half, sz - half, 'arr')
            a.touch(arr + base + half - 1); a.write(c_A)
            a.touch(arr + base + half);     a.write(c_B)
            for k in range(sz):
                li = k if k < half else half - 1
                ri = k - half if k >= half else 0
                if li == half - 1:
                    a.touch(c_A)
                else:
                    a.touch(arr + base + li)
                if ri == 0:
                    a.touch(c_B)
                else:
                    a.touch(arr + base + half + ri)
                a.write(arr + base + k)

    rec(0, N, 'arr')
    a.read_output()
    return a.cost


# ============================================================================
# Longest Common Subsequence DP
# ============================================================================

# ============================================================================
# LU / Gaussian elimination
# ============================================================================

def manual_lu_no_pivot(n: int) -> int:
    """In-place no-pivot LU with hoisted scratchpads and lazy loading.
    Two tight scratchpads occupy addresses 1..n+1:
      c_A  (addr 1)         — hot scalar for pivot / A[i][k]
      c_C  (addr 2..n+1)    — row buffer caching A[k][k+1..n-1]
    The n² upfront preload is replaced with lazy reads from the arg
    stack on the first outer-k pass; every subsequent access comes
    from scratch A just above c_C."""
    a = _alloc()
    A_in = a.alloc_arg(n * n)
    c_A = a.alloc(1)
    c_C = a.alloc(n)
    A = a.alloc(n * n)
    a.set_output_range(A, A + n * n)

    def _read_A(i, j, k):
        if k == 0:
            a.touch_arg(A_in + i * n + j)
        else:
            a.touch(A + i * n + j)

    for k in range(n):
        # Cache pivot A[k][k] into c_A.
        _read_A(k, k, k); a.write(c_A)
        # Divide column k by pivot.
        for i in range(k + 1, n):
            _read_A(i, k, k); a.touch(c_A)
            a.write(A + i * n + k)
        # Cache row k's trailing tail A[k][k+1..n-1] into c_C.
        for j in range(k + 1, n):
            _read_A(k, j, k); a.write(c_C + (j - k - 1))
        # Schur update — for each i, cache A[i][k] into c_A then sweep j.
        for i in range(k + 1, n):
            a.touch(A + i * n + k); a.write(c_A)
            for j in range(k + 1, n):
                _read_A(i, j, k)
                a.touch(c_A)
                a.touch(c_C + (j - k - 1))
                a.write(A + i * n + j)
    a.read_output()
    return a.cost


def manual_blocked_lu(n: int, NB: int = 8) -> int:
    """One-level blocked LU with optimal caching and lazy loading.
    Actively hoists blocks, rows, and scalars into extremely low-address
    stack variables to mimic L1/register reuse, and lazily evaluates the
    argument array to bypass the sequential copy overhead.

    Three unified scratchpads occupy the bottom 73 scratch slots:
      c_A  (addr 1)        — scalar hoist for hot single values
      c_C  (addr 2..NB+1)  — 1D row buffer
      c_B  (addr NB+2..)   — 2D NB×NB block buffer
    c_B is multiplexed across the four stages (diagonal factor, panel
    update, row-strip update, trailing GEMM); c_C caches an A-row
    during the panel + GEMM updates so all (i, j, k)-body reads draw
    from addresses 1..NB²+NB+1. The n² preload is skipped — each A
    cell is touched from the arg stack on its first visit (kb == 0)
    and from scratch A thereafter.
    """
    a = _alloc()
    A_in = a.alloc_arg(n * n)

    # 1. Tight scratchpads at the very bottom of the scratch stack.
    c_A = a.alloc(1)
    c_C = a.alloc(NB)
    c_B = a.alloc(NB * NB)

    # 2. Main target array — right above the scratchpads.
    A = a.alloc(n * n)
    a.set_output_range(A, A + n * n)

    for kb in range(0, n, NB):
        ke = min(kb + NB, n)
        sz = ke - kb

        # (a) Factor the diagonal block locally in c_B.
        for i in range(kb, ke):
            for j in range(kb, ke):
                if kb == 0:
                    a.touch_arg(A_in + i * n + j)
                else:
                    a.touch(A + i * n + j)
                a.write(c_B + (i - kb) * NB + (j - kb))

        for k in range(sz):
            pivot_addr = c_B + k * NB + k
            a.touch(pivot_addr); a.write(c_A)
            for i in range(k + 1, sz):
                a.touch(c_B + i * NB + k); a.touch(c_A)
                a.write(c_B + i * NB + k)
            for i in range(k + 1, sz):
                a.touch(c_B + i * NB + k); a.write(c_A)
                for j in range(k + 1, sz):
                    a.touch(c_B + i * NB + j)
                    a.touch(c_A)
                    a.touch(c_B + k * NB + j)
                    a.write(c_B + i * NB + j)

        for i in range(kb, ke):
            for j in range(kb, ke):
                a.touch(c_B + (i - kb) * NB + (j - kb))
                a.write(A + i * n + j)

        # (b) Panel update A[ke:n, kb:ke] — cache each row into c_C.
        for ib in range(ke, n, NB):
            ie = min(ib + NB, n)
            for i in range(ib, ie):
                for j in range(kb, ke):
                    if kb == 0:
                        a.touch_arg(A_in + i * n + j)
                    else:
                        a.touch(A + i * n + j)
                    a.write(c_C + (j - kb))
                for k in range(sz):
                    a.touch(c_C + k)
                    a.touch(c_B + k * NB + k)
                    a.write(c_C + k)

                    a.touch(c_C + k); a.write(c_A)
                    for j in range(k + 1, sz):
                        a.touch(c_C + j)
                        a.touch(c_A)
                        a.touch(c_B + k * NB + j)
                        a.write(c_C + j)
                for j in range(kb, ke):
                    a.touch(c_C + (j - kb))
                    a.write(A + i * n + j)

        # (c) Row-strip update A[kb:ke, ke:n] — buffer block into c_B.
        for jb in range(ke, n, NB):
            je = min(jb + NB, n)
            sz_j = je - jb
            for k in range(kb, ke):
                for j in range(jb, je):
                    if kb == 0:
                        a.touch_arg(A_in + k * n + j)
                    else:
                        a.touch(A + k * n + j)
                    a.write(c_B + (k - kb) * NB + (j - jb))
            for k in range(sz):
                for i in range(k + 1, sz):
                    a.touch(A + (kb + i) * n + (kb + k))
                    a.write(c_A)
                    for j in range(sz_j):
                        a.touch(c_B + i * NB + j)
                        a.touch(c_A)
                        a.touch(c_B + k * NB + j)
                        a.write(c_B + i * NB + j)
            for k in range(kb, ke):
                for j in range(jb, je):
                    a.touch(c_B + (k - kb) * NB + (j - jb))
                    a.write(A + k * n + j)

        # (d) Trailing GEMM update — block-row register loading.
        for jb in range(ke, n, NB):
            je = min(jb + NB, n)
            sz_j = je - jb

            for k in range(kb, ke):
                for j in range(jb, je):
                    a.touch(A + k * n + j)
                    a.write(c_B + (k - kb) * NB + (j - jb))

            for ib in range(ke, n, NB):
                ie = min(ib + NB, n)
                for i in range(ib, ie):
                    for j in range(jb, je):
                        if kb == 0:
                            a.touch_arg(A_in + i * n + j)
                        else:
                            a.touch(A + i * n + j)
                        a.write(c_C + (j - jb))

                    for k in range(sz):
                        a.touch(A + i * n + (kb + k))
                        a.write(c_A)
                        for j in range(sz_j):
                            a.touch(c_C + j)
                            a.touch(c_A)
                            a.touch(c_B + k * NB + j)
                            a.write(c_C + j)

                    for j in range(jb, je):
                        a.touch(c_C + (j - jb))
                        a.write(A + i * n + j)

    a.read_output()
    return a.cost


def manual_recursive_lu(n: int) -> int:
    """Recursive LU with hoisted scratchpads. Each of the three Schur-
    style inner loops caches one operand of its `A[i][j] -= A[i][k] *
    A[k][j]` body into a low-address scratchpad, cutting the inner-loop
    read traffic from 3 bulk A reads to 1 bulk + 2 hot.

      c_A   (addr 1)      — pivot / row-k scalar
      c_B   (addr 2)      — column-k scalar (A[i][k] after divide)
      c_C   (addr 3..n+2) — row-k trailing buffer"""
    a = _alloc()
    A_in = a.alloc_arg(n * n)
    c_A = a.alloc(1)
    c_B = a.alloc(1)
    c_C = a.alloc(n)
    A = a.alloc(n * n)
    a.set_output_range(A, A + n * n)
    for i in range(n * n):
        a.touch_arg(A_in + i); a.write(A + i)

    def rec(r0: int, c0: int, sz: int) -> None:
        if sz == 1:
            return
        h = sz // 2
        rec(r0, c0, h)

        # --- Column panel update A[r0+h..r0+sz, c0..c0+h] ---
        for k in range(c0, c0 + h):
            a.touch(A + k * n + k); a.write(c_A)                 # pivot
            for j in range(k + 1, c0 + h):
                a.touch(A + k * n + j); a.write(c_C + (j - k - 1))
            for i in range(r0 + h, r0 + sz):
                a.touch(A + i * n + k); a.touch(c_A)
                a.write(A + i * n + k)                            # divide
                a.touch(A + i * n + k); a.write(c_B)              # hot row-k
                for j in range(k + 1, c0 + h):
                    a.touch(A + i * n + j)
                    a.touch(c_B)
                    a.touch(c_C + (j - k - 1))
                    a.write(A + i * n + j)

        # --- Row-strip update A[r0..r0+h, c0+h..c0+sz] ---
        for k in range(r0, r0 + h):
            # cache A[k][c0+h..c0+sz-1] into c_C
            for j in range(c0 + h, c0 + sz):
                a.touch(A + k * n + j); a.write(c_C + (j - (c0 + h)))
            for i in range(k + 1, r0 + h):
                a.touch(A + i * n + k); a.write(c_B)              # hot col-k
                for j in range(c0 + h, c0 + sz):
                    a.touch(A + i * n + j)
                    a.touch(c_B)
                    a.touch(c_C + (j - (c0 + h)))
                    a.write(A + i * n + j)

        # --- Trailing submatrix update A[r0+h.., c0+h..] ---
        for k in range(c0, c0 + h):
            # cache A[k][c0+h..c0+sz-1] into c_C
            for j in range(c0 + h, c0 + sz):
                a.touch(A + k * n + j); a.write(c_C + (j - (c0 + h)))
            for i in range(r0 + h, r0 + sz):
                a.touch(A + i * n + k); a.write(c_B)              # hot col-k
                for j in range(c0 + h, c0 + sz):
                    a.touch(A + i * n + j)
                    a.touch(c_B)
                    a.touch(c_C + (j - (c0 + h)))
                    a.write(A + i * n + j)

        rec(r0 + h, c0 + h, sz - h)

    rec(0, 0, n)
    a.read_output()
    return a.cost


def manual_lu_partial_pivot(n: int) -> int:
    """LU with partial pivoting. Hoisted c_A + c_C scratchpads and lazy
    loading, same pattern as manual_lu_no_pivot, plus a column-scan
    pivot-selection phase and a row-swap pass each outer step."""
    a = _alloc()
    A_in = a.alloc_arg(n * n)
    c_A = a.alloc(1)
    c_C = a.alloc(n)
    A = a.alloc(n * n)
    a.set_output_range(A, A + n * n)

    def _read(i, j, k):
        if k == 0:
            a.touch_arg(A_in + i * n + j)
        else:
            a.touch(A + i * n + j)

    for k in range(n):
        # Pivot selection — scan column k below & including diagonal.
        for i in range(k, n):
            _read(i, k, k); a.touch(A + k * n + k) if k > 0 else a.touch_arg(
                A_in + k * n + k)
        # Row swap: rows k and p across columns [k, n).
        p = k + 1 if k + 1 < n else k
        for j in range(k, n):
            _read(k, j, k); _read(p, j, k)
            a.write(A + k * n + j); a.write(A + p * n + j)

        # Cache pivot into c_A (now in scratch post-swap).
        a.touch(A + k * n + k); a.write(c_A)
        # Divide column k.
        for i in range(k + 1, n):
            a.touch(A + i * n + k); a.touch(c_A)
            a.write(A + i * n + k)
        # Cache row k's trailing tail into c_C.
        for j in range(k + 1, n):
            a.touch(A + k * n + j)
            a.write(c_C + (j - k - 1))
        # Schur update — hot A[i][k] in c_A, row in c_C.
        for i in range(k + 1, n):
            a.touch(A + i * n + k); a.write(c_A)
            for j in range(k + 1, n):
                a.touch(A + i * n + j)
                a.touch(c_A)
                a.touch(c_C + (j - k - 1))
                a.write(A + i * n + j)
    a.read_output()
    return a.cost


# ============================================================================
# Cholesky
# ============================================================================

def manual_cholesky(n: int) -> int:
    """Right-looking Cholesky with hoisted scratchpads and lazy loading.
    Two tight scratchpads at the bottom of the stack:
      c_A  (addr 1)         — hot scalar for pivot / A[j][k] reuse
      c_C  (addr 2..n+1)    — column buffer caching A[k+1..n-1][k]
    The Schur update reuses A[j][k] as a constant across its inner
    i-loop (held in c_A) and the column-k values A[i][k] across both
    indices (held in c_C). Lower triangle only, so ~half the Schur
    traffic of a full LU."""
    a = _alloc()
    A_in = a.alloc_arg(n * n)
    c_A = a.alloc(1)
    c_C = a.alloc(n)
    A = a.alloc(n * n)
    a.set_output_range(A, A + n * n)

    def _read(i, j, k):
        if k == 0:
            a.touch_arg(A_in + i * n + j)
        else:
            a.touch(A + i * n + j)

    for k in range(n):
        # Pivot: A[k][k] = sqrt(A[k][k]). Hoist pivot into c_A.
        _read(k, k, k); a.write(A + k * n + k)
        a.touch(A + k * n + k); a.write(c_A)
        # Divide column k: A[i][k] /= pivot.
        for i in range(k + 1, n):
            _read(i, k, k); a.touch(c_A)
            a.write(A + i * n + k)
        # Cache column k (below diagonal) into c_C.
        for i in range(k + 1, n):
            a.touch(A + i * n + k)
            a.write(c_C + (i - k - 1))
        # Schur update — for each j, pin A[j][k] into c_A and sweep i.
        for j in range(k + 1, n):
            a.touch(c_C + (j - k - 1)); a.write(c_A)
            for i in range(j, n):
                _read(i, j, k)
                a.touch(c_C + (i - k - 1))
                a.touch(c_A)
                a.write(A + i * n + j)
    a.read_output()
    return a.cost


# ============================================================================
# QR — Householder family
# ============================================================================

def manual_householder_qr(m: int, n: int) -> int:
    """Classical Householder QR with hoisted scratchpads. Same pattern
    as manual_blocked_qr but without the block-column structure:
      c_A  (addr 1)      — dot-product accumulator
      c_V  (addr 2..m+1) — reflector-column buffer, loaded once per
                           reflector k and reused across n trailing cols"""
    a = _alloc()
    A_in = a.alloc_arg(m * n)
    c_A = a.alloc(1)
    c_V = a.alloc(m)
    A = a.alloc(m * n)
    a.set_output_range(A, A + m * n)
    for i in range(m * n):
        a.touch_arg(A_in + i); a.write(A + i)
    for k in range(min(m, n)):
        # Reflector norm stage.
        a.touch(A + k * n + k)
        for i in range(k + 1, m):
            a.touch(A + i * n + k)
        a.write(A + k * n + k)
        for i in range(k + 1, m):
            a.write(A + i * n + k)
        # Cache reflector column into c_V.
        a.touch(A + k * n + k); a.write(c_V + 0)
        for i in range(k + 1, m):
            a.touch(A + i * n + k); a.write(c_V + (i - k))
        # Apply to every trailing column j.
        for j in range(k + 1, n):
            a.touch(c_V + 0); a.touch(A + k * n + j); a.write(c_A)
            for i in range(k + 1, m):
                a.touch(c_V + (i - k)); a.touch(A + i * n + j)
                a.touch(c_A); a.write(c_A)
            a.touch(c_A); a.touch(c_V + 0); a.touch(A + k * n + j)
            a.write(A + k * n + j)
            for i in range(k + 1, m):
                a.touch(c_A); a.touch(c_V + (i - k))
                a.touch(A + i * n + j); a.write(A + i * n + j)
    a.read_output()
    return a.cost


def manual_blocked_qr(m: int, n: int, NB: int = 8) -> int:
    """Blocked QR (WY form, simplified) with hoisted scratchpads.

    Three tight scratchpads at the bottom of the stack:
      c_A   (addr 1)      — dot-product accumulator (hot scalar)
      c_V   (addr 2..m+1) — reflector-column buffer, loaded once per
                            reflector k and reused across all trailing
                            columns j
      c_W   (addr m+2..m+NB+1) — per-reflector dot-product cache in the
                                 intra-panel update step (was `w`)

    Observation: the inner A[i][k] reads of the trailing-panel update
    can be pulled into c_V because each reflector is independent across
    trailing columns (their updates write disjoint columns). So
    restructuring the loop as `for k outer, for j inner` lets us load
    each reflector into c_V exactly once and hit it 2·(m-k)·(n-ke)
    times at near-top-of-stack depth."""
    a = _alloc()
    A_in = a.alloc_arg(m * n)
    c_A = a.alloc(1)
    c_V = a.alloc(m)
    c_W = a.alloc(NB)
    A = a.alloc(m * n)
    a.set_output_range(A, A + m * n)
    for i in range(m * n):
        a.touch_arg(A_in + i); a.write(A + i)

    for kb in range(0, min(m, n), NB):
        ke = min(kb + NB, min(m, n))

        # --- Panel reduction: sequential reflectors within the panel. ---
        for k in range(kb, ke):
            # Build reflector: read column k below+including diagonal.
            a.touch(A + k * n + k)
            for i in range(k + 1, m):
                a.touch(A + i * n + k)
            a.write(A + k * n + k)
            for i in range(k + 1, m):
                a.write(A + i * n + k)
            # Cache the reflector column into c_V.
            a.touch(A + k * n + k); a.write(c_V + 0)
            for i in range(k + 1, m):
                a.touch(A + i * n + k); a.write(c_V + (i - k))
            # Apply the reflector to the remaining panel columns j.
            for j in range(k + 1, ke):
                # Dot product v · A[:, j] accumulated in c_A.
                a.touch(c_V + 0); a.touch(A + k * n + j); a.write(c_A)
                for i in range(k + 1, m):
                    a.touch(c_V + (i - k)); a.touch(A + i * n + j)
                    a.touch(c_A); a.write(c_A)
                # Rank-1 update using c_A and c_V.
                a.touch(c_A); a.touch(c_V + 0); a.touch(A + k * n + j)
                a.write(A + k * n + j)
                for i in range(k + 1, m):
                    a.touch(c_A); a.touch(c_V + (i - k))
                    a.touch(A + i * n + j); a.write(A + i * n + j)

        # --- Trailing-panel update: reflector-outer, column-inner. ---
        for k in range(kb, ke):
            # Cache reflector k into c_V.
            a.touch(A + k * n + k); a.write(c_V + 0)
            for i in range(k + 1, m):
                a.touch(A + i * n + k); a.write(c_V + (i - k))
            for j in range(ke, n):
                # Dot v · A[:, j] → c_A.
                a.touch(c_V + 0); a.touch(A + k * n + j); a.write(c_A)
                for i in range(k + 1, m):
                    a.touch(c_V + (i - k)); a.touch(A + i * n + j)
                    a.touch(c_A); a.write(c_A)
                # Update.
                a.touch(c_A); a.touch(c_V + 0); a.touch(A + k * n + j)
                a.write(A + k * n + j)
                for i in range(k + 1, m):
                    a.touch(c_A); a.touch(c_V + (i - k))
                    a.touch(A + i * n + j); a.write(A + i * n + j)
    a.read_output()
    return a.cost


def manual_tsqr(m: int, n: int, block_rows: int = 8) -> int:
    """Tall-skinny QR with hoisted scratchpads.
    Phase 1 does local Householder QR per row-tile. Phase 2 tree-reduces
    pairs of R factors. Both phases cache the current reflector column
    into c_V and reuse it across the n trailing columns — exactly the
    same trick as manual_blocked_qr.

      c_A  (addr 1)                    — dot-product accumulator
      c_V  (addr 2..block_rows+2)      — reflector column buffer
                                         (Phase 2 needs one extra slot
                                         for the left-block pivot element)"""
    a = _alloc()
    A_in = a.alloc_arg(m * n)
    c_A = a.alloc(1)
    c_V = a.alloc(block_rows + 1)
    A = a.alloc(m * n)
    a.set_output_range(A, A + m * n)
    for i in range(m * n):
        a.touch_arg(A_in + i); a.write(A + i)

    # --- Phase 1: local QR per row-tile. Cache the reflector column
    # into c_V and reuse across all n trailing columns. -----------------
    for row0 in range(0, m, block_rows):
        row1 = min(row0 + block_rows, m)
        for k in range(min(row1 - row0, n)):
            kk = row0 + k
            a.touch(A + kk * n + k)
            for i in range(kk + 1, row1):
                a.touch(A + i * n + k)
            a.write(A + kk * n + k)
            for i in range(kk + 1, row1):
                a.write(A + i * n + k)
            a.touch(A + kk * n + k); a.write(c_V + 0)
            for i in range(kk + 1, row1):
                a.touch(A + i * n + k); a.write(c_V + (i - kk))
            for j in range(k + 1, n):
                a.touch(c_V + 0); a.touch(A + kk * n + j); a.write(c_A)
                for i in range(kk + 1, row1):
                    a.touch(c_V + (i - kk)); a.touch(A + i * n + j)
                    a.touch(c_A); a.write(c_A)
                a.touch(c_A); a.touch(c_V + 0); a.touch(A + kk * n + j)
                a.write(A + kk * n + j)
                for i in range(kk + 1, row1):
                    a.touch(c_A); a.touch(c_V + (i - kk))
                    a.touch(A + i * n + j); a.write(A + i * n + j)

    # --- Phase 2: pairwise tree-reduction over R factors. ---------------
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
                # Cache reflector: one pivot element from left + right block.
                a.touch(A + (left_row + k) * n + k); a.write(c_V + 0)
                for i in range(right_row + k, right_end):
                    a.touch(A + i * n + k); a.write(c_V + 1 + (i - right_row - k))
                for j in range(k + 1, n):
                    a.touch(c_V + 0); a.touch(A + (left_row + k) * n + j)
                    a.write(c_A)
                    for i in range(right_row + k, right_end):
                        a.touch(c_V + 1 + (i - right_row - k))
                        a.touch(A + i * n + j)
                        a.touch(c_A); a.write(c_A)
                    a.touch(c_A); a.touch(c_V + 0)
                    a.touch(A + (left_row + k) * n + j)
                    a.write(A + (left_row + k) * n + j)
                    for i in range(right_row + k, right_end):
                        a.touch(c_A); a.touch(c_V + 1 + (i - right_row - k))
                        a.touch(A + i * n + j); a.write(A + i * n + j)
        stride *= 2
    a.read_output()
    return a.cost


# ============================================================================
# Longest Common Subsequence DP
# ============================================================================

def manual_lcs_dp(m: int, n: int) -> int:
    """Row-major LCS DP with a rolling 2-row buffer (only D[m][n] is
    returned, so the full (m+1)(n+1) table is unnecessary).

    Hoisted scratchpads:
      c_A    (addr 1)          — hot scalar x[i-1], constant in j-sweep
      row_a  (addr 2..n+2)     — one DP row
      row_b  (addr n+3..2n+3)  — the other DP row
    row_a / row_b ping-pong each outer i; the inner-j reads (D[i-1][j-1],
    D[i-1][j], D[i][j-1]) all hit these low-address buffers. A single
    `answer` cell just above holds the final D[m][n] for the output
    epilogue."""
    a = _alloc()
    x = a.alloc_arg(m); y = a.alloc_arg(n)
    c_A = a.alloc(1)
    row_a = a.alloc(n + 1)
    row_b = a.alloc(n + 1)
    answer = a.alloc(1)
    a.set_output_range(answer, answer + 1)

    row_prev, row_cur = row_a, row_b
    for i in range(1, m + 1):
        a.touch_arg(x + i - 1); a.write(c_A)
        for j in range(1, n + 1):
            a.touch(row_prev + (j - 1))   # D[i-1][j-1]
            a.touch(row_prev + j)         # D[i-1][j]
            a.touch(row_cur + (j - 1))    # D[i][j-1]
            a.touch(c_A)                  # x[i-1]
            a.touch_arg(y + j - 1)        # y[j-1]
            a.write(row_cur + j)
        row_prev, row_cur = row_cur, row_prev
    a.touch(row_prev + n); a.write(answer)
    a.read_output()
    return a.cost


# ============================================================================
# Time-skewed stencils
# ============================================================================

def manual_stencil_time_naive(n: int, T: int = 4) -> int:
    """T sweeps of 5-point Jacobi. Input A on arg stack; cur/nxt ping-
    pong on scratch. Each sweep reads cur (bulk) and writes nxt (bulk)."""
    a = _alloc()
    A = a.alloc_arg(n * n)
    cur = a.alloc(n * n); nxt = a.alloc(n * n)
    a.set_output_range(cur, cur + n * n)
    for i in range(n * n):
        a.touch_arg(A + i); a.write(cur + i)
    for t in range(T):
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                a.touch(cur + i * n + j)
                a.touch(cur + (i - 1) * n + j)
                a.touch(cur + (i + 1) * n + j)
                a.touch(cur + i * n + j - 1)
                a.touch(cur + i * n + j + 1)
                a.write(nxt + i * n + j)
        cur, nxt = nxt, cur
        a.set_output_range(cur, cur + n * n)
    a.read_output()
    return a.cost


def manual_stencil_time_diamond(n: int, T: int = 4, block: int = 4) -> int:
    """Perfectly in-place diamond time-tiling with L1 row caching and
    sliding scalar registers (gemini/optimize-stencil-time-diamond.md).

    Three optimizations layered onto the naive halo-buffered schedule:
      (1) Lazy arg loading: only arg cells that are actually inside
          the Manhattan-distance diamond at the current block get
          touched, and only on their very first visit.
      (2) In-place time-stepping: the second "buf_nxt" array is
          dropped entirely. The sliding horizontal window (c_left,
          c_center, c_right) plus a top-row buffer prev_row lets every
          update read the stale neighbor values before the in-place
          write touches that cell.
      (3) Diamond pruning: each time step t clips to the shrinking
          dependence cone (dist_i + dist_j <= T - 1 - t), skipping
          cells whose values would be overwritten by halo before they
          become needed.

    Layout:
      c_left, c_center, c_right  (addrs 1..3)
      prev_row                   (addrs 4..stride+3)
      buf_cur                    (addrs stride+4..stride²+stride+3)
      cur                        (addrs stride²+stride+4..)
    """
    a = _alloc()
    A = a.alloc_arg(n * n)
    c_left = a.alloc(1); c_center = a.alloc(1); c_right = a.alloc(1)
    stride = block + 2 * T
    prev_row = a.alloc(stride)
    buf_cur = a.alloc(stride * stride)
    cur = a.alloc(n * n)
    a.set_output_range(cur, cur + n * n)

    loaded = [[False] * n for _ in range(n)]

    for bi in range(0, n, block):
        for bj in range(0, n, block):
            rr = max(0, bi - T); cc = max(0, bj - T)
            rows = min(n, bi + block + T) - rr
            cols = min(n, bj + block + T) - cc

            # (a) Lazy-load just the diamond subset into buf_cur.
            for ii in range(rows):
                for jj in range(cols):
                    r_glob = rr + ii
                    c_glob = cc + jj
                    dist_i = 0
                    if r_glob < bi: dist_i = bi - r_glob
                    elif r_glob >= bi + block: dist_i = r_glob - (bi + block - 1)
                    dist_j = 0
                    if c_glob < bj: dist_j = bj - c_glob
                    elif c_glob >= bj + block: dist_j = c_glob - (bj + block - 1)
                    if dist_i + dist_j <= T:
                        if not loaded[r_glob][c_glob]:
                            a.touch_arg(A + r_glob * n + c_glob)
                            loaded[r_glob][c_glob] = True
                        else:
                            a.touch(cur + r_glob * n + c_glob)
                        a.write(buf_cur + ii * stride + jj)

            # (b) In-place time-stepping with sliding register window.
            for t in range(T):
                for jj in range(cols):
                    a.touch(buf_cur + 0 * stride + jj)
                    a.write(prev_row + jj)
                for ii in range(1, rows - 1):
                    a.touch(buf_cur + ii * stride + 0); a.write(c_left)
                    a.touch(buf_cur + ii * stride + 1); a.write(c_center)
                    for jj in range(1, cols - 1):
                        a.touch(buf_cur + ii * stride + jj + 1)
                        a.write(c_right)

                        r_glob = rr + ii
                        c_glob = cc + jj
                        dist_i = 0
                        if r_glob < bi: dist_i = bi - r_glob
                        elif r_glob >= bi + block: dist_i = r_glob - (bi + block - 1)
                        dist_j = 0
                        if c_glob < bj: dist_j = bj - c_glob
                        elif c_glob >= bj + block: dist_j = c_glob - (bj + block - 1)

                        if dist_i + dist_j <= T - 1 - t:
                            if (0 < r_glob < n - 1) and (0 < c_glob < n - 1):
                                a.touch(c_center)
                                a.touch(prev_row + jj)
                                a.touch(buf_cur + (ii + 1) * stride + jj)
                                a.touch(c_left)
                                a.touch(c_right)
                                a.write(buf_cur + ii * stride + jj)

                        # Slide registers forward every column.
                        a.touch(c_center); a.write(prev_row + jj)
                        a.touch(c_center); a.write(c_left)
                        a.touch(c_right);  a.write(c_center)

            # (c) Flush the block interior to cur.
            for i in range(bi, min(bi + block, n)):
                for j in range(bj, min(bj + block, n)):
                    li = i - rr; lj = j - cc
                    a.touch(buf_cur + li * stride + lj)
                    a.write(cur + i * n + j)
    a.read_output()
    return a.cost


# ============================================================================
# Floyd-Warshall
# ============================================================================

def manual_floyd_warshall_naive(V: int) -> int:
    """Standard 3-loop APSP with hoisted scratchpads and lazy loading.
    Same inner body as lu_no_pivot's Schur update — apply the same
    pattern:
      c_A  (addr 1)         — hot scalar for D[i][k]
      c_C  (addrs 2..V+1)   — row buffer caching D[k][0..V-1]
    Lazy arg reads at k=0 replace the V² preload: every D cell is
    first-touched at k=0 (either in the row-cache step, the c_A
    cache step, or the Schur inner body) so it's safe to route
    arg→scratch on the first visit."""
    a = _alloc()
    M = a.alloc_arg(V * V)
    c_A = a.alloc(1)
    c_C = a.alloc(V)
    D = a.alloc(V * V)
    a.set_output_range(D, D + V * V)

    def _read(i, j, k):
        if k == 0:
            a.touch_arg(M + i * V + j)
        else:
            a.touch(D + i * V + j)

    for k in range(V):
        # Cache row k into c_C.
        for j in range(V):
            _read(k, j, k); a.write(c_C + j)
        for i in range(V):
            _read(i, k, k); a.write(c_A)
            for j in range(V):
                _read(i, j, k)
                a.touch(c_A)
                a.touch(c_C + j)
                a.write(D + i * V + j)
    a.read_output()
    return a.cost


def manual_floyd_warshall_recursive(V: int) -> int:
    """Kleene's cache-oblivious APSP: 8 recursive quadrant calls. Reads
    stay inside the current submatrix when sz is small, giving a
    rmm-like cache profile."""
    a = _alloc()
    M = a.alloc_arg(V * V)
    D = a.alloc(V * V)
    a.set_output_range(D, D + V * V)
    for i in range(V * V):
        a.touch_arg(M + i); a.write(D + i)

    def leaf(r0: int, c0: int, sz: int) -> None:
        for k in range(r0, r0 + sz):
            for i in range(r0, r0 + sz):
                for j in range(c0, c0 + sz):
                    a.touch(D + i * V + j)
                    a.touch(D + i * V + k)
                    a.touch(D + k * V + j)
                    a.write(D + i * V + j)

    def rec(r0: int, c0: int, sz: int) -> None:
        if sz <= 2:
            leaf(r0, c0, sz); return
        h = sz // 2
        for dr, dc in [(0, 0), (0, h), (h, 0), (h, h),
                       (h, h), (h, 0), (0, h), (0, 0)]:
            rec(r0 + dr, c0 + dc, h)

    rec(0, 0, V)
    a.read_output()
    return a.cost


# ============================================================================
# LayerNorm
# ============================================================================

def manual_layernorm_unfused(N: int) -> int:
    """Three-pass LayerNorm. x on arg; s/v/mean/inv_std scalars and y
    output on scratch. Each of the 3 passes reads the full x vector."""
    a = _alloc()
    x = a.alloc_arg(N)
    s = a.alloc(1); v = a.alloc(1); mean = a.alloc(1); inv_std = a.alloc(1)
    tmp = a.alloc(1)
    y = a.alloc(N)
    a.set_output_range(y, y + N)
    # Pass 1: mean
    a.touch_arg(x + 0); a.write(s)
    for i in range(1, N):
        a.touch_arg(x + i); a.touch(s); a.write(s)
    a.touch(s); a.write(mean)
    # Pass 2: variance
    a.touch_arg(x + 0); a.touch(mean); a.touch(tmp); a.write(v)
    for i in range(1, N):
        a.touch_arg(x + i); a.touch(mean)
        a.touch(tmp); a.touch(v); a.write(v)
    a.touch(v); a.write(inv_std)
    # Pass 3: y[i] = (x[i] - mean) * inv_std
    for i in range(N):
        a.touch_arg(x + i); a.touch(mean); a.touch(inv_std)
        a.write(y + i)
    a.read_output()
    return a.cost


def manual_layernorm_fused(N: int) -> int:
    """Welford's online mean+variance in a single pass, then a second
    pass to normalize. x on arg; running scalars mu, m2 at addr 1-2 so
    they're cache-top across all N updates."""
    a = _alloc()
    x = a.alloc_arg(N)
    mu = a.alloc(1); m2 = a.alloc(1); inv_std = a.alloc(1)
    delta = a.alloc(1); delta2 = a.alloc(1)
    y = a.alloc(N)
    a.set_output_range(y, y + N)
    # Init from x[0]
    a.touch_arg(x + 0); a.write(mu)
    a.touch_arg(x + 0); a.write(m2)
    # Welford sweep
    for i in range(1, N):
        a.touch_arg(x + i); a.touch(mu); a.write(delta)
        a.touch(mu); a.touch(delta); a.write(mu)
        a.touch_arg(x + i); a.touch(mu); a.write(delta2)
        a.touch(m2); a.touch(delta); a.touch(delta2); a.write(m2)
    a.touch(m2); a.write(inv_std)
    # Normalize sweep
    for i in range(N):
        a.touch_arg(x + i); a.touch(mu); a.touch(inv_std)
        a.write(y + i)
    a.read_output()
    return a.cost


# ============================================================================
# Matrix Powers Kernel
# ============================================================================

def manual_matrix_powers_naive(n: int, s: int = 4) -> int:
    """Run matvec s times naively. A, x on arg; cur/nxt vectors ping-pong
    on scratch. A is re-read in full for each of the s steps."""
    a = _alloc()
    A = a.alloc_arg(n * n); x0 = a.alloc_arg(n)
    cur = a.alloc(n); nxt = a.alloc(n); acc = a.alloc(1); tmp = a.alloc(1)
    a.set_output_range(cur, cur + n)
    for i in range(n):
        a.touch_arg(x0 + i); a.write(cur + i)
    for step in range(s):
        for i in range(n):
            a.touch_arg(A + i * n + 0); a.touch(cur + 0); a.write(acc)
            for j in range(1, n):
                a.touch_arg(A + i * n + j); a.touch(cur + j)
                a.touch(acc); a.touch(tmp); a.write(acc)
            a.touch(acc); a.write(nxt + i)
        # swap cur <- nxt for next step
        for i in range(n):
            a.touch(nxt + i); a.write(cur + i)
        a.set_output_range(cur, cur + n)
    a.read_output()
    return a.cost


def manual_matrix_powers_ca(n: int, s: int = 4, block: int = 4) -> int:
    """Communication-avoiding s-step: for each row-block of A, compute
    its contribution to all output-vector positions across steps locally
    before moving on. A is read once per (block, step) instead of once
    per step. The CA win under two-stack is bounded because A already
    lives on the arg stack — but the heuristics should still pick up
    the reduced re-read count."""
    a = _alloc()
    A = a.alloc_arg(n * n); x0 = a.alloc_arg(n)
    cur = a.alloc(n); nxt = a.alloc(n); acc = a.alloc(1); tmp = a.alloc(1)
    a.set_output_range(cur, cur + n)
    for i in range(n):
        a.touch_arg(x0 + i); a.write(cur + i)
    for step in range(s):
        for bi in range(0, n, block):
            for i in range(bi, min(bi + block, n)):
                a.touch_arg(A + i * n + 0); a.touch(cur + 0); a.write(acc)
                for j in range(1, n):
                    a.touch_arg(A + i * n + j); a.touch(cur + j)
                    a.touch(acc); a.touch(tmp); a.write(acc)
                a.touch(acc); a.write(nxt + i)
        for i in range(n):
            a.touch(nxt + i); a.write(cur + i)
        a.set_output_range(cur, cur + n)
    a.read_output()
    return a.cost


# ============================================================================
# Left-looking Cholesky
# ============================================================================

def manual_cholesky_left_looking(n: int) -> int:
    """Left-looking Cholesky with hoisted scratchpads and lazy loading.
    For column k, pulls from all previously factored columns 0..k-1.
    Two tight scratchpads at the bottom of the stack:
      c_A  (addr 1)       — accumulator pinned to L[i][k] during the
                             inner past-factor sweep
      c_C  (addr 2..n+1)  — row buffer caching L[k][0..k-1]

    Inner loop per (i, j): reads L[i][j] (bulk), c_C[j] (hot), c_A (hot)
    → writes c_A. Only one bulk read per inner op, versus two in the
    naive version. Lazy arg-stack reads replace the n² preload."""
    a = _alloc()
    A_in = a.alloc_arg(n * n)
    c_A = a.alloc(1)
    c_C = a.alloc(n)
    L = a.alloc(n * n)
    a.set_output_range(L, L + n * n)

    for k in range(n):
        # Accumulation (empty at k=0). Cache row k's past values into c_C.
        if k > 0:
            for j in range(k):
                a.touch(L + k * n + j); a.write(c_C + j)
            for i in range(k, n):
                a.touch(L + i * n + k); a.write(c_A)
                for j in range(k):
                    a.touch(L + i * n + j)
                    a.touch(c_C + j)
                    a.touch(c_A)
                    a.write(c_A)
                a.touch(c_A); a.write(L + i * n + k)
        # Diagonal sqrt. Lazy-load A[k][k] at k=0.
        if k == 0:
            a.touch_arg(A_in + k * n + k)
        else:
            a.touch(L + k * n + k)
        a.write(L + k * n + k)
        # Pin pivot into c_A.
        a.touch(L + k * n + k); a.write(c_A)
        # Divide column k — lazy-load column 0 at k=0.
        for i in range(k + 1, n):
            if k == 0:
                a.touch_arg(A_in + i * n + k)
            else:
                a.touch(L + i * n + k)
            a.touch(c_A)
            a.write(L + i * n + k)
    a.read_output()
    return a.cost


# ============================================================================
# Sparse matvec (CSR)
# ============================================================================

def _manual_spmv(n: int, row_ptr, col_ind) -> int:
    """Shared manual driver for CSR spmv: vals and x on arg stack,
    y output and accumulator on scratch. row_ptr/col_ind are compile-
    time integers — they don't incur memory cost (i.e., they're fused
    into the instruction stream)."""
    a = _alloc()
    vals = a.alloc_arg(len(col_ind)); x = a.alloc_arg(n)
    acc = a.alloc(1); tmp = a.alloc(1)
    y = a.alloc(n)
    a.set_output_range(y, y + n)
    for i in range(n):
        start = row_ptr[i]; end = row_ptr[i + 1]
        if start == end:
            a.touch_arg(x + 0); a.write(y + i)
            continue
        a.touch_arg(vals + start); a.touch_arg(x + col_ind[start])
        a.write(acc)
        for k in range(start + 1, end):
            a.touch_arg(vals + k); a.touch_arg(x + col_ind[k])
            a.touch(acc); a.touch(tmp); a.write(acc)
        a.touch(acc); a.write(y + i)
    a.read_output()
    return a.cost


def manual_spmv_csr_banded(n: int, bandwidth: int = 3) -> int:
    row_ptr, col_ind = [0], []
    total = 0
    for i in range(n):
        for j in range(max(0, i - bandwidth), min(n, i + bandwidth + 1)):
            col_ind.append(j); total += 1
        row_ptr.append(total)
    return _manual_spmv(n, row_ptr, col_ind)


def manual_spmv_csr_random(n: int, nnz_per_row: int = 7,
                           seed: int = 0xC0FFEE) -> int:
    import random
    rng = random.Random(seed)
    row_ptr, col_ind = [0], []
    total = 0
    for i in range(n):
        cols = sorted(rng.sample(range(n), min(nnz_per_row, n)))
        for j in cols:
            col_ind.append(j); total += 1
        row_ptr.append(total)
    return _manual_spmv(n, row_ptr, col_ind)


# ============================================================================
# Bitonic sort (data-oblivious network)
# ============================================================================

def manual_bitonic_sort(N: int) -> int:
    """Sorting network: input arr preloaded from arg stack to scratch.
    For each (k, j), every pair (i, i^j) with l=i^j > i is read together
    and written back — same butterfly pattern as iterative FFT."""
    a = _alloc()
    arr_in = a.alloc_arg(N)
    arr = a.alloc(N)
    a.set_output_range(arr, arr + N)
    for i in range(N):
        a.touch_arg(arr_in + i); a.write(arr + i)
    k = 2
    while k <= N:
        j = k // 2
        while j > 0:
            for i in range(N):
                l = i ^ j
                if l > i:
                    a.touch(arr + i); a.touch(arr + l)
                    a.write(arr + i); a.write(arr + l)
            j //= 2
        k *= 2
    a.read_output()
    return a.cost
