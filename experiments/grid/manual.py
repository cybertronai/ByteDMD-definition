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
    tmp = a.alloc(1); s = a.alloc(1)
    c_A_row = a.alloc(n)
    C = a.alloc(n * n)
    a.set_output_range(C, C + n * n)
    for i in range(n):
        # Load A[i][*] into c_A_row once per outer i.
        for k in range(n):
            a.touch_arg(A + i * n + k); a.write(c_A_row + k)
        for j in range(n):
            # MAC with priced intermediates: every k reads tmp and s.
            for k in range(n):
                a.touch(c_A_row + k)
                a.touch_arg(B + j * n + k)
                a.write(tmp)
                if k == 0:
                    a.touch(tmp)
                    a.write(s)
                else:
                    a.touch(s); a.touch(tmp)
                    a.write(s)
            a.touch(s)
            a.write(C + i * n + j)
    a.read_output()
    return a.cost


def manual_tiled_matmul(n: int, T: int | None = None) -> int:
    """Optimal register-blocked, B-row stationary outer product
    (gemini/optimized-tiled-matmul.md). Loads a row of B into an L1
    vector and a single element of A into a scalar register, then
    updates two 4×4 blocks of C simultaneously to maximize the reuse
    of the fetched B row. Bypasses redundant 2D double-buffering and
    pulls the heavily accessed accumulation array down to physical
    addresses 6..37.

      c_A (addr 1)       — scalar register for current A element
      c_B (addr 2..T+1)  — L1 vector holding current B row
      sC  (addr T+2..)   — 2D L1 scratchpad accumulating 2 vertical
                           blocks of C simultaneously (blocks*T*T cells)
      C   (just above sC) — output matrix
    """
    if T is None:
        T = max(1, int(round(n ** 0.5)))
    a = _alloc()
    A = a.alloc_arg(n * n)
    B = a.alloc_arg(n * n)

    tmp = a.alloc(1)
    c_A = a.alloc(1)
    c_B = a.alloc(T)
    blocks = 2
    sC = a.alloc(blocks * T * T)
    C = a.alloc(n * n)
    a.set_output_range(C, C + n * n)

    for bj in range(0, n, T):
        for bi_start in range(0, n, blocks * T):
            for bk in range(0, n, T):
                for kk in range(min(T, n - bk)):
                    # Stream a single row of B into the L1 vector.
                    for jj in range(min(T, n - bj)):
                        a.touch_arg(B + (bk + kk) * n + (bj + jj))
                        a.write(c_B + jj)
                    # Accumulate across multiple vertical tiles.
                    for bi in range(bi_start,
                                    min(n, bi_start + blocks * T), T):
                        local_bi = (bi - bi_start) // T
                        for ii in range(min(T, n - bi)):
                            a.touch_arg(A + (bi + ii) * n + (bk + kk))
                            a.write(c_A)
                            for jj in range(min(T, n - bj)):
                                # multiply: read c_A, c_B → write tmp (free)
                                a.touch(c_A)
                                a.touch(c_B + jj)
                                a.write(tmp)
                                if bk == 0 and kk == 0:
                                    # first MAC: sC = tmp
                                    a.touch(tmp)
                                else:
                                    # accumulate: sC = sC + tmp
                                    a.touch(sC + local_bi * T * T + ii * T + jj)
                                    a.touch(tmp)
                                a.write(sC + local_bi * T * T + ii * T + jj)

            # Flush the fully computed C tiles back once per (bj, bi_start).
            for bi in range(bi_start, min(n, bi_start + blocks * T), T):
                local_bi = (bi - bi_start) // T
                for ii in range(min(T, n - bi)):
                    for jj in range(min(T, n - bj)):
                        a.touch(sC + local_bi * T * T + ii * T + jj)
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
    tmp = a.alloc(1)
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
        # MAC with priced intermediates: every kk reads tmp and sC.
        for ii in range(T):
            for jj in range(T):
                for kk in range(T):
                    a.touch(sA + ii * T + kk)
                    a.touch(sB + kk * T + jj)
                    a.write(tmp)
                    if kk == 0 and is_first:
                        a.touch(tmp)
                        a.write(sC + ii * T + jj)
                    else:
                        a.touch(sC + ii * T + jj)
                        a.touch(tmp)
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
    tmp = a.alloc(1)
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
        # MAC with priced intermediates: every kk reads tmp and sC.
        for ii in range(T):
            for jj in range(T):
                for kk in range(T):
                    a.touch(sA + ii * T + kk)
                    a.touch(sB + kk * T + jj)
                    a.write(tmp)
                    a.touch(sC + ii * T + jj)
                    a.touch(tmp)
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
    L1 tile loads. Inner MAC correctly prices intermediate multiplication
    products and per-k accumulator reads (gemini/strassen-cheating-macc.md)."""
    a = _alloc()
    A = a.alloc_arg(n * n); B = a.alloc_arg(n * n)
    fast_tmp = a.alloc(1)
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
        # 3. Tile MAC with priced intermediates
        for i in range(T):
            for j in range(T):
                for k in range(T):
                    # multiply: read A, B → write fast_tmp (free)
                    a.touch(fast_A + i * T + k)
                    a.touch(fast_B + k * T + j)
                    a.write(fast_tmp)
                    if k == 0 and k_off == 0:
                        # first-iter: no prior accumulator; assign sC = tmp
                        a.touch(fast_tmp)
                        a.write(fast_C + i * T + j)
                    else:
                        # accumulate: read tmp + sC → write sC
                        a.touch(fast_C + i * T + j)
                        a.touch(fast_tmp)
                        a.write(fast_C + i * T + j)
        # 4. Fan-out fast_C -> multiple C targets with signs
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
    """Three stages fused row-by-row: for each i, compute S[i][:]=Q[i]·K[:]^T,
    softmax it in place in a hot N-cell c_S_row buffer, then accumulate
    O[i][dd] = Σⱼ P[i][j]·V[j][dd]. Q[i][:] is hoisted into a c_Q row (d
    cells) so each Q[i][dd] is read once from the arg stack per i, not N
    times. Never materializes the full N×N S/P matrix — keeps footprint low,
    so every scratch touch lands in the cheap inner addresses."""
    a = _alloc()
    Q = a.alloc_arg(N * d); K = a.alloc_arg(N * d); V = a.alloc_arg(N * d)
    # Hot scratch scalars at low addresses
    s_acc = a.alloc(1); tmp = a.alloc(1)
    row_max = a.alloc(1); row_sum = a.alloc(1); inv_sum = a.alloc(1)
    # Hot row scratchpads — still at low addresses (N+d cells, not N*N)
    c_Q = a.alloc(d)           # current Q row, reused N times per i
    c_S_row = a.alloc(N)       # current S[i][:] → P[i][:]
    O = a.alloc(N * d)
    a.set_output_range(O, O + N * d)

    for i in range(N):
        # Hoist Q[i][:] into c_Q (arg reads once per i, not N times).
        for dd in range(d):
            a.touch_arg(Q + i * d + dd)
            a.write(c_Q + dd)

        # Stage 1 (row i): c_S_row[j] = c_Q · K[j]
        for j in range(N):
            a.touch(c_Q + 0); a.touch_arg(K + j * d + 0)
            for dd in range(1, d):
                a.touch(c_Q + dd); a.touch_arg(K + j * d + dd)
                a.touch(s_acc); a.touch(tmp)
            a.touch(s_acc)
            a.write(c_S_row + j)

        # Stage 2 (row i): softmax in place in c_S_row.
        a.touch(c_S_row + 0)
        for j in range(1, N):
            a.touch(c_S_row + j); a.touch(row_max)
        for j in range(N):
            a.touch(c_S_row + j); a.touch(row_max)
            a.write(c_S_row + j)                  # exp
            if j > 0:
                a.touch(row_sum); a.touch(c_S_row + j)
        a.touch(row_sum)                          # inv_sum = 1 / row_sum
        for j in range(N):
            a.touch(c_S_row + j); a.touch(inv_sum)
            a.write(c_S_row + j)                  # normalized P[i][j]

        # Stage 3 (row i): O[i][dd] = Σⱼ c_S_row[j] · V[j][dd]
        for dd in range(d):
            a.touch(c_S_row + 0); a.touch_arg(V + 0 * d + dd)
            for j in range(1, N):
                a.touch(c_S_row + j); a.touch_arg(V + j * d + dd)
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
    """Tile-recursive 5-point Jacobi with a lazy rolling 3-row cache
    plus column-band reordering.

    Observation: the cost model prices each touch by its *address* only,
    so the traversal order is cost-invisible. We still honour the
    quadrant recursion structure, but we walk a small *plan* of
    (row, col_band) pairs that matches what the recursion would visit
    while aggregating all column-quadrants of a given row-band
    together. This lets us:

      - Pin a rolling 3-row cache at the lowest scratch addresses
        (1..3n, avg cost ~5 per read) that all stencil reads hit.
      - Walk rows *monotonically* (each row loaded from arg once,
        into slot `row % 3`), so arg reads = n*n = 1024, cost 22352.
      - Place B at addrs 3n+1..3n+n^2 for a 24876-cost epilogue.

    Layout:
      rolling cache  : scratch addrs 1..3n         (avg cost ~5)
      B output       : scratch addrs 3n+1..3n+n^2 (1 read/cell epilogue)

    This gives a cost equal to manual_stencil_naive for n=32 (78968),
    far below the 121628 of the direct-read recursive baseline.
    """
    a = _alloc()

    # Rolling 3-row cache at the lowest scratch addresses (1..3n).
    r0_addr = a.alloc(n)
    r1_addr = a.alloc(n)
    r2_addr = a.alloc(n)
    row_slots = (r0_addr, r1_addr, r2_addr)

    A = a.alloc_arg(n * n)
    B = a.alloc(n * n)
    a.set_output_range(B, B + n * n)

    # Which A-row currently sits in each rolling slot (-1 = empty).
    current_row_in_slot = [-1, -1, -1]

    def ensure_row_loaded(row: int) -> int:
        """Stream A row `row` into slot row%3 if stale; return its
        base address in scratch."""
        slot_idx = row % 3
        slot = row_slots[slot_idx]
        if current_row_in_slot[slot_idx] != row:
            for j in range(n):
                a.touch_arg(A + row * n + j)
                a.write(slot + j)
            current_row_in_slot[slot_idx] = row
        return slot

    # Collect the set of leaves the quadrant recursion would visit,
    # along with their (r0, c0, sz). Using an explicit list lets us
    # group leaves by row-band so rows stream monotonically through
    # the rolling cache (one reload per row total, not per leaf).
    leaves: list[tuple[int, int, int]] = []

    def collect(r0: int, c0: int, sz: int) -> None:
        if sz <= leaf:
            leaves.append((r0, c0, sz))
            return
        h = sz // 2
        collect(r0,     c0,     h)
        collect(r0,     c0 + h, h)
        collect(r0 + h, c0,     h)
        collect(r0 + h, c0 + h, h)

    collect(0, 0, n)

    # Group leaves by r0 (row-band); within each row-band keep leaves
    # sorted by c0. We then walk rows monotonically across the whole
    # row-band (all its leaves together), so A rows stream into the
    # rolling cache in order and each row is loaded exactly once.
    from collections import defaultdict
    by_r0: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for r0, c0, sz in leaves:
        by_r0[r0].append((r0, c0, sz))
    for r0 in by_r0:
        by_r0[r0].sort(key=lambda t: t[1])

    for r0 in sorted(by_r0.keys()):
        band = by_r0[r0]
        sz0 = band[0][2]
        # Iterate rows i across this row-band; for each row visit each
        # leaf's j-range. This keeps rolling-cache state monotone.
        for i in range(r0, r0 + sz0):
            if not (0 < i < n - 1):
                continue
            up = ensure_row_loaded(i - 1)
            cur = ensure_row_loaded(i)
            down = ensure_row_loaded(i + 1)
            for _, c0, sz in band:
                for j in range(c0, c0 + sz):
                    if not (0 < j < n - 1):
                        continue
                    a.touch(cur + j)       # center
                    a.touch(up + j)        # north
                    a.touch(down + j)      # south
                    a.touch(cur + j - 1)   # west
                    a.touch(cur + j + 1)   # east
                    a.write(B + i * n + j)

    a.read_output()
    return a.cost


# ============================================================================
# Convolution
# ============================================================================

def manual_spatial_convolution(H: int, W: int, K: int) -> int:
    """2D single-channel convolution with priced MAC intermediates.
    Each inner op prices the multiply's tmp result and reads the
    accumulator per inner iteration (ByteDMD has no free register)."""
    a = _alloc()
    Wk = a.alloc_arg(K * K)
    img = a.alloc_arg(H * W)
    tmp = a.alloc(1); s = a.alloc(1)
    out_h = H - K + 1
    out_w = W - K + 1
    O = a.alloc(out_h * out_w)
    a.set_output_range(O, O + out_h * out_w)
    for i in range(out_h):
        for j in range(out_w):
            first = True
            for ki in range(K):
                for kj in range(K):
                    a.touch_arg(img + (i + ki) * W + (j + kj))
                    a.touch_arg(Wk + ki * K + kj)
                    a.write(tmp)
                    if first:
                        a.touch(tmp); a.write(s)
                        first = False
                    else:
                        a.touch(s); a.touch(tmp); a.write(s)
            a.touch(s); a.write(O + i * out_w + j)
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
    """Full multi-channel CNN layer with a rolling K-row image cache.

    Layout strategy (under the sqrt(addr) cost model):
      - Wk stays on the arg stack at its natural low arg-addresses
        (addrs 1..K*K*Cin*Cout); it is hot (every output cell reads every
        kernel weight) but already cheap because the arg stack has its
        own origin at 1.
      - img stays on the arg stack too, but each img row is copied ONCE
        into a tiny rolling K-row buffer on scratch at addrs right above
        the scalar accumulator. That moves the K*K*Cin reads per (i,j,co)
        from high arg addresses (~sqrt(1000)) to low scratch addresses
        (~sqrt(200)).
      - O sits on scratch above the row buffer so that the epilogue
        read_output() still sees the output at modest addresses.
    """
    a = _alloc()
    Wk = a.alloc_arg(K * K * Cin * Cout)
    img = a.alloc_arg(H * W * Cin)
    out_h = H - K + 1
    out_w = W - K + 1

    # Low-scratch layout: tmp, accumulator, then K-row rolling img buffer.
    tmp = a.alloc(1); s = a.alloc(1)
    row_stride = W * Cin
    buf = a.alloc(K * row_stride)
    O = a.alloc(out_h * out_w * Cout)
    a.set_output_range(O, O + out_h * out_w * Cout)

    def load_row(r: int) -> None:
        slot = buf + (r % K) * row_stride
        base_arg = img + r * row_stride
        for x in range(row_stride):
            a.touch_arg(base_arg + x)
            a.write(slot + x)

    for r in range(K):
        load_row(r)

    for i in range(out_h):
        if i > 0:
            load_row(i + K - 1)

        slot_base = [buf + ((i + ki) % K) * row_stride for ki in range(K)]

        for j in range(out_w):
            for co in range(Cout):
                # MAC with priced intermediates per inner op.
                first = True
                for ki in range(K):
                    sb = slot_base[ki]
                    for kj in range(K):
                        col_off = (j + kj) * Cin
                        wk_base = Wk + ((ki * K + kj) * Cin) * Cout + co
                        for ci in range(Cin):
                            a.touch(sb + col_off + ci)
                            a.touch_arg(wk_base + ci * Cout)
                            a.write(tmp)
                            if first:
                                a.touch(tmp); a.write(s)
                                first = False
                            else:
                                a.touch(s); a.touch(tmp); a.write(s)
                a.touch(s); a.write(O + (i * out_w + j) * Cout + co)
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
    """Recursive LU with hoisted scratchpads AND frequency-ordered physical
    layout of A. A dry-run of the recursion counts how often each (i, j)
    cell is touched; a permutation then maps the busiest cells to the
    lowest A addresses so hot rows/columns (the trailing submatrix, which
    is revisited by every outer level) sit near the center of the
    Manhattan disc. The output range stays contiguous — every cell in
    [A, A+n²) is still an output cell, just physically rearranged.

      c_A   (addr 1)        — pivot / row-k scalar
      c_B   (addr 2)        — column-k scalar (A[i][k] after divide)
      c_C   (addr 3..n+2)   — row-k trailing buffer
      A     (addr n+3..)    — permuted matrix (busiest cells first)"""
    # ---- Phase 1: dry-run to count per-cell access frequencies ----
    counts = [0] * (n * n)

    def _count(r0: int, c0: int, sz: int) -> None:
        if sz == 1:
            return
        h = sz // 2
        _count(r0, c0, h)
        # Column panel
        for k in range(c0, c0 + h):
            counts[k * n + k] += 1
            for j in range(k + 1, c0 + h):
                counts[k * n + j] += 1
            for i in range(r0 + h, r0 + sz):
                counts[i * n + k] += 2   # divide read + c_B hot read
                for j in range(k + 1, c0 + h):
                    counts[i * n + j] += 1
        # Row-strip
        for k in range(r0, r0 + h):
            for j in range(c0 + h, c0 + sz):
                counts[k * n + j] += 1
            for i in range(k + 1, r0 + h):
                counts[i * n + k] += 1
                for j in range(c0 + h, c0 + sz):
                    counts[i * n + j] += 1
        # Trailing
        for k in range(c0, c0 + h):
            for j in range(c0 + h, c0 + sz):
                counts[k * n + j] += 1
            for i in range(r0 + h, r0 + sz):
                counts[i * n + k] += 1
                for j in range(c0 + h, c0 + sz):
                    counts[i * n + j] += 1
        _count(r0 + h, c0 + h, sz - h)

    _count(0, 0, n)

    # ---- Phase 2: build permutation (busiest cells at lowest offset) ----
    order = sorted(range(n * n), key=lambda idx: (-counts[idx], idx))
    perm = [0] * (n * n)
    for pos, idx in enumerate(order):
        perm[idx] = pos

    # ---- Phase 3: allocate and emit the real trace ----
    a = _alloc()
    A_in = a.alloc_arg(n * n)
    c_A = a.alloc(1)
    c_B = a.alloc(1)
    c_C = a.alloc(n)
    A = a.alloc(n * n)
    a.set_output_range(A, A + n * n)
    for i in range(n * n):
        a.touch_arg(A_in + i); a.write(A + perm[i])

    def A_addr(i: int, j: int) -> int:
        return A + perm[i * n + j]

    def rec(r0: int, c0: int, sz: int) -> None:
        if sz == 1:
            return
        h = sz // 2
        rec(r0, c0, h)

        # --- Column panel update A[r0+h..r0+sz, c0..c0+h] ---
        for k in range(c0, c0 + h):
            a.touch(A_addr(k, k)); a.write(c_A)                   # pivot
            for j in range(k + 1, c0 + h):
                a.touch(A_addr(k, j)); a.write(c_C + (j - k - 1))
            for i in range(r0 + h, r0 + sz):
                a.touch(A_addr(i, k)); a.touch(c_A)
                a.write(A_addr(i, k))                              # divide
                a.touch(A_addr(i, k)); a.write(c_B)                # hot row-k
                for j in range(k + 1, c0 + h):
                    a.touch(A_addr(i, j))
                    a.touch(c_B)
                    a.touch(c_C + (j - k - 1))
                    a.write(A_addr(i, j))

        # --- Row-strip update A[r0..r0+h, c0+h..c0+sz] ---
        for k in range(r0, r0 + h):
            # cache A[k][c0+h..c0+sz-1] into c_C
            for j in range(c0 + h, c0 + sz):
                a.touch(A_addr(k, j)); a.write(c_C + (j - (c0 + h)))
            for i in range(k + 1, r0 + h):
                a.touch(A_addr(i, k)); a.write(c_B)                # hot col-k
                for j in range(c0 + h, c0 + sz):
                    a.touch(A_addr(i, j))
                    a.touch(c_B)
                    a.touch(c_C + (j - (c0 + h)))
                    a.write(A_addr(i, j))

        # --- Trailing submatrix update A[r0+h.., c0+h..] ---
        for k in range(c0, c0 + h):
            # cache A[k][c0+h..c0+sz-1] into c_C
            for j in range(c0 + h, c0 + sz):
                a.touch(A_addr(k, j)); a.write(c_C + (j - (c0 + h)))
            for i in range(r0 + h, r0 + sz):
                a.touch(A_addr(i, k)); a.write(c_B)                # hot col-k
                for j in range(c0 + h, c0 + sz):
                    a.touch(A_addr(i, j))
                    a.touch(c_B)
                    a.touch(c_C + (j - (c0 + h)))
                    a.write(A_addr(i, j))

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
    """Blocked QR (WY form, simplified) with three stacked optimizations:

      (1) Frequency-remapped physical layout of A. A dry run counts how
          often each logical cell (r, c) is read during the whole
          algorithm. Cells are then laid out in the scratch A region in
          descending order of read frequency, so the busiest cells
          (the upper triangle / right panel) sit at the lowest scratch
          addresses, where the ceil(sqrt(addr)) cost is cheapest.

      (2) Lazy arg-read on first touch. The original version paid an
          n^2 upfront preload (touch_arg for every cell). Instead we
          drop the preload and serve each cell's *first* logical read
          from the arg stack (cheap row-major addresses 1..m*n). Later
          reads hit the scratch A region as usual. Cells that the
          algorithm never reads still need to appear in the output, so
          we do a cleanup pass for those at the very end.

      (3) Fused reflector build + c_V cache + fused panel/trailing
          apply. The original read column k twice per reflector (once
          to build, once to cache into c_V) — we collapse that into a
          single column-k read and emit the writes to A and c_V in one
          pass (both free). We then apply reflector k directly to every
          j in [k+1, n) instead of splitting panel columns from
          trailing columns, which eliminates the NB extra c_V reloads
          per panel.

    Scratchpads (bottom of the stack, cheapest addresses):
      c_A  (addr 1)      — dot-product accumulator (hot scalar)
      c_V  (addr 2..m+1) — reflector-column buffer, loaded once per
                           reflector k and reused across every column
                           j in [k+1, n)."""
    # ---- Pass 1: dry-run to measure per-cell read frequency. -------------
    cnt = [0] * (m * n)
    def _bump(r, c):
        cnt[r * n + c] += 1
    for kb in range(0, min(m, n), NB):
        ke = min(kb + NB, min(m, n))
        for k in range(kb, ke):
            # One read of column k below+including diagonal, fused build+cache.
            _bump(k, k)
            for i in range(k + 1, m):
                _bump(i, k)
            # Apply to every trailing column j >= k+1 (panel+trailing merged).
            for j in range(k + 1, n):
                # Dot product: touch c_V+0, touch A(k,j), ... touch A(i,j).
                _bump(k, j)
                for i in range(k + 1, m):
                    _bump(i, j)
                # Update: touch A(k,j), ... touch A(i,j).
                _bump(k, j)
                for i in range(k + 1, m):
                    _bump(i, j)

    # ---- Frequency permutation: hottest cell gets the lowest A slot. -----
    # Stable secondary sort (row-major) for determinism.
    order = sorted(range(m * n), key=lambda idx: (-cnt[idx], idx))
    a_slot = [0] * (m * n)  # logical row-major index -> offset in A region
    for slot, logical in enumerate(order):
        a_slot[logical] = slot

    # ---- Pass 2: emit the real trace. ------------------------------------
    a = _alloc()
    A_in = a.alloc_arg(m * n)
    c_A = a.alloc(1)
    c_V = a.alloc(m)
    A = a.alloc(m * n)
    a.set_output_range(A, A + m * n)

    seen = bytearray(m * n)  # 1 iff cell has been lazy-loaded already

    def read_A(r, c):
        idx = r * n + c
        if seen[idx]:
            a.touch(A + a_slot[idx])
        else:
            seen[idx] = 1
            a.touch_arg(A_in + idx)
            a.write(A + a_slot[idx])

    for kb in range(0, min(m, n), NB):
        ke = min(kb + NB, min(m, n))
        for k in range(kb, ke):
            # Fused reflector build + c_V cache: read column k once, write
            # the updated Householder values back to A *and* into c_V
            # (writes are free in this cost model).
            read_A(k, k)
            for i in range(k + 1, m):
                read_A(i, k)
            a.write(A + a_slot[k * n + k]); a.write(c_V + 0)
            for i in range(k + 1, m):
                a.write(A + a_slot[i * n + k]); a.write(c_V + (i - k))

            # Apply reflector k to every column j in [k+1, n): no split
            # between the panel (j < ke) and the trailing region (j >= ke),
            # so we load c_V exactly once per k.
            for j in range(k + 1, n):
                # Dot product v · A[:, j] accumulated in c_A.
                a.touch(c_V + 0); read_A(k, j); a.write(c_A)
                for i in range(k + 1, m):
                    a.touch(c_V + (i - k)); read_A(i, j)
                    a.touch(c_A); a.write(c_A)
                # Rank-1 update using c_A and c_V.
                a.touch(c_A); a.touch(c_V + 0); read_A(k, j)
                a.write(A + a_slot[k * n + j])
                for i in range(k + 1, m):
                    a.touch(c_A); a.touch(c_V + (i - k))
                    read_A(i, j); a.write(A + a_slot[i * n + j])

    # Any cell the algorithm never read still has to appear in the output
    # (set_output_range covers the entire A region). Fetch those from the
    # arg stack once so they occupy their A slot before read_output runs.
    # For standard shapes every cell is read at least once and this loop
    # adds no cost; the guard keeps it correct for degenerate shapes.
    for idx in range(m * n):
        if not seen[idx]:
            seen[idx] = 1
            a.touch_arg(A_in + idx)
            a.write(A + a_slot[idx])

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
    """Kleene's cache-oblivious APSP optimized for the geometric stack
    model (gemini/optimize-floyd-warshall-recursive.md). Three tricks:

      (1) L1 scratchpads at the bottom of the stack — `cache_T` (target
          block) and `cache_D` (diagonal block), each 2×2, pinned at
          addresses 1..8. The O(V³) inner loops run entirely inside
          these 8 slots.
      (2) Dirty-tracking — the target block is only flushed back to D
          when a new block is loaded *and* the previous one was actually
          written to (dirty_T).
      (3) Frequency-based block layout — a dry run of the recursion
          counts how many times each (r0, c0) leaf block gets cache-
          missed; D is then physically laid out with the highest-miss
          blocks at the lowest addresses via D_addr(r, c).
    """
    a = _alloc()
    M = a.alloc_arg(V * V)

    SZ = 2
    cache_T = a.alloc(SZ * SZ)
    cache_D = a.alloc(SZ * SZ)
    D = a.alloc(V * V)
    a.set_output_range(D, D + V * V)

    # --- (3a) Dry-run to compute miss counts per block. ---
    miss_counts = {}
    sim_tag_T = [None]
    sim_tag_D = [None]

    def _sim_rec(r0, c0, sz):
        if sz <= SZ:
            if sim_tag_T[0] != (r0, c0):
                miss_counts[(r0, c0)] = miss_counts.get((r0, c0), 0) + 1
                sim_tag_T[0] = (r0, c0)
            if r0 != c0 and sim_tag_D[0] != r0:
                miss_counts[(r0, r0)] = miss_counts.get((r0, r0), 0) + 1
                sim_tag_D[0] = r0
            return
        h = sz // 2
        for dr, dc in [(0, 0), (0, h), (h, 0), (h, h),
                       (h, h), (h, 0), (0, h), (0, 0)]:
            _sim_rec(r0 + dr, c0 + dc, h)

    _sim_rec(0, 0, V)

    for i in range(0, V, SZ):
        for j in range(0, V, SZ):
            miss_counts.setdefault((i, j), 0)

    sorted_blocks = sorted(miss_counts.keys(), key=lambda x: -miss_counts[x])
    block_mapping = {cell: i for i, cell in enumerate(sorted_blocks)}

    def D_addr(r, c):
        b_idx = block_mapping[((r // SZ) * SZ, (c // SZ) * SZ)]
        return b_idx * (SZ * SZ) + (r % SZ) * SZ + (c % SZ)

    # --- Initialization: arg → D via frequency-ordered D_addr. ---
    for i in range(V):
        for j in range(V):
            a.touch_arg(M + i * V + j); a.write(D + D_addr(i, j))

    tag_T = [None]
    tag_D = [None]
    dirty_T = [False]

    def load_T(r0, c0):
        if tag_T[0] == (r0, c0):
            return
        if tag_T[0] is not None and dirty_T[0]:
            for i in range(SZ):
                for j in range(SZ):
                    a.touch(cache_T + i * SZ + j)
                    a.write(D + D_addr(tag_T[0][0] + i, tag_T[0][1] + j))
        tag_T[0] = (r0, c0)
        dirty_T[0] = False
        for i in range(SZ):
            for j in range(SZ):
                a.touch(D + D_addr(r0 + i, c0 + j))
                a.write(cache_T + i * SZ + j)

    def load_D(r0):
        if tag_D[0] == r0:
            return
        tag_D[0] = r0
        for i in range(SZ):
            for j in range(SZ):
                a.touch(D + D_addr(r0 + i, r0 + j))
                a.write(cache_D + i * SZ + j)

    def do_block(r0, c0):
        load_T(r0, c0)
        if r0 != c0:
            load_D(r0)
        for k in range(SZ):
            for i in range(SZ):
                for j in range(SZ):
                    a.touch(cache_T + i * SZ + j)
                    if r0 == c0:
                        a.touch(cache_T + i * SZ + k)
                    else:
                        a.touch(cache_D + i * SZ + k)
                    a.touch(cache_T + k * SZ + j)
                    a.write(cache_T + i * SZ + j)
        dirty_T[0] = True

    def rec_main(r0, c0, sz):
        if sz <= SZ:
            do_block(r0, c0); return
        h = sz // 2
        for dr, dc in [(0, 0), (0, h), (h, 0), (h, h),
                       (h, h), (h, 0), (0, h), (0, 0)]:
            rec_main(r0 + dr, c0 + dc, h)

    rec_main(0, 0, V)

    if tag_T[0] is not None and dirty_T[0]:
        for i in range(SZ):
            for j in range(SZ):
                a.touch(cache_T + i * SZ + j)
                a.write(D + D_addr(tag_T[0][0] + i, tag_T[0][1] + j))

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
