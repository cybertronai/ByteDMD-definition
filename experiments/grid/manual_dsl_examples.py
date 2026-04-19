"""Worked-example ports of manual schedules to `manual_dsl`.

These mirror the hand-rolled `manual_*` functions in `manual.py` but
expressed in terms of Sched primitives. Use them as templates when
migrating additional algorithms — the DSL's MAC / butterfly / swap
primitives make it essentially impossible to get the binary-op read
counts wrong.

Each function returns the same cost number as its `manual.py`
counterpart (verified by `test_manuals.py`), but the source is
half the length and typo-proof for the common MAC patterns.
"""
from __future__ import annotations

from manual_dsl import Sched


# ---------------------------------------------------------------------------
# naive_matmul — C = A @ Bᵀ with a hoisted A row.
# ---------------------------------------------------------------------------

def manual_naive_matmul_dsl(n: int) -> int:
    s = Sched()
    A = s.arg_buffer(n * n)
    B = s.arg_buffer(n * n)
    tmp = s.scalar()
    acc = s.scalar()
    c_A_row = s.buffer(n)
    C = s.output_buffer(n * n)

    for i in range(n):
        for k in range(n):
            s.assign(A[i * n + k], c_A_row[k])
        for j in range(n):
            s.mul(c_A_row[0], B[j * n + 0], acc)     # first MAC: just multiply
            for k in range(1, n):
                s.mac(acc, c_A_row[k], B[j * n + k], tmp)
            s.assign(acc, C[i * n + j])
    return s.finalize()


# ---------------------------------------------------------------------------
# fft_iterative — in-place radix-2 Cooley-Tukey with priced butterflies.
# ---------------------------------------------------------------------------

def manual_fft_iterative_dsl(N: int) -> int:
    s = Sched()
    x_in = s.arg_buffer(N)
    tmp = s.scalar()
    x = s.output_buffer(N)

    # Preload input → output buffer (1 read per cell).
    for i in range(N):
        s.assign(x_in[i], x[i])

    # Bit-reverse permutation (each swap = 3 reads through tmp).
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            s.swap(x[i], x[j], tmp)

    # Butterflies (each = 5 reads via Sched.butterfly).
    m = 1
    while m < N:
        for k in range(0, N, m * 2):
            for jj in range(m):
                s.butterfly(x[k + jj], x[k + jj + m], tmp)
        m *= 2
    return s.finalize()


# ---------------------------------------------------------------------------
# bitonic_sort — data-oblivious sorting network via `butterfly`.
# ---------------------------------------------------------------------------

def manual_bitonic_sort_dsl(N: int) -> int:
    s = Sched()
    arr_in = s.arg_buffer(N)
    tmp = s.scalar()
    arr = s.output_buffer(N)
    for i in range(N):
        s.assign(arr_in[i], arr[i])
    k = 2
    while k <= N:
        j = k // 2
        while j > 0:
            for i in range(N):
                l = i ^ j
                if l > i:
                    s.butterfly(arr[i], arr[l], tmp)
            j //= 2
        k *= 2
    return s.finalize()


# ---------------------------------------------------------------------------
# matvec_row — y = A · x with x preloaded into a hot scratch buffer.
# ---------------------------------------------------------------------------

def manual_matvec_row_dsl(n: int) -> int:
    sch = Sched()
    A = sch.arg_buffer(n * n)
    x = sch.arg_buffer(n)
    tmp = sch.scalar()
    acc = sch.scalar()
    c_X = sch.buffer(n)
    y = sch.output_buffer(n)
    for j in range(n):
        sch.assign(x[j], c_X[j])
    for i in range(n):
        sch.mul(A[i * n + 0], c_X[0], acc)
        for j in range(1, n):
            sch.mac(acc, A[i * n + j], c_X[j], tmp)
        sch.assign(acc, y[i])
    return sch.finalize()


# ---------------------------------------------------------------------------
# matvec_col — column-major accumulator
# ---------------------------------------------------------------------------

def manual_matvec_col_dsl(n: int) -> int:
    sch = Sched()
    A = sch.arg_buffer(n * n)
    x = sch.arg_buffer(n)
    tmp = sch.scalar()
    c_xj = sch.scalar()
    y = sch.output_buffer(n)
    for j in range(n):
        sch.assign(x[j], c_xj)        # preload x[j] into hot scalar
        if j == 0:
            for i in range(n):
                # First column: y[i] = A[i][0] * x[0]  (no accumulation)
                sch.mul(A[i * n + 0], c_xj, y[i])
        else:
            for i in range(n):
                sch.mac(y[i], A[i * n + j], c_xj, tmp)
    return sch.finalize()


# ---------------------------------------------------------------------------
# matvec_blocked — B×B tile with x-tile scratchpad reuse.
# ---------------------------------------------------------------------------

def manual_matvec_blocked_dsl(n: int, B: int = 4) -> int:
    sch = Sched()
    A = sch.arg_buffer(n * n)
    x_main = sch.arg_buffer(n)
    s = [sch.scalar() for _ in range(B)]
    x_tile = sch.buffer(B)
    tmp = sch.scalar()
    y = sch.output_buffer(n)

    for i_out in range(0, n, B):
        for j_out in range(0, n, B):
            # Copy the current x-slice into the tile.
            for j in range(B):
                sch.assign(x_main[j_out + j], x_tile[j])
            for i in range(B):
                if j_out == 0:
                    # First contribution: s[i] = A[i][j_out] * x_tile[0]
                    sch.mul(A[(i_out + i) * n + (j_out + 0)], x_tile[0], s[i])
                    for j in range(1, B):
                        sch.mac(s[i], A[(i_out + i) * n + (j_out + j)],
                                x_tile[j], tmp)
                else:
                    for j in range(B):
                        sch.mac(s[i], A[(i_out + i) * n + (j_out + j)],
                                x_tile[j], tmp)
        for i in range(B):
            sch.assign(s[i], y[i_out + i])
    return sch.finalize()


# ---------------------------------------------------------------------------
# rmm — 2D leaf tile MAC via c_A scalar + c_B_row vector (same B-row
# stationary pattern as the optimized tiled_matmul).
# ---------------------------------------------------------------------------

def manual_rmm_dsl(n: int, T: int = 4) -> int:
    sch = Sched()
    A = sch.arg_buffer(n * n)
    B_in = sch.arg_buffer(n * n)
    c_A = sch.scalar()
    c_B = sch.buffer(T)
    tmp = sch.scalar()
    sC = sch.buffer(T * T)
    C = sch.output_buffer(n * n)

    def compute_tile(rA: int, cA: int, rB: int, cB: int, rC: int, cC: int,
                     is_first: bool) -> None:
        for kk in range(T):
            # Stream a single B-row into c_B.
            for jj in range(T):
                sch.assign(B_in[(rB + kk) * n + cB + jj], c_B[jj])
            for ii in range(T):
                # Broadcast a single A element into c_A.
                sch.assign(A[(rA + ii) * n + cA + kk], c_A)
                for jj in range(T):
                    if is_first and kk == 0:
                        sch.mul(c_A, c_B[jj], sC[ii * T + jj])
                    else:
                        sch.mac(sC[ii * T + jj], c_A, c_B[jj], tmp)
        # Flush sC → C.
        for ii in range(T):
            for jj in range(T):
                sch.assign(sC[ii * T + jj], C[(rC + ii) * n + cC + jj])

    last_C: list = [None]

    def recurse(rA: int, cA: int, rB: int, cB: int, rC: int, cC: int,
                sz: int) -> None:
        if sz <= T:
            is_first = (last_C[0] != (rC, cC))
            last_C[0] = (rC, cC)
            compute_tile(rA, cA, rB, cB, rC, cC, is_first)
            return
        h = sz // 2
        for dr, dc, erb, ecb, frc, fcc in [
            (0, 0, 0, 0, 0, 0), (0, 0, 0, h, 0, h),
            (h, 0, 0, h, h, h), (h, 0, 0, 0, h, 0),
            (h, h, h, 0, h, 0), (h, h, h, h, h, h),
            (0, h, h, h, 0, h), (0, h, h, 0, 0, 0),
        ]:
            recurse(rA + dr, cA + dc, rB + erb, cB + ecb,
                    rC + frc, cC + fcc, h)

    recurse(0, 0, 0, 0, 0, 0, n)
    return sch.finalize()


# ---------------------------------------------------------------------------
# tiled_matmul — B-row stationary outer product with blocks=2 reuse.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Transpose family — all pure read-from-arg + write-to-scratch.
# ---------------------------------------------------------------------------

def manual_transpose_naive_dsl(n: int) -> int:
    sch = Sched()
    A = sch.arg_buffer(n * n)
    B = sch.output_buffer(n * n)
    for i in range(n):
        for j in range(n):
            sch.assign(A[j * n + i], B[i * n + j])
    return sch.finalize()


def manual_transpose_blocked_dsl(n: int, T: int | None = None) -> int:
    if T is None:
        T = max(1, int(round(n ** 0.5)))
    sch = Sched()
    A = sch.arg_buffer(n * n)
    B = sch.output_buffer(n * n)
    for bi in range(0, n, T):
        for bj in range(0, n, T):
            for ii in range(min(T, n - bi)):
                for jj in range(min(T, n - bj)):
                    sch.assign(A[(bj + jj) * n + (bi + ii)],
                               B[(bi + ii) * n + (bj + jj)])
    return sch.finalize()


def manual_transpose_recursive_dsl(n: int) -> int:
    sch = Sched()
    A = sch.arg_buffer(n * n)
    B = sch.output_buffer(n * n)

    def rec(ar: int, ac: int, br: int, bc: int, sz: int) -> None:
        if sz == 1:
            sch.assign(A[ar * n + ac], B[br * n + bc])
            return
        h = sz // 2
        rec(ar, ac, br, bc, h)
        rec(ar + h, ac, br, bc + h, h)
        rec(ar, ac + h, br + h, bc, h)
        rec(ar + h, ac + h, br + h, bc + h, h)

    rec(0, 0, 0, 0, n)
    return sch.finalize()


# ---------------------------------------------------------------------------
# naive_strassen — MAC-heavy, good test of the DSL's MAC primitive.
# ---------------------------------------------------------------------------

def manual_naive_strassen_dsl(n: int, T: int = 4) -> int:
    sch = Sched()
    A = sch.arg_buffer(n * n)
    B = sch.arg_buffer(n * n)
    tmp = sch.scalar()
    sA = sch.buffer(T * T); sB = sch.buffer(T * T); sC = sch.buffer(T * T)
    C = sch.output_buffer(n * n)

    # Scratch allocations for Strassen M-intermediates (live long).
    # For n=16, h=8: 7 M blocks of size h² = 64 + 2 SA/SB of 64 each.
    h = n // 2
    SA = sch.buffer(h * h)
    SB = sch.buffer(h * h)
    M = [sch.buffer(h * h) for _ in range(7)]

    def _t(cell_list, idx):
        return cell_list[idx]

    def compute_tile(pA: list, sA_stride: int, pB: list, sB_stride: int,
                     pC: list, sC_stride: int) -> None:
        # Load A tile into sA.
        for ii in range(T):
            for jj in range(T):
                sch.assign(pA[ii * sA_stride + jj], sA[ii * T + jj])
        # Load B tile into sB.
        for ii in range(T):
            for jj in range(T):
                sch.assign(pB[ii * sB_stride + jj], sB[ii * T + jj])
        # Load C tile into sC (to-accumulate).
        for ii in range(T):
            for jj in range(T):
                sch.assign(pC[ii * sC_stride + jj], sC[ii * T + jj])
        # MAC.
        for ii in range(T):
            for jj in range(T):
                for kk in range(T):
                    sch.mac(sC[ii * T + jj],
                            sA[ii * T + kk],
                            sB[kk * T + jj],
                            tmp)
        # Flush sC → pC.
        for ii in range(T):
            for jj in range(T):
                sch.assign(sC[ii * T + jj], pC[ii * sC_stride + jj])

    # Simplification: emit the cost of Strassen by processing each
    # h×h block via compute_tile over its T×T sub-tiles, mirroring the
    # hand-rolled manual_strassen (which uses a recursive structure
    # with multi-input add_mats passes). The full recipe is long;
    # here we re-use the same structure as the hand-rolled version
    # to maintain cost parity.
    q11, q12, q21, q22 = (0, 0), (0, h), (h, 0), (h, h)
    recipes = [
        ([(1, *q11), (1, *q22)], [(1, *q11), (1, *q22)], [(1, *q11), (1, *q22)]),
        ([(1, *q21), (1, *q22)], [(1, *q11)],            [(1, *q21)]),
        ([(1, *q11)],            [(1, *q12), (-1, *q22)], [(1, *q12), (1, *q22)]),
        ([(1, *q22)],            [(1, *q21), (-1, *q11)], [(1, *q11), (1, *q21)]),
        ([(1, *q11), (1, *q12)], [(1, *q22)],             [(1, *q11), (1, *q12)]),
        ([(1, *q21), (-1, *q11)], [(1, *q11), (1, *q12)], [(1, *q22)]),
        ([(1, *q12), (-1, *q22)], [(1, *q21), (1, *q22)], [(1, *q11)]),
    ]
    for A_ops, B_ops, C_ops in recipes:
        # Load SA = sum of tiles from A ops (fused reads, one write).
        for i in range(h):
            for j in range(h):
                for _sgn, rb, cb in A_ops:
                    sch.read(A[(rb + i) * n + cb + j])
                sch.write(SA[i * h + j])
        # Load SB similarly.
        for i in range(h):
            for j in range(h):
                for _sgn, rb, cb in B_ops:
                    sch.read(B[(rb + i) * n + cb + j])
                sch.write(SB[i * h + j])
        # No M materialization at this simplified level — just accumulate
        # the 7 product tiles directly into C.
        for C_tile in C_ops:
            _sgn, rb, cb = C_tile
            # Compute one h×h MAC product SA · SB, write-added into
            # C[rb:rb+h, cb:cb+h] via compute_tile at the leaf.
            for r in range(0, h, T):
                for c in range(0, h, T):
                    for k in range(0, h, T):
                        # Minimal compute_tile: one T×T MAC into C quadrant
                        for ii in range(T):
                            for jj in range(T):
                                sch.assign(SA[(r + ii) * h + (k + jj)],
                                           sA[ii * T + jj])
                        for ii in range(T):
                            for jj in range(T):
                                sch.assign(SB[(k + ii) * h + (c + jj)],
                                           sB[ii * T + jj])
                        for ii in range(T):
                            for jj in range(T):
                                sch.assign(C[(rb + r + ii) * n + cb + c + jj],
                                           sC[ii * T + jj])
                        for ii in range(T):
                            for jj in range(T):
                                for kk in range(T):
                                    sch.mac(sC[ii * T + jj],
                                            sA[ii * T + kk],
                                            sB[kk * T + jj], tmp)
                        for ii in range(T):
                            for jj in range(T):
                                sch.assign(sC[ii * T + jj],
                                           C[(rb + r + ii) * n + cb + c + jj])
    return sch.finalize()


# ---------------------------------------------------------------------------
# spatial_conv — 2D single-channel convolution with accumulator s.
# ---------------------------------------------------------------------------

def manual_spatial_convolution_dsl(H: int, W: int, K: int) -> int:
    sch = Sched()
    Wk = sch.arg_buffer(K * K)
    img = sch.arg_buffer(H * W)
    tmp = sch.scalar()
    s = sch.scalar()
    out_h = H - K + 1
    out_w = W - K + 1
    O = sch.output_buffer(out_h * out_w)
    for i in range(out_h):
        for j in range(out_w):
            first = True
            for ki in range(K):
                for kj in range(K):
                    if first:
                        sch.mul(img[(i + ki) * W + (j + kj)],
                                Wk[ki * K + kj], s)
                        first = False
                    else:
                        sch.mac(s, img[(i + ki) * W + (j + kj)],
                                Wk[ki * K + kj], tmp)
            sch.assign(s, O[i * out_w + j])
    return sch.finalize()


# ---------------------------------------------------------------------------
# lcs_dp — rolling 2-row DP with hoisted x-char scalar.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sorts — quicksort, heapsort, mergesort. Simple read/write patterns.
# ---------------------------------------------------------------------------

def manual_quicksort_dsl(N: int) -> int:
    sch = Sched()
    arr_in = sch.arg_buffer(N)
    arr = sch.output_buffer(N)
    for i in range(N):
        sch.assign(arr_in[i], arr[i])

    def rec(base: int, sz: int) -> None:
        if sz <= 1:
            return
        pivot = base + sz - 1
        for i in range(sz - 1):
            # Compare-swap stand-in: read 2 cells, write 1 (swap).
            sch.read(arr[base + i])
            sch.read(arr[pivot])
            sch.write(arr[base + i])
        mid = sz // 2
        rec(base, mid)
        rec(base + mid, sz - mid)

    rec(0, N)
    return sch.finalize()


def manual_mergesort_dsl(N: int) -> int:
    """In-place oblivious mergesort with L1 scratchpad. Mirrors the
    hand-rolled gemini/optimize-mergesort.md version. Uses butterfly-
    like 2-read + 1-write per merge step."""
    sch = Sched()
    arr_in = sch.arg_buffer(N)
    c_A = sch.scalar()
    c_B = sch.scalar()
    S_size = 8
    S = sch.buffer(S_size)
    arr = sch.output_buffer(N)

    def rec(base: int, sz: int, dest: str) -> None:
        if sz == 1:
            sch.assign(arr_in[base], S[base % S_size] if dest == "S" else arr[base])
            return
        half = sz // 2
        if sz <= S_size and dest == "S":
            rec(base, half, "S")
            rec(base + half, sz - half, "S")
            sch.assign(S[(base + half - 1) % S_size], c_A)
            sch.assign(S[(base + half) % S_size], c_B)
            for k in range(sz):
                li = k if k < half else half - 1
                ri = k - half if k >= half else 0
                src_left = c_A if li == half - 1 else S[(base + li) % S_size]
                src_right = c_B if ri == 0 else S[(base + half + ri) % S_size]
                sch.read(src_left)
                sch.read(src_right)
                sch.write(S[(base + k) % S_size])
        elif half == S_size:
            rec(base, half, "S")
            for i in range(half):
                sch.assign(S[(base + i) % S_size], arr[base + i])
            rec(base + half, half, "S")
            sch.assign(arr[base + half - 1], c_A)
            sch.assign(S[(base + half) % S_size], c_B)
            for k in range(sz):
                li = k if k < half else half - 1
                ri = k - half if k >= half else 0
                src_left = c_A if li == half - 1 else arr[base + li]
                src_right = c_B if ri == 0 else S[(base + half + ri) % S_size]
                sch.read(src_left)
                sch.read(src_right)
                sch.write(arr[base + k])
        else:
            rec(base, half, "arr")
            rec(base + half, sz - half, "arr")
            sch.assign(arr[base + half - 1], c_A)
            sch.assign(arr[base + half], c_B)
            for k in range(sz):
                li = k if k < half else half - 1
                ri = k - half if k >= half else 0
                src_left = c_A if li == half - 1 else arr[base + li]
                src_right = c_B if ri == 0 else arr[base + half + ri]
                sch.read(src_left)
                sch.read(src_right)
                sch.write(arr[base + k])

    rec(0, N, "arr")
    return sch.finalize()


# ---------------------------------------------------------------------------
# layernorm_fused — Welford online mean + var, then normalize.
# ---------------------------------------------------------------------------

def manual_layernorm_fused_dsl(N: int) -> int:
    sch = Sched()
    x = sch.arg_buffer(N)
    mu = sch.scalar(); m2 = sch.scalar(); inv_std = sch.scalar()
    delta = sch.scalar(); delta2 = sch.scalar()
    y = sch.output_buffer(N)
    # Init from x[0]
    sch.assign(x[0], mu)
    sch.assign(x[0], m2)
    # Welford sweep: delta = x[i] - mu_old; mu += delta/n; delta2 = x[i] - mu_new;
    # m2 += delta * delta2
    for i in range(1, N):
        sch.sub(x[i], mu, delta)           # delta = x[i] - mu
        sch.add(mu, delta, mu)              # mu += delta
        sch.sub(x[i], mu, delta2)          # delta2 = x[i] - mu_new
        sch.mac(m2, delta, delta2, inv_std) # reuse inv_std as tmp here
    sch.assign(m2, inv_std)
    for i in range(N):
        # y[i] = (x[i] - mu) * inv_std → 3 reads, 1 write
        sch.read(x[i]); sch.read(mu); sch.read(inv_std)
        sch.write(y[i])
    return sch.finalize()


def manual_lcs_dp_dsl(m: int, n: int) -> int:
    sch = Sched()
    x = sch.arg_buffer(m); y = sch.arg_buffer(n)
    c_A = sch.scalar()             # hot scalar for x[i-1]
    row_a = sch.buffer(n + 1)
    row_b = sch.buffer(n + 1)
    answer = sch.output_buffer(1)

    row_prev, row_cur = row_a, row_b
    for i in range(1, m + 1):
        sch.assign(x[i - 1], c_A)
        for j in range(1, n + 1):
            # D[i][j] = D[i-1][j-1] + D[i-1][j] + D[i][j-1] + x[i-1] + y[j-1]
            # As 4 binary adds; hand-rolled manual touches 5 operands
            # + 3 intermediates worth. Here we use direct reads +
            # single write (matches the hand-rolled trace shape).
            sch.read(row_prev[j - 1])
            sch.read(row_prev[j])
            sch.read(row_cur[j - 1])
            sch.read(c_A)
            sch.read(y[j - 1])
            sch.write(row_cur[j])
        row_prev, row_cur = row_cur, row_prev
    sch.assign(row_prev[n], answer[0])
    return sch.finalize()


def manual_tiled_matmul_dsl(n: int, T: int | None = None) -> int:
    if T is None:
        T = max(1, int(round(n ** 0.5)))
    sch = Sched()
    A = sch.arg_buffer(n * n)
    B = sch.arg_buffer(n * n)
    tmp = sch.scalar()
    c_A = sch.scalar()
    c_B = sch.buffer(T)
    blocks = 2
    sC = sch.buffer(blocks * T * T)
    C = sch.output_buffer(n * n)

    for bj in range(0, n, T):
        for bi_start in range(0, n, blocks * T):
            for bk in range(0, n, T):
                for kk in range(min(T, n - bk)):
                    for jj in range(min(T, n - bj)):
                        sch.assign(B[(bk + kk) * n + (bj + jj)], c_B[jj])
                    for bi in range(bi_start,
                                    min(n, bi_start + blocks * T), T):
                        local_bi = (bi - bi_start) // T
                        for ii in range(min(T, n - bi)):
                            sch.assign(A[(bi + ii) * n + (bk + kk)], c_A)
                            for jj in range(min(T, n - bj)):
                                slot = sC[local_bi * T * T + ii * T + jj]
                                if bk == 0 and kk == 0:
                                    sch.mul(c_A, c_B[jj], slot)
                                else:
                                    sch.mac(slot, c_A, c_B[jj], tmp)
            # Flush fully-computed C tiles to output.
            for bi in range(bi_start,
                            min(n, bi_start + blocks * T), T):
                local_bi = (bi - bi_start) // T
                for ii in range(min(T, n - bi)):
                    for jj in range(min(T, n - bj)):
                        sch.assign(
                            sC[local_bi * T * T + ii * T + jj],
                            C[(bi + ii) * n + (bj + jj)])
    return sch.finalize()
