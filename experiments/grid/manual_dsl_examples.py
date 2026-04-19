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


# ---------------------------------------------------------------------------
# fft_recursive — radix-2 Cooley-Tukey, priced butterflies at the merge.
# ---------------------------------------------------------------------------

def manual_fft_recursive_dsl(N: int) -> int:
    sch = Sched()
    x_in = sch.arg_buffer(N)
    tmp = sch.scalar()
    x = sch.output_buffer(N)

    def rec(base: int, sz: int, stride: int, offset: int) -> None:
        if sz == 1:
            sch.assign(x_in[offset], x[base])
            return
        rec(base, sz // 2, stride * 2, offset)
        rec(base + sz // 2, sz // 2, stride * 2, offset + stride)
        for k in range(sz // 2):
            sch.butterfly(x[base + k], x[base + sz // 2 + k], tmp)

    rec(0, N, 1, 0)
    return sch.finalize()


# ---------------------------------------------------------------------------
# heapsort — compare-swap bodies expressed as (read, read, write) pairs.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# fused_strassen — Zero-Allocation Fused Strassen (ZAFS). MAC pattern
# with fused arg→scratch loads and signed C fan-out. Stress test for
# the DSL's MAC primitive.
# ---------------------------------------------------------------------------

def manual_fused_strassen_dsl(n: int, T: int = 4) -> int:
    sch = Sched()
    A = sch.arg_buffer(n * n); B = sch.arg_buffer(n * n)
    tmp = sch.scalar()
    fast_A = sch.buffer(T * T); fast_B = sch.buffer(T * T)
    fast_C = sch.buffer(T * T)
    C = sch.output_buffer(n * n)

    def compute_fused_tile(ops_A, ops_B, ops_C, r, c, k_off):
        # 1. Fused arg→fast load of A tile (multi-operand add, 1 write).
        for i in range(T):
            for j in range(T):
                for _sgn, rb, cb in ops_A:
                    sch.read(A[(rb + r + i) * n + cb + k_off + j])
                sch.write(fast_A[i * T + j])
        # 2. Fused arg→fast load of B tile.
        for i in range(T):
            for j in range(T):
                for _sgn, rb, cb in ops_B:
                    sch.read(B[(rb + k_off + i) * n + cb + c + j])
                sch.write(fast_B[i * T + j])
        # 3. Tile MAC.
        for i in range(T):
            for j in range(T):
                for k in range(T):
                    if k == 0 and k_off == 0:
                        sch.mul(fast_A[i * T + k], fast_B[k * T + j],
                                fast_C[i * T + j])
                    else:
                        sch.mac(fast_C[i * T + j],
                                fast_A[i * T + k],
                                fast_B[k * T + j], tmp)
        # 4. Fan-out fast_C → signed C targets.
        for _sgn, rb, cb, is_first in ops_C:
            for i in range(T):
                for j in range(T):
                    if is_first:
                        sch.assign(fast_C[i * T + j],
                                   C[(rb + r + i) * n + cb + c + j])
                    else:
                        # C += fast_C: 2 reads, 1 write.
                        sch.read(fast_C[i * T + j])
                        sch.read(C[(rb + r + i) * n + cb + c + j])
                        sch.write(C[(rb + r + i) * n + cb + c + j])

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
    return sch.finalize()


# ---------------------------------------------------------------------------
# spmv (CSR) — row-wise MAC with arg-stack vals + x, scratch acc.
# ---------------------------------------------------------------------------

def _manual_spmv_dsl(n: int, row_ptr, col_ind) -> int:
    sch = Sched()
    vals = sch.arg_buffer(len(col_ind))
    x = sch.arg_buffer(n)
    acc = sch.scalar(); tmp = sch.scalar()
    y = sch.output_buffer(n)
    for i in range(n):
        start = row_ptr[i]; end = row_ptr[i + 1]
        if start == end:
            sch.assign(x[0], y[i])
            continue
        sch.mul(vals[start], x[col_ind[start]], acc)
        for k in range(start + 1, end):
            sch.mac(acc, vals[k], x[col_ind[k]], tmp)
        sch.assign(acc, y[i])
    return sch.finalize()


def manual_spmv_csr_banded_dsl(n: int, bandwidth: int = 3) -> int:
    row_ptr, col_ind = [0], []
    total = 0
    for i in range(n):
        for j in range(max(0, i - bandwidth), min(n, i + bandwidth + 1)):
            col_ind.append(j); total += 1
        row_ptr.append(total)
    return _manual_spmv_dsl(n, row_ptr, col_ind)


def manual_spmv_csr_random_dsl(n: int, nnz_per_row: int = 7,
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
    return _manual_spmv_dsl(n, row_ptr, col_ind)


# ===========================================================================
# Schur-family ports (lazy_matrix). All share the pattern:
#   A = sch.lazy_matrix(n*n)   # arg base reserved
#   ... allocate scratchpads ...
#   sch.lazy_output_buffer(A)  # scratch base (highest addresses), is_output
# Each A[i*n+j] reads from arg until first written, then from scratch.
# ===========================================================================

def manual_lu_no_pivot_dsl(n: int) -> int:
    sch = Sched()
    A = sch.lazy_matrix(n * n)
    tmp = sch.scalar()
    c_A = sch.scalar()
    c_C = sch.buffer(n)
    sch.lazy_output_buffer(A)

    for k in range(n):
        # Pivot: cache A[k][k] into c_A.
        sch.assign(A[k * n + k], c_A)
        # Divide column k: A[i][k] /= c_A
        for i in range(k + 1, n):
            sch.read(A[i * n + k]); sch.read(c_A); sch.write(A[i * n + k])
        # Row cache: A[k][k+1..n-1] -> c_C
        for j in range(k + 1, n):
            sch.assign(A[k * n + j], c_C[j - k - 1])
        # Schur update.
        for i in range(k + 1, n):
            sch.assign(A[i * n + k], c_A)
            for j in range(k + 1, n):
                sch.mul(c_A, c_C[j - k - 1], tmp)
                sch.sub(A[i * n + j], tmp, A[i * n + j])
    return sch.finalize()


def manual_cholesky_dsl(n: int) -> int:
    sch = Sched()
    A = sch.lazy_matrix(n * n)
    tmp = sch.scalar()
    c_A = sch.scalar()
    c_C = sch.buffer(n)
    sch.lazy_output_buffer(A)

    for k in range(n):
        # Pivot sqrt: A[k][k] = sqrt(A[k][k])  (1 read + 1 write).
        sch.read(A[k * n + k]); sch.write(A[k * n + k])
        sch.assign(A[k * n + k], c_A)
        # Divide column k.
        for i in range(k + 1, n):
            sch.read(A[i * n + k]); sch.read(c_A); sch.write(A[i * n + k])
        # Cache column k below diagonal.
        for i in range(k + 1, n):
            sch.assign(A[i * n + k], c_C[i - k - 1])
        # Schur update (lower triangle).
        for j in range(k + 1, n):
            sch.assign(c_C[j - k - 1], c_A)
            for i in range(j, n):
                sch.mul(c_C[i - k - 1], c_A, tmp)
                sch.sub(A[i * n + j], tmp, A[i * n + j])
    return sch.finalize()


def manual_floyd_warshall_naive_dsl(V: int) -> int:
    sch = Sched()
    D = sch.lazy_matrix(V * V)
    tmp = sch.scalar()
    c_A = sch.scalar()
    c_C = sch.buffer(V)
    sch.lazy_output_buffer(D)

    for k in range(V):
        for j in range(V):
            sch.assign(D[k * V + j], c_C[j])
        for i in range(V):
            sch.assign(D[i * V + k], c_A)
            for j in range(V):
                # Matches hand-rolled: 4 reads + 1 write (D, c_A, c_C, tmp).
                sch.read(D[i * V + j])
                sch.read(c_A)
                sch.read(c_C[j])
                sch.read(tmp)
                sch.write(D[i * V + j])
    return sch.finalize()


def manual_lu_partial_pivot_dsl(n: int) -> int:
    sch = Sched()
    A = sch.lazy_matrix(n * n)
    tmp = sch.scalar()
    c_A = sch.scalar()
    c_C = sch.buffer(n)
    sch.lazy_output_buffer(A)

    for k in range(n):
        # Pivot selection — scan column k.
        for i in range(k, n):
            sch.read(A[i * n + k])
            sch.read(A[k * n + k])
        # Row swap: rows k and p across cols [k, n).
        p = k + 1 if k + 1 < n else k
        for j in range(k, n):
            sch.read(A[k * n + j]); sch.read(A[p * n + j])
            sch.write(A[k * n + j]); sch.write(A[p * n + j])
        sch.assign(A[k * n + k], c_A)
        for i in range(k + 1, n):
            sch.read(A[i * n + k]); sch.read(c_A); sch.write(A[i * n + k])
        for j in range(k + 1, n):
            sch.assign(A[k * n + j], c_C[j - k - 1])
        for i in range(k + 1, n):
            sch.assign(A[i * n + k], c_A)
            for j in range(k + 1, n):
                # Hand-rolled: 3 reads + 1 write (no tmp).
                sch.read(A[i * n + j])
                sch.read(c_A)
                sch.read(c_C[j - k - 1])
                sch.write(A[i * n + j])
    return sch.finalize()


def manual_householder_qr_dsl(m: int, n: int) -> int:
    sch = Sched()
    A_in = sch.arg_buffer(m * n)
    tmp = sch.scalar()
    c_A = sch.scalar()
    c_V = sch.buffer(m)
    A = sch.output_buffer(m * n)
    for i in range(m * n):
        sch.assign(A_in[i], A[i])

    for k in range(min(m, n)):
        # Reflector norm: reads column k, writes it back.
        sch.read(A[k * n + k])
        for i in range(k + 1, m):
            sch.read(A[i * n + k])
        sch.write(A[k * n + k])
        for i in range(k + 1, m):
            sch.write(A[i * n + k])
        # Cache reflector column into c_V.
        sch.assign(A[k * n + k], c_V[0])
        for i in range(k + 1, m):
            sch.assign(A[i * n + k], c_V[i - k])
        # Apply reflector to each trailing column j.
        for j in range(k + 1, n):
            # Dot product: c_A = Σ c_V[.] * A[.][j]
            sch.read(c_V[0]); sch.read(A[k * n + j]); sch.write(c_A)
            for i in range(k + 1, m):
                sch.read(c_V[i - k]); sch.read(A[i * n + j])
                sch.read(c_A); sch.write(c_A)
            # Rank-1 update.
            sch.read(c_A); sch.read(c_V[0]); sch.read(A[k * n + j])
            sch.write(A[k * n + j])
            for i in range(k + 1, m):
                sch.read(c_A); sch.read(c_V[i - k])
                sch.read(A[i * n + j]); sch.write(A[i * n + j])
    return sch.finalize()


def manual_blocked_lu_dsl(n: int, NB: int = 8) -> int:
    sch = Sched()
    A = sch.lazy_matrix(n * n)
    tmp = sch.scalar()
    c_A = sch.scalar()
    c_C = sch.buffer(NB)
    c_B = sch.buffer(NB * NB)
    sch.lazy_output_buffer(A)

    for kb in range(0, n, NB):
        ke = min(kb + NB, n)
        sz = ke - kb

        # (a) Diagonal block: load, factor locally in c_B, flush.
        for i in range(kb, ke):
            for j in range(kb, ke):
                sch.assign(A[i * n + j], c_B[(i - kb) * NB + (j - kb)])
        for k in range(sz):
            pivot_idx = k * NB + k
            sch.assign(c_B[pivot_idx], c_A)
            for i in range(k + 1, sz):
                sch.read(c_B[i * NB + k]); sch.read(c_A)
                sch.write(c_B[i * NB + k])
            for i in range(k + 1, sz):
                sch.assign(c_B[i * NB + k], c_A)
                for j in range(k + 1, sz):
                    # 4 reads + 1 write (standard MAC).
                    sch.read(c_B[i * NB + j])
                    sch.read(c_A)
                    sch.read(c_B[k * NB + j])
                    sch.write(c_B[i * NB + j])
        for i in range(kb, ke):
            for j in range(kb, ke):
                sch.assign(c_B[(i - kb) * NB + (j - kb)], A[i * n + j])

        # (b) Panel update A[ke:n, kb:ke] via c_C row.
        for ib in range(ke, n, NB):
            ie = min(ib + NB, n)
            for i in range(ib, ie):
                for j in range(kb, ke):
                    sch.assign(A[i * n + j], c_C[j - kb])
                for k in range(sz):
                    sch.read(c_C[k]); sch.read(c_B[k * NB + k])
                    sch.write(c_C[k])
                    sch.assign(c_C[k], c_A)
                    for j in range(k + 1, sz):
                        sch.read(c_C[j])
                        sch.read(c_A)
                        sch.read(c_B[k * NB + j])
                        sch.write(c_C[j])
                for j in range(kb, ke):
                    sch.read(c_C[j - kb])
                    sch.write(A[i * n + j])

        # (c) Row-strip update A[kb:ke, ke:n] via c_B block.
        for jb in range(ke, n, NB):
            je = min(jb + NB, n)
            sz_j = je - jb
            for k in range(kb, ke):
                for j in range(jb, je):
                    sch.assign(A[k * n + j], c_B[(k - kb) * NB + (j - jb)])
            for k in range(sz):
                for i in range(k + 1, sz):
                    sch.assign(A[(kb + i) * n + (kb + k)], c_A)
                    for j in range(sz_j):
                        sch.read(c_B[i * NB + j])
                        sch.read(c_A)
                        sch.read(c_B[k * NB + j])
                        sch.write(c_B[i * NB + j])
            for k in range(kb, ke):
                for j in range(jb, je):
                    sch.read(c_B[(k - kb) * NB + (j - jb)])
                    sch.write(A[k * n + j])

        # (d) Trailing GEMM: A[ke:n, jb:je] for each (ib, jb) pair.
        for jb in range(ke, n, NB):
            je = min(jb + NB, n)
            sz_j = je - jb
            # Load row-strip block into c_B.
            for k in range(kb, ke):
                for j in range(jb, je):
                    sch.assign(A[k * n + j], c_B[(k - kb) * NB + (j - jb)])
            for ib in range(ke, n, NB):
                ie = min(ib + NB, n)
                for i in range(ib, ie):
                    for j in range(jb, je):
                        sch.assign(A[i * n + j], c_C[j - jb])
                    for k in range(sz):
                        sch.assign(A[i * n + (kb + k)], c_A)
                        for j in range(sz_j):
                            sch.read(c_C[j])
                            sch.read(c_A)
                            sch.read(c_B[k * NB + j])
                            sch.write(c_C[j])
                    for j in range(jb, je):
                        sch.read(c_C[j - jb])
                        sch.write(A[i * n + j])
    return sch.finalize()


def manual_floyd_warshall_recursive_dsl(V: int) -> int:
    SZ = 2

    # Dry run to compute per-block miss counts (matches hand-rolled).
    miss_counts: dict = {}
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

    def D_idx(r, c):
        b = block_mapping[((r // SZ) * SZ, (c // SZ) * SZ)]
        return b * (SZ * SZ) + (r % SZ) * SZ + (c % SZ)

    sch = Sched()
    M = sch.arg_buffer(V * V)
    tmp = sch.scalar()
    cache_T = sch.buffer(SZ * SZ)
    cache_D = sch.buffer(SZ * SZ)
    D = sch.output_buffer(V * V)

    for i in range(V):
        for j in range(V):
            sch.assign(M[i * V + j], D[D_idx(i, j)])

    tag_T = [None]
    tag_D = [None]
    dirty_T = [False]

    def load_T(r0, c0):
        if tag_T[0] == (r0, c0):
            return
        if tag_T[0] is not None and dirty_T[0]:
            pr, pc = tag_T[0]
            for i in range(SZ):
                for j in range(SZ):
                    sch.assign(cache_T[i * SZ + j], D[D_idx(pr + i, pc + j)])
        tag_T[0] = (r0, c0)
        dirty_T[0] = False
        for i in range(SZ):
            for j in range(SZ):
                sch.assign(D[D_idx(r0 + i, c0 + j)], cache_T[i * SZ + j])

    def load_D(r0):
        if tag_D[0] == r0:
            return
        tag_D[0] = r0
        for i in range(SZ):
            for j in range(SZ):
                sch.assign(D[D_idx(r0 + i, r0 + j)], cache_D[i * SZ + j])

    def do_block(r0, c0):
        load_T(r0, c0)
        if r0 != c0:
            load_D(r0)
        for k in range(SZ):
            for i in range(SZ):
                for j in range(SZ):
                    sch.read(cache_T[i * SZ + j])
                    if r0 == c0:
                        sch.read(cache_T[i * SZ + k])
                    else:
                        sch.read(cache_D[i * SZ + k])
                    sch.read(cache_T[k * SZ + j])
                    sch.write(cache_T[i * SZ + j])
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
        pr, pc = tag_T[0]
        for i in range(SZ):
            for j in range(SZ):
                sch.assign(cache_T[i * SZ + j], D[D_idx(pr + i, pc + j)])
    return sch.finalize()


# ---------------------------------------------------------------------------
# fft_conv — FFT-based convolution: forward FFT on X and Y, pointwise
# multiply, inverse FFT (here just another forward, cost parity only).
# ---------------------------------------------------------------------------

def manual_fft_conv_dsl(N: int) -> int:
    sch = Sched()
    X_in = sch.arg_buffer(N); Y_in = sch.arg_buffer(N)
    tmp = sch.scalar()
    X = sch.buffer(N); Y = sch.buffer(N); Z = sch.output_buffer(N)
    for i in range(N):
        sch.assign(X_in[i], X[i])
        sch.assign(Y_in[i], Y[i])

    def fft_in_place(base):
        j = 0
        for i in range(1, N):
            bit = N >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                sch.swap(base[i], base[j], tmp)
        m = 1
        while m < N:
            for k in range(0, N, m * 2):
                for jj in range(m):
                    sch.butterfly(base[k + jj], base[k + jj + m], tmp)
            m *= 2

    fft_in_place(X)
    fft_in_place(Y)
    for k in range(N):
        # Z[k] = X[k] * Y[k] — single binary op, 2 reads + free write.
        sch.mul(X[k], Y[k], Z[k])
    fft_in_place(Z)
    return sch.finalize()


# ---------------------------------------------------------------------------
# matrix_powers — naive vs. communication-avoiding. Both are just
# repeated matvecs with A on arg + a scratch ping-pong buffer.
# ---------------------------------------------------------------------------

def manual_matrix_powers_naive_dsl(n: int, s: int = 4) -> int:
    sch = Sched()
    A = sch.arg_buffer(n * n); x0 = sch.arg_buffer(n)
    cur = sch.buffer(n); nxt = sch.buffer(n)
    acc = sch.scalar(); tmp = sch.scalar()
    for i in range(n):
        sch.assign(x0[i], cur[i])
    for _step in range(s):
        for i in range(n):
            sch.mul(A[i * n + 0], cur[0], acc)
            for j in range(1, n):
                sch.mac(acc, A[i * n + j], cur[j], tmp)
            sch.assign(acc, nxt[i])
        for i in range(n):
            sch.assign(nxt[i], cur[i])
    # The hand-rolled calls set_output_range only on the final cur.
    sch._a.set_output_range(cur[0].addr, cur[0].addr + n)
    sch._output_cells = cur
    return sch.finalize()


def manual_matrix_powers_ca_dsl(n: int, s: int = 4, block: int = 4) -> int:
    sch = Sched()
    A = sch.arg_buffer(n * n); x0 = sch.arg_buffer(n)
    cur = sch.buffer(n); nxt = sch.buffer(n)
    acc = sch.scalar(); tmp = sch.scalar()
    for i in range(n):
        sch.assign(x0[i], cur[i])
    for _step in range(s):
        for bi in range(0, n, block):
            for i in range(bi, min(bi + block, n)):
                sch.mul(A[i * n + 0], cur[0], acc)
                for j in range(1, n):
                    sch.mac(acc, A[i * n + j], cur[j], tmp)
                sch.assign(acc, nxt[i])
        for i in range(n):
            sch.assign(nxt[i], cur[i])
    sch._a.set_output_range(cur[0].addr, cur[0].addr + n)
    sch._output_cells = cur
    return sch.finalize()


# ---------------------------------------------------------------------------
# regular_convolution — multi-channel CNN layer with rolling K-row img cache.
# ---------------------------------------------------------------------------

def manual_regular_convolution_dsl(H: int, W: int, K: int,
                                   Cin: int, Cout: int) -> int:
    sch = Sched()
    Wk = sch.arg_buffer(K * K * Cin * Cout)
    img = sch.arg_buffer(H * W * Cin)
    tmp = sch.scalar(); s = sch.scalar()
    row_stride = W * Cin
    buf = sch.buffer(K * row_stride)
    out_h = H - K + 1
    out_w = W - K + 1
    O = sch.output_buffer(out_h * out_w * Cout)

    def load_row(r):
        slot_off = (r % K) * row_stride
        base_arg = r * row_stride
        for x in range(row_stride):
            sch.assign(img[base_arg + x], buf[slot_off + x])

    for r in range(K):
        load_row(r)

    for i in range(out_h):
        if i > 0:
            load_row(i + K - 1)
        for j in range(out_w):
            for co in range(Cout):
                first = True
                for ki in range(K):
                    slot_off = ((i + ki) % K) * row_stride
                    for kj in range(K):
                        col_off = (j + kj) * Cin
                        wk_off = ((ki * K + kj) * Cin) * Cout + co
                        for ci in range(Cin):
                            if first:
                                sch.mul(buf[slot_off + col_off + ci],
                                        Wk[wk_off + ci * Cout], s)
                                first = False
                            else:
                                sch.mac(s, buf[slot_off + col_off + ci],
                                        Wk[wk_off + ci * Cout], tmp)
                sch.assign(s, O[(i * out_w + j) * Cout + co])
    return sch.finalize()


# ---------------------------------------------------------------------------
# stencil_naive — rolling 3-row buffer, 5-point Jacobi sweep.
# ---------------------------------------------------------------------------

def manual_stencil_naive_dsl(n: int) -> int:
    sch = Sched()
    A = sch.arg_buffer(n * n)
    r0 = sch.buffer(n); r1 = sch.buffer(n); r2 = sch.buffer(n)
    rows = [r0, r1, r2]
    B = sch.output_buffer(n * n)

    # Preload rows 0, 1, 2.
    for row in range(min(3, n)):
        slot = rows[row % 3]
        for j in range(n):
            sch.assign(A[row * n + j], slot[j])

    for i in range(1, n - 1):
        up = rows[(i - 1) % 3]
        cur = rows[i % 3]
        down = rows[(i + 1) % 3]
        for j in range(1, n - 1):
            # 5 stencil reads, 1 write (fused multi-operand op).
            sch.read(cur[j])
            sch.read(up[j])
            sch.read(down[j])
            sch.read(cur[j - 1])
            sch.read(cur[j + 1])
            sch.write(B[i * n + j])
        # Stream next A-row into the stale slot.
        if i + 2 < n:
            replace = rows[(i - 1) % 3]
            for j in range(n):
                sch.assign(A[(i + 2) * n + j], replace[j])
    return sch.finalize()


# ---------------------------------------------------------------------------
# stencil_recursive — tile-recursive 5-point Jacobi with lazy row cache
# and row-band reordering. Same cost as stencil_naive when leaves fit
# monotonic row streaming.
# ---------------------------------------------------------------------------

def manual_stencil_recursive_dsl(n: int, leaf: int = 8) -> int:
    sch = Sched()
    # Rolling 3-row cache at lowest scratch addrs (matches hand-rolled layout).
    row_slots = [sch.buffer(n) for _ in range(3)]
    A = sch.arg_buffer(n * n)
    B = sch.output_buffer(n * n)

    current_row_in_slot = [-1, -1, -1]

    def ensure_row_loaded(row: int):
        slot_idx = row % 3
        slot = row_slots[slot_idx]
        if current_row_in_slot[slot_idx] != row:
            for j in range(n):
                sch.assign(A[row * n + j], slot[j])
            current_row_in_slot[slot_idx] = row
        return slot

    leaves: list = []

    def collect(r0, c0, sz):
        if sz <= leaf:
            leaves.append((r0, c0, sz))
            return
        h = sz // 2
        collect(r0,     c0,     h)
        collect(r0,     c0 + h, h)
        collect(r0 + h, c0,     h)
        collect(r0 + h, c0 + h, h)

    collect(0, 0, n)

    from collections import defaultdict
    by_r0: dict = defaultdict(list)
    for r0, c0, sz in leaves:
        by_r0[r0].append((r0, c0, sz))
    for r0 in by_r0:
        by_r0[r0].sort(key=lambda t: t[1])

    for r0 in sorted(by_r0.keys()):
        band = by_r0[r0]
        sz0 = band[0][2]
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
                    sch.read(cur[j])
                    sch.read(up[j])
                    sch.read(down[j])
                    sch.read(cur[j - 1])
                    sch.read(cur[j + 1])
                    sch.write(B[i * n + j])
    return sch.finalize()


def manual_heapsort_dsl(N: int) -> int:
    sch = Sched()
    arr_in = sch.arg_buffer(N)
    arr = sch.output_buffer(N)
    for i in range(N):
        sch.assign(arr_in[i], arr[i])

    def sift_down(j: int, heap_size: int) -> None:
        while 2 * j + 1 < heap_size:
            child = 2 * j + 1
            if child + 1 < heap_size:
                # Compare arr[child] vs arr[child+1]; pick larger (in-place).
                sch.read(arr[child])
                sch.read(arr[child + 1])
                sch.write(arr[child])
            # Swap-or-keep arr[j] with arr[child].
            sch.read(arr[j])
            sch.read(arr[child])
            sch.write(arr[j])
            sch.write(arr[child])
            j = child

    for i in range(N // 2 - 1, -1, -1):
        sift_down(i, N)
    for k in range(N - 1, 0, -1):
        sch.read(arr[k])
        sch.read(arr[0])
        sch.write(arr[k])
        sch.write(arr[0])
        sift_down(0, k)
    return sch.finalize()
