"""
4x4 matmul strategies under manual allocation + flat Manhattan cost.

Scope: only n^3 = 64 scalar products, (*, +)-semiring. Not exhaustive
— a curated catalog of loop orders, caching choices, and storage
policies that cover the interesting tradeoffs. Each strategy is
simulated end-to-end with an explicit Allocator (bump pointer + push/
pop), so cell lifetimes are honored exactly. Cost = sum over reads of
ceil(sqrt(addr)); writes are free.

Arg layout: A @ 1..16, B @ 17..32, both row-major. Every n^3 schedule
reads each A cell and each B cell 4 times, so the "pure arg baseline"
(if no hoisting happened) is

    4 * sum(ceil(sqrt(a)) for a in 1..32)
      = 4 * (1*1 + 2*3 + 3*5 + 4*7 + 5*9 + 6*7)
      = 4 * 137 = 548.

Strategies that hoist rows/columns into scratch reduce this at the
cost of extra scratch reads and scratch address pressure.

Output: each strategy returns a dict {name, cost, peak_scratch,
peak_arg, reads}. `reads` is the total number of priced reads
(useful for intuition — cost = sum of ceil(sqrt(addr)) across these).
"""

from tracer import Allocator


N = 4


def _setup():
    a = Allocator()
    A = a.alloc_arg(N * N)
    B = a.alloc_arg(N * N)
    return a, A, B


def _A(A, i, k):
    return A + i * N + k


def _B(B, k, j):
    return B + k * N + j


def _C(C, i, j):
    return C + i * N + j


def _epilogue(a, C):
    for idx in range(N * N):
        a.read(C + idx)


def _pack(name, alloc):
    return {
        'name': name,
        'cost': alloc.cost,
        'peak_scratch': alloc.peak,
        'peak_arg': alloc.arg_peak,
        'reads': len(alloc.log),
    }


# ---------------------------------------------------------------------------
# Naive (no caching) — pure in-place C accumulator
# ---------------------------------------------------------------------------

def naive_ijk_direct():
    """i-j-k loop. First MUL of each (i,j) writes directly to C[i][j];
    remaining three products go through a shared scratch `tmp`,
    accumulated into C in place. Minimal footprint: 1 tmp + 16 C = 17."""
    a, A, B = _setup()
    tmp = a.alloc(1)
    C = a.alloc(N * N)
    for i in range(N):
        for j in range(N):
            a.read_arg(_A(A, i, 0)); a.read_arg(_B(B, 0, j))
            a.write(_C(C, i, j))
            for k in range(1, N):
                a.read_arg(_A(A, i, k)); a.read_arg(_B(B, k, j))
                a.write(tmp)
                a.read(_C(C, i, j)); a.read(tmp)
                a.write(_C(C, i, j))
    _epilogue(a, C)
    return _pack('naive_ijk_direct', a)


def naive_ijk_always_acc():
    """i-j-k, every product goes through tmp and accumulates into a
    zero-initialized C. Same cell set as direct, but every pair pays
    one extra tmp-assign — shows the cost of skipping the MUL1-to-C
    shortcut."""
    a, A, B = _setup()
    tmp = a.alloc(1)
    C = a.alloc(N * N)
    for idx in range(N * N):
        a.write(C + idx)   # conceptual zero-init (free)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                a.read_arg(_A(A, i, k)); a.read_arg(_B(B, k, j))
                a.write(tmp)
                a.read(_C(C, i, j)); a.read(tmp)
                a.write(_C(C, i, j))
    _epilogue(a, C)
    return _pack('naive_ijk_always_acc', a)


def naive_jik_direct():
    """j-i-k loop — outer i and j swapped vs ijk. Under flat-addr cost
    the reorder is invisible; this strategy is included to verify the
    invariant empirically."""
    a, A, B = _setup()
    tmp = a.alloc(1)
    C = a.alloc(N * N)
    for j in range(N):
        for i in range(N):
            a.read_arg(_A(A, i, 0)); a.read_arg(_B(B, 0, j))
            a.write(_C(C, i, j))
            for k in range(1, N):
                a.read_arg(_A(A, i, k)); a.read_arg(_B(B, k, j))
                a.write(tmp)
                a.read(_C(C, i, j)); a.read(tmp)
                a.write(_C(C, i, j))
    _epilogue(a, C)
    return _pack('naive_jik_direct', a)


def naive_kij_rank1():
    """k-outer rank-1 update. For each k, sweep all (i, j) and do
    C[i][j] += A[i][k] * B[k][j]. The k=0 pass writes C directly
    (initializes); the remaining three passes accumulate via tmp."""
    a, A, B = _setup()
    tmp = a.alloc(1)
    C = a.alloc(N * N)
    for k in range(N):
        for i in range(N):
            for j in range(N):
                a.read_arg(_A(A, i, k)); a.read_arg(_B(B, k, j))
                if k == 0:
                    a.write(_C(C, i, j))
                else:
                    a.write(tmp)
                    a.read(_C(C, i, j)); a.read(tmp)
                    a.write(_C(C, i, j))
    _epilogue(a, C)
    return _pack('naive_kij_rank1', a)


# ---------------------------------------------------------------------------
# A-row / B-col / both hoisted
# ---------------------------------------------------------------------------

def naive_a_row_cached():
    """Hoist A[i][*] into a hot 4-cell scratch buffer once per outer i
    (grid's manual_naive_matmul_cached shape, minus the scalar s)."""
    a, A, B = _setup()
    tmp = a.alloc(1)
    a_row = a.alloc(N)
    C = a.alloc(N * N)
    for i in range(N):
        for k in range(N):
            a.read_arg(_A(A, i, k)); a.write(a_row + k)
        for j in range(N):
            a.read(a_row + 0); a.read_arg(_B(B, 0, j))
            a.write(_C(C, i, j))
            for k in range(1, N):
                a.read(a_row + k); a.read_arg(_B(B, k, j))
                a.write(tmp)
                a.read(_C(C, i, j)); a.read(tmp)
                a.write(_C(C, i, j))
    _epilogue(a, C)
    return _pack('naive_a_row_cached', a)


def naive_a_row_scalar_acc():
    """A-row cached plus scalar accumulator s. Compute the full dot
    product for one (i, j) in s, flush to C once. One MAC per k (no
    direct-to-C shortcut) so s is read every k."""
    a, A, B = _setup()
    tmp = a.alloc(1)
    s = a.alloc(1)
    a_row = a.alloc(N)
    C = a.alloc(N * N)
    for i in range(N):
        for k in range(N):
            a.read_arg(_A(A, i, k)); a.write(a_row + k)
        for j in range(N):
            # First MAC writes s directly.
            a.read(a_row + 0); a.read_arg(_B(B, 0, j))
            a.write(s)
            for k in range(1, N):
                a.read(a_row + k); a.read_arg(_B(B, k, j))
                a.write(tmp)
                a.read(s); a.read(tmp)
                a.write(s)
            a.read(s); a.write(_C(C, i, j))
    _epilogue(a, C)
    return _pack('naive_a_row_scalar_acc', a)


def naive_b_col_cached():
    """Hoist B[*][j] into a 4-cell scratch buffer per outer j."""
    a, A, B = _setup()
    tmp = a.alloc(1)
    b_col = a.alloc(N)
    C = a.alloc(N * N)
    for j in range(N):
        for k in range(N):
            a.read_arg(_B(B, k, j)); a.write(b_col + k)
        for i in range(N):
            a.read_arg(_A(A, i, 0)); a.read(b_col + 0)
            a.write(_C(C, i, j))
            for k in range(1, N):
                a.read_arg(_A(A, i, k)); a.read(b_col + k)
                a.write(tmp)
                a.read(_C(C, i, j)); a.read(tmp)
                a.write(_C(C, i, j))
    _epilogue(a, C)
    return _pack('naive_b_col_cached', a)


def outer_b_row_a_scalar():
    """k-i-j outer-product style with B[k][*] hoisted and a scalar
    register for A[i][k]. For each k load B[k][*] (4 cells), then for
    each i load A[i][k] into a scalar and fan out across j, writing or
    accumulating into C[i][j]."""
    a, A, B = _setup()
    tmp = a.alloc(1)
    a_scalar = a.alloc(1)
    b_row = a.alloc(N)
    C = a.alloc(N * N)
    for k in range(N):
        for j in range(N):
            a.read_arg(_B(B, k, j)); a.write(b_row + j)
        for i in range(N):
            a.read_arg(_A(A, i, k)); a.write(a_scalar)
            for j in range(N):
                a.read(a_scalar); a.read(b_row + j)
                if k == 0:
                    a.write(_C(C, i, j))
                else:
                    a.write(tmp)
                    a.read(_C(C, i, j)); a.read(tmp)
                    a.write(_C(C, i, j))
    _epilogue(a, C)
    return _pack('outer_b_row_a_scalar', a)


# ---------------------------------------------------------------------------
# Fully stored products
# ---------------------------------------------------------------------------

def batched_per_pair():
    """For each (i, j): compute its 4 products into a 4-cell scratch
    buffer, then fold them into C[i][j] pairwise. The 4-cell buffer is
    reused across pairs (sequential). No per-product C reads mid-loop."""
    a, A, B = _setup()
    P = a.alloc(N)                 # 4 product cells, reused per pair
    C = a.alloc(N * N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                a.read_arg(_A(A, i, k)); a.read_arg(_B(B, k, j))
                a.write(P + k)
            a.read(P + 0); a.read(P + 1); a.write(_C(C, i, j))
            a.read(_C(C, i, j)); a.read(P + 2); a.write(_C(C, i, j))
            a.read(_C(C, i, j)); a.read(P + 3); a.write(_C(C, i, j))
    _epilogue(a, C)
    return _pack('batched_per_pair', a)


def batched_all_64():
    """All 64 products live at once, then 16 sums. Maximal footprint
    (64 P + 16 C = 80 scratch cells) — this is the upper bound on how
    much storing products can cost."""
    a, A, B = _setup()
    P = a.alloc(N * N * N)         # 64 product cells
    C = a.alloc(N * N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                a.read_arg(_A(A, i, k)); a.read_arg(_B(B, k, j))
                a.write(P + i * N * N + j * N + k)
    for i in range(N):
        for j in range(N):
            base = P + i * N * N + j * N
            a.read(base + 0); a.read(base + 1); a.write(_C(C, i, j))
            a.read(_C(C, i, j)); a.read(base + 2); a.write(_C(C, i, j))
            a.read(_C(C, i, j)); a.read(base + 3); a.write(_C(C, i, j))
    _epilogue(a, C)
    return _pack('batched_all_64', a)


# ---------------------------------------------------------------------------
# 2x2 block recursion
# ---------------------------------------------------------------------------

def block_2x2_direct():
    """Treat the 4x4 as a 2x2 block-multiply of 2x2 blocks.

    For each of the four output blocks C_IJ:
      C_IJ = A_I0 * B_0J + A_I1 * B_1J
    The first 2x2 block multiplication writes directly into the C_IJ
    block (overwrite); the second goes into a 4-cell tmp_block and is
    then added elementwise into C_IJ. Each inner 2x2 matmul is the
    2x2-exhaustive winner: direct-mode in-place on its 4 output cells
    with a single shared inner_tmp. Footprint: 1 + 4 + 16 = 21 scratch
    cells."""
    a, A, B = _setup()
    inner_tmp = a.alloc(1)
    tmp_block = a.alloc(N)        # 2x2 scratch block (4 cells)
    C = a.alloc(N * N)

    def inner_mul_to(out_addrs, ai0, ak0, bk0, bj0):
        for aa in range(2):
            for bb in range(2):
                dst = out_addrs[aa][bb]
                a.read_arg(_A(A, ai0 + aa, ak0 + 0))
                a.read_arg(_B(B, bk0 + 0, bj0 + bb))
                a.write(dst)
                a.read_arg(_A(A, ai0 + aa, ak0 + 1))
                a.read_arg(_B(B, bk0 + 1, bj0 + bb))
                a.write(inner_tmp)
                a.read(dst); a.read(inner_tmp); a.write(dst)

    def block_mul_to_C(ci0, cj0, ai0, ak0, bk0, bj0):
        out = [[_C(C, ci0 + aa, cj0 + bb) for bb in range(2)]
               for aa in range(2)]
        inner_mul_to(out, ai0, ak0, bk0, bj0)

    def block_accum_into_C(ci0, cj0, ai0, ak0, bk0, bj0):
        out = [[tmp_block + aa * 2 + bb for bb in range(2)]
               for aa in range(2)]
        inner_mul_to(out, ai0, ak0, bk0, bj0)
        for aa in range(2):
            for bb in range(2):
                a.read(_C(C, ci0 + aa, cj0 + bb))
                a.read(tmp_block + aa * 2 + bb)
                a.write(_C(C, ci0 + aa, cj0 + bb))

    # Four output blocks. First product of each block writes to C;
    # second product goes through tmp_block and is added in.
    for bi in (0, 2):
        for bj in (0, 2):
            block_mul_to_C(bi, bj, bi, 0, 0, bj)
            block_accum_into_C(bi, bj, bi, 2, 2, bj)

    _epilogue(a, C)
    return _pack('block_2x2_direct', a)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES = [
    naive_ijk_direct,
    naive_ijk_always_acc,
    naive_jik_direct,
    naive_kij_rank1,
    naive_a_row_cached,
    naive_a_row_scalar_acc,
    naive_b_col_cached,
    outer_b_row_a_scalar,
    batched_per_pair,
    batched_all_64,
    block_2x2_direct,
]
