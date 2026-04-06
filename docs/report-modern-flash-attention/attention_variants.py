#!/usr/bin/env python3
"""
Pure-Python attention implementations for ByteDMD tracing.

Four variants:
  1. naive_attention       — materializes full N×N score matrix
  2. flash_attention_v1    — basic tiling with online softmax (Dao et al. 2022)
  3. flash_attention_v2    — swizzled K/V block ordering for inter-row locality
  4. flash_attention_v3    — double-tiled (Q-blocks × K/V-blocks) for K/V reuse

Auxiliary ops (exp, max, inv) are modeled as minimal tracked operations:
  max(a,b) → a+b  (binary: 2 reads)
  exp(x)   → -x   (unary: 1 read)
  inv(x)   → -x   (unary: 1 read)
"""


# --- Auxiliary ops with correct read counts ---

def _max2(a, b):
    """Binary op modeling max: reads a and b."""
    return a + b


def _exp(x):
    """Unary op modeling exp: reads x."""
    return -x


def _inv(x):
    """Unary op modeling 1/x: reads x."""
    return -x


# ---------------------------------------------------------------------------
# 1. Naive (standard) attention
# ---------------------------------------------------------------------------

def naive_attention(Q, K, V):
    """
    softmax(Q @ K^T) @ V — materializes the full N×N score matrix.
    Q, K, V: [N][d] list-of-lists.  Returns [N][d].
    """
    N, d = len(Q), len(Q[0])

    # Phase 1: S = Q @ K^T  (full N×N)
    S = [[None] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            s = Q[i][0] * K[j][0]
            for k in range(1, d):
                s = s + Q[i][k] * K[j][k]
            S[i][j] = s

    # Phase 2: row-wise softmax
    P = [[None] * N for _ in range(N)]
    for i in range(N):
        mx = S[i][0]
        for j in range(1, N):
            mx = _max2(mx, S[i][j])
        row_sum = None
        for j in range(N):
            P[i][j] = _exp(S[i][j] - mx)
            if row_sum is None:
                row_sum = P[i][j]
            else:
                row_sum = row_sum + P[i][j]
        inv_s = _inv(row_sum)
        for j in range(N):
            P[i][j] = P[i][j] * inv_s

    # Phase 3: O = P @ V
    O = [[None] * d for _ in range(N)]
    for i in range(N):
        for k in range(d):
            s = P[i][0] * V[0][k]
            for j in range(1, N):
                s = s + P[i][j] * V[j][k]
            O[i][k] = s
    return O


# ---------------------------------------------------------------------------
# 2. Flash Attention v1 — basic tiling with online softmax
# ---------------------------------------------------------------------------

def flash_attention_v1(Q, K, V, Bk=2):
    """
    Flash attention — processes K/V in blocks of size Bk.
    Uses online softmax to avoid materializing the full N×N matrix.
    """
    N, d = len(Q), len(Q[0])
    num_blocks = (N + Bk - 1) // Bk
    O = [[None] * d for _ in range(N)]

    for i in range(N):
        m_prev = l_prev = None
        o_acc = [None] * d

        for kb in range(num_blocks):
            ks = kb * Bk
            ke = min(ks + Bk, N)
            bsz = ke - ks

            # Block scores
            sb = [None] * bsz
            for j in range(bsz):
                acc = Q[i][0] * K[ks + j][0]
                for dd in range(1, d):
                    acc = acc + Q[i][dd] * K[ks + j][dd]
                sb[j] = acc

            # Block max
            mb = sb[0]
            for j in range(1, bsz):
                mb = _max2(mb, sb[j])

            # Exp + block sum
            pb = [None] * bsz
            lb = None
            for j in range(bsz):
                pb[j] = _exp(sb[j] - mb)
                if lb is None:
                    lb = pb[j]
                else:
                    lb = lb + pb[j]

            # Block output
            ob = [None] * d
            for dd in range(d):
                acc = pb[0] * V[ks][dd]
                for j in range(1, bsz):
                    acc = acc + pb[j] * V[ks + j][dd]
                ob[dd] = acc

            # Online softmax merge
            if m_prev is None:
                m_prev, l_prev = mb, lb
                for dd in range(d):
                    o_acc[dd] = ob[dd]
            else:
                m_new = _max2(m_prev, mb)
                alpha = _exp(m_prev - m_new)
                beta = _exp(mb - m_new)
                l_prev = alpha * l_prev + beta * lb
                for dd in range(d):
                    o_acc[dd] = alpha * o_acc[dd] + beta * ob[dd]
                m_prev = m_new

        # Normalize
        inv_l = _inv(l_prev)
        for dd in range(d):
            O[i][dd] = o_acc[dd] * inv_l
    return O


# ---------------------------------------------------------------------------
# 3. Flash Attention v2 — swizzled K/V block ordering
# ---------------------------------------------------------------------------

def flash_attention_v2(Q, K, V, Bk=2):
    """
    Flash attention with swizzled K/V block ordering.

    For even query rows, process K/V blocks 0,1,2,...
    For odd  query rows, process K/V blocks ...,2,1,0

    This "snake" pattern means the last K/V block used by row i is the
    first K/V block used by row i+1, keeping it hot at the top of the
    LRU stack.  The data movement pattern for Q·K^T and P·V is identical
    in arithmetic; only the order changes.
    """
    N, d = len(Q), len(Q[0])
    num_blocks = (N + Bk - 1) // Bk
    O = [[None] * d for _ in range(N)]

    for i in range(N):
        m_prev = l_prev = None
        o_acc = [None] * d

        # Swizzle: alternate block ordering per row
        if i % 2 == 0:
            block_order = range(num_blocks)
        else:
            block_order = range(num_blocks - 1, -1, -1)

        for kb in block_order:
            ks = kb * Bk
            ke = min(ks + Bk, N)
            bsz = ke - ks

            # Block scores
            sb = [None] * bsz
            for j in range(bsz):
                acc = Q[i][0] * K[ks + j][0]
                for dd in range(1, d):
                    acc = acc + Q[i][dd] * K[ks + j][dd]
                sb[j] = acc

            # Block max
            mb = sb[0]
            for j in range(1, bsz):
                mb = _max2(mb, sb[j])

            # Exp + block sum
            pb = [None] * bsz
            lb = None
            for j in range(bsz):
                pb[j] = _exp(sb[j] - mb)
                if lb is None:
                    lb = pb[j]
                else:
                    lb = lb + pb[j]

            # Block output
            ob = [None] * d
            for dd in range(d):
                acc = pb[0] * V[ks][dd]
                for j in range(1, bsz):
                    acc = acc + pb[j] * V[ks + j][dd]
                ob[dd] = acc

            # Online softmax merge
            if m_prev is None:
                m_prev, l_prev = mb, lb
                for dd in range(d):
                    o_acc[dd] = ob[dd]
            else:
                m_new = _max2(m_prev, mb)
                alpha = _exp(m_prev - m_new)
                beta = _exp(mb - m_new)
                l_prev = alpha * l_prev + beta * lb
                for dd in range(d):
                    o_acc[dd] = alpha * o_acc[dd] + beta * ob[dd]
                m_prev = m_new

        # Normalize
        inv_l = _inv(l_prev)
        for dd in range(d):
            O[i][dd] = o_acc[dd] * inv_l
    return O


# ---------------------------------------------------------------------------
# 4. Flash Attention v3 — double-tiled (Q-blocks × K/V-blocks)
# ---------------------------------------------------------------------------

def flash_attention_v3(Q, K, V, Bq=2, Bk=2):
    """
    Double-tiled flash attention: tile both Q and K/V dimensions.

    The outer loop iterates over Q-blocks of size Bq.  For each Q-block,
    all K/V blocks are processed.  This maximizes K/V reuse: the same K/V
    block is read once and used for Bq query rows before being evicted.

    Additionally uses snake ordering on K/V blocks for inter-Q-block locality.
    """
    N, d = len(Q), len(Q[0])
    num_kv_blocks = (N + Bk - 1) // Bk
    num_q_blocks = (N + Bq - 1) // Bq
    O = [[None] * d for _ in range(N)]

    # Per-query running state
    m_prev_all = [None] * N
    l_prev_all = [None] * N
    o_acc_all  = [[None] * d for _ in range(N)]

    for qb in range(num_q_blocks):
        qs = qb * Bq
        qe = min(qs + Bq, N)

        # Snake ordering: alternate K/V block direction per Q-block
        if qb % 2 == 0:
            kv_order = range(num_kv_blocks)
        else:
            kv_order = range(num_kv_blocks - 1, -1, -1)

        for kb in kv_order:
            ks = kb * Bk
            ke = min(ks + Bk, N)
            bsz = ke - ks

            # Process each query in this Q-block against this K/V block
            for i in range(qs, qe):
                # Block scores
                sb = [None] * bsz
                for j in range(bsz):
                    acc = Q[i][0] * K[ks + j][0]
                    for dd in range(1, d):
                        acc = acc + Q[i][dd] * K[ks + j][dd]
                    sb[j] = acc

                # Block max
                mb = sb[0]
                for j in range(1, bsz):
                    mb = _max2(mb, sb[j])

                # Exp + block sum
                pb = [None] * bsz
                lb = None
                for j in range(bsz):
                    pb[j] = _exp(sb[j] - mb)
                    if lb is None:
                        lb = pb[j]
                    else:
                        lb = lb + pb[j]

                # Block output
                ob = [None] * d
                for dd in range(d):
                    acc = pb[0] * V[ks][dd]
                    for j in range(1, bsz):
                        acc = acc + pb[j] * V[ks + j][dd]
                    ob[dd] = acc

                # Online softmax merge
                m_prev = m_prev_all[i]
                l_prev = l_prev_all[i]
                o_acc = o_acc_all[i]

                if m_prev is None:
                    m_prev_all[i] = mb
                    l_prev_all[i] = lb
                    for dd in range(d):
                        o_acc[dd] = ob[dd]
                else:
                    m_new = _max2(m_prev, mb)
                    alpha = _exp(m_prev - m_new)
                    beta = _exp(mb - m_new)
                    l_prev_all[i] = alpha * l_prev + beta * lb
                    for dd in range(d):
                        o_acc[dd] = alpha * o_acc[dd] + beta * ob[dd]
                    m_prev_all[i] = m_new

    # Normalize all rows
    for i in range(N):
        inv_l = _inv(l_prev_all[i])
        for dd in range(d):
            O[i][dd] = o_acc_all[i][dd] * inv_l
    return O


# ---------------------------------------------------------------------------
# Analytical FLOP counters
# ---------------------------------------------------------------------------

def naive_flops(N, d):
    """Exact FLOP count matching naive_attention implementation."""
    f = 0
    f += N * N * (2 * d - 1)                          # Q @ K^T
    f += N * ((N - 1) + N + N + (N - 1) + 1 + N)      # softmax per row
    f += N * d * (2 * N - 1)                           # P @ V
    return f


def flash_flops(N, d, Bk):
    """Exact FLOP count matching flash_attention_v1/v2 implementation."""
    f = 0
    num_blocks = (N + Bk - 1) // Bk
    for _ in range(N):  # per query row
        for kb in range(num_blocks):
            bsz = min(Bk, N - kb * Bk)
            f += bsz * (2 * d - 1)       # block scores
            f += max(0, bsz - 1)         # block max
            f += bsz + bsz              # sub_max + exp
            f += max(0, bsz - 1)         # block sum
            f += d * (2 * bsz - 1)       # block output
            if kb > 0:                   # merge
                f += 1 + 2 + 2 + 3 + d * 3  # max, alpha, beta, l_update, o_update
        f += 1 + d                       # inv + normalize
    return f


def flash_v3_flops(N, d, Bq, Bk):
    """Exact FLOP count for flash_attention_v3 (double-tiled).
    Same arithmetic as v1/v2, just reordered — FLOPs are identical to v1."""
    return flash_flops(N, d, Bk)


# ---------------------------------------------------------------------------
# Matrix factory
# ---------------------------------------------------------------------------

def make_matrix(rows, cols):
    """Create a rows×cols matrix of 1.0 values."""
    return [[1.0] * cols for _ in range(rows)]
