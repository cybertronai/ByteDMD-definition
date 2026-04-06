#!/usr/bin/env python3
"""
Pure-Python naive & flash attention for ByteDMD tracing.

Auxiliary ops (exp, max, inv) are modeled as minimal tracked operations
with the correct read count:
  max(a,b) → a+b   (binary: 2 reads)
  exp(x)   → -x    (unary:  1 read)
  inv(x)   → -x    (unary:  1 read)

Values differ from real attention but ByteDMD only measures the
*pattern* of memory accesses, not actual numbers.
"""


def _max2(a, b):
    """Binary op modeling max(a,b): reads a and b."""
    return a + b


def _exp(x):
    """Unary op modeling exp(x): reads x."""
    return -x


def _inv(x):
    """Unary op modeling 1/x: reads x."""
    return -x


# ---------------------------------------------------------------------------
# Naive (standard) attention
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
# Flash attention (tiled, online softmax)
# ---------------------------------------------------------------------------

def flash_attention(Q, K, V, Bk=2):
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
# Analytical FLOP counters (match implementations exactly)
# ---------------------------------------------------------------------------

def naive_flops(N, d):
    f = 0
    f += N * N * (2 * d - 1)                          # Q @ K^T
    f += N * ((N - 1) + N + N + (N - 1) + 1 + N)      # softmax per row
    f += N * d * (2 * N - 1)                           # P @ V
    return f


def flash_flops(N, d, Bk):
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
