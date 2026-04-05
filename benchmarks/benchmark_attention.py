#!/usr/bin/env python3
"""
Measure ByteDMD costs for naive attention vs flash attention.

Flash attention has the same FLOPs as naive attention but dramatically
better data locality. ByteDMD (which measures data movement cost via
LRU stack distances) should capture this difference, while FLOPs cannot.
"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bytedmd import bytedmd, traced_eval, trace_to_bytedmd


# --- Naive (standard) attention ---

def naive_attention(Q, K, V):
    """
    Standard attention: softmax(Q @ K^T / sqrt(d)) @ V
    Q, K, V are lists-of-lists: [seq_len][head_dim]
    Returns: [seq_len][head_dim]
    """
    N = len(Q)
    d = len(Q[0])
    scale = d ** -0.5

    # Compute S = Q @ K^T, scaled
    S = [[None] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            acc = Q[i][0] * K[j][0]
            for dd in range(1, d):
                acc = acc + Q[i][dd] * K[j][dd]
            S[i][j] = acc * scale

    # Row-wise softmax
    P = [[None] * N for _ in range(N)]
    for i in range(N):
        # Find max for numerical stability
        mx = S[i][0]
        for j in range(1, N):
            # Use arithmetic comparison to avoid bool short-circuit issues
            diff = S[i][j] - mx
            # We need a branch here; use a simple approach
            mx = _max_of_two(mx, S[i][j])
        # Exponentiate and sum
        row_sum = None
        for j in range(N):
            P[i][j] = _exp_approx(S[i][j] - mx)
            if row_sum is None:
                row_sum = P[i][j]
            else:
                row_sum = row_sum + P[i][j]
        # Normalize
        inv_sum = _inv(row_sum)
        for j in range(N):
            P[i][j] = P[i][j] * inv_sum

    # Output O = P @ V
    O = [[None] * d for _ in range(N)]
    for i in range(N):
        for dd in range(d):
            acc = P[i][0] * V[0][dd]
            for j in range(1, N):
                acc = acc + P[i][j] * V[j][dd]
            O[i][dd] = acc

    return O


# --- Flash attention (tiled) ---

def flash_attention(Q, K, V, Bk=2):
    """
    Flash attention with tiling over K/V blocks.

    Instead of materializing the full N x N attention matrix, processes
    K/V in blocks of size Bk. This keeps the working set small,
    reducing LRU stack distances (and thus ByteDMD cost).

    Uses online softmax (Milakov & Gimelshein 2018) to combine blocks.
    """
    N = len(Q)
    d = len(Q[0])
    scale = d ** -0.5
    num_kv_blocks = (N + Bk - 1) // Bk

    # For each query row, accumulate attention across K/V blocks
    O = [[None] * d for _ in range(N)]

    for i in range(N):
        # Running max, running sum of exp, running output
        m_prev = None  # running max
        l_prev = None  # running sum of exp(s - m)
        o_acc = [None] * d  # running weighted output (unnormalized)

        for kb in range(num_kv_blocks):
            k_start = kb * Bk
            k_end = k_start + Bk
            if k_end > N:
                k_end = N
            block_size = k_end - k_start

            # Compute attention scores for this block: s[j] = q . k[j] / sqrt(d)
            s_block = [None] * block_size
            for j in range(block_size):
                kj = k_start + j
                acc = Q[i][0] * K[kj][0]
                for dd in range(1, d):
                    acc = acc + Q[i][dd] * K[kj][dd]
                s_block[j] = acc * scale

            # Find block max
            m_block = s_block[0]
            for j in range(1, block_size):
                m_block = _max_of_two(m_block, s_block[j])

            # Exponentiate block scores
            p_block = [None] * block_size
            l_block = None
            for j in range(block_size):
                p_block[j] = _exp_approx(s_block[j] - m_block)
                if l_block is None:
                    l_block = p_block[j]
                else:
                    l_block = l_block + p_block[j]

            # Compute block's contribution to output: sum_j p[j] * V[j]
            o_block = [None] * d
            for dd in range(d):
                acc = p_block[0] * V[k_start][dd]
                for j in range(1, block_size):
                    acc = acc + p_block[j] * V[k_start + j][dd]
                o_block[dd] = acc

            # Online softmax: merge this block with running state
            if m_prev is None:
                # First block
                m_prev = m_block
                l_prev = l_block
                for dd in range(d):
                    o_acc[dd] = o_block[dd]
            else:
                # Merge: new_m = max(m_prev, m_block)
                m_new = _max_of_two(m_prev, m_block)
                # Rescale factors
                alpha = _exp_approx(m_prev - m_new)  # rescale old
                beta = _exp_approx(m_block - m_new)   # rescale new block
                l_new = alpha * l_prev + beta * l_block
                for dd in range(d):
                    o_acc[dd] = alpha * o_acc[dd] + beta * o_block[dd]
                m_prev = m_new
                l_prev = l_new

        # Final normalization
        inv_l = _inv(l_prev)
        for dd in range(d):
            O[i][dd] = o_acc[dd] * inv_l

    return O


# --- Helper functions that work with ByteDMD tracking ---

def _max_of_two(a, b):
    """Max of two values. Reads both a and b (2 reads), returns result.
    Real hardware max reads 2 operands and writes 1 result.
    We use a+b which has the same read pattern."""
    return a + b  # same data movement as max: reads a and b once each


def _exp_approx(x):
    """Approximate exp(x). Reads x once, produces result.
    Real exp reads 1 operand, writes 1 result."""
    return x * x  # reads x, produces new value -- 1 read like real exp


def _inv(x):
    """Approximate 1/x. Reads x once, produces result."""
    return x * x  # reads x, produces new value -- 1 read like real inv


# --- FLOPs counting ---

def count_attention_flops(N, d):
    """Count FLOPs for standard attention (same for naive and flash)."""
    qk_flops = N * N * (2 * d - 1)   # Q @ K^T: N*N dot products of size d
    softmax_flops = N * N * 2 + N     # exp + normalize per row (approximate)
    av_flops = N * d * (2 * N - 1)    # P @ V: N*d dot products of size N
    return qk_flops + softmax_flops + av_flops


def make_matrix(rows, cols):
    return [[1.0] * cols for _ in range(rows)]


# --- Run benchmarks ---

def run_benchmark(name, func, args):
    """Run ByteDMD measurement."""
    cost = bytedmd(func, args)
    return cost


if __name__ == '__main__':
    import json

    # Configurations: (seq_len, head_dim, flash_block_size)
    # These are small for tractability but show the trend
    configs = [
        # (seq_len, head_dim, flash_block_size)
        # Vary seq_len to show scaling
        (4, 2, 2),
        (8, 2, 2),
        (8, 2, 4),
        (16, 2, 2),
        (16, 2, 4),
        (16, 2, 8),
        (32, 2, 2),
        (32, 2, 4),
        (32, 2, 8),
        (32, 2, 16),
        # Vary head_dim
        (8, 4, 2),
        (8, 4, 4),
        (16, 4, 2),
        (16, 4, 4),
    ]

    results = []

    print(f"{'Config':<22} {'Method':<20} {'ByteDMD':>10} {'FLOPs':>10} {'ByteDMD/FLOP':>14}")
    print("=" * 80)

    for N, d, Bk in configs:
        Q = make_matrix(N, d)
        K = make_matrix(N, d)
        V = make_matrix(N, d)

        config_str = f"N={N}, d={d}"
        flops = count_attention_flops(N, d)

        # Naive attention
        t0 = time.time()
        naive_cost = run_benchmark("naive", naive_attention, (Q, K, V))
        t_naive = time.time() - t0

        print(f"{config_str:<22} {'naive':<20} {naive_cost:>10} {flops:>10} {naive_cost/flops:>14.4f}")

        # Flash attention with given block size
        flash_name = f"flash (Bk={Bk})"
        t0 = time.time()
        flash_cost = run_benchmark(flash_name,
                                   lambda Q, K, V: flash_attention(Q, K, V, Bk=Bk),
                                   (Q, K, V))
        t_flash = time.time() - t0

        # Flash has slightly more FLOPs due to online softmax rescaling
        # Extra per query row per KV block merge: ~5d + 5 FLOPs
        num_merges = max(0, ((N + Bk - 1) // Bk) - 1)
        flash_extra_flops = N * num_merges * (5 * d + 5)
        flash_flops = flops + flash_extra_flops

        print(f"{'':<22} {flash_name:<20} {flash_cost:>10} {flash_flops:>10} {flash_cost/flash_flops:>14.4f}")

        ratio = naive_cost / flash_cost if flash_cost > 0 else float('inf')
        print(f"{'':<22} {'ByteDMD ratio (naive/flash)':<20} {ratio:>10.2f}x")
        print()

        results.append({
            'seq_len': N,
            'head_dim': d,
            'flash_block_size': Bk,
            'naive_bytedmd': naive_cost,
            'flash_bytedmd': flash_cost,
            'naive_flops': flops,
            'flash_flops': flash_flops,
            'bytedmd_ratio': round(ratio, 4),
            'flop_ratio': round(flops / flash_flops, 4),
            'measurement_time_naive': round(t_naive, 3),
            'measurement_time_flash': round(t_flash, 3),
        })

    # Save results as JSON
    results_path = os.path.join(os.path.dirname(__file__), 'attention_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
