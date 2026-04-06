#!/usr/bin/env python3
"""
Determine where (if ever) Strassen's ByteDMD cost crosses below recursive matmul.
Extends measurements to larger sizes and extrapolates.
"""

import sys, os, json, time, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fast-flash-attention'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import bytedmd_fast as _bfast

_OrigCtx = _bfast._FastContextFenwick
class _BigContext(_OrigCtx):
    def __init__(self, max_ops=50_000_000):
        super().__init__(max_ops)
_bfast._Context = _BigContext

from bytedmd_fast import bytedmd


# ---------------------------------------------------------------------------
# Algorithms (same as benchmark_strassen.py)
# ---------------------------------------------------------------------------

def _zeros(n):
    return [[None] * n for _ in range(n)]

def _add(A, B):
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def _sub(A, B):
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

def _split(M):
    n = len(M)
    h = n // 2
    M11 = [[M[i][j] for j in range(h)] for i in range(h)]
    M12 = [[M[i][j] for j in range(h, n)] for i in range(h)]
    M21 = [[M[i][j] for j in range(h)] for i in range(h, n)]
    M22 = [[M[i][j] for j in range(h, n)] for i in range(h, n)]
    return M11, M12, M21, M22

def _join(C11, C12, C21, C22):
    h = len(C11)
    n = 2 * h
    return [[C11[i][j] if j < h else C12[i][j - h] for j in range(n)] for i in range(h)] + \
           [[C21[i][j] if j < h else C22[i][j - h] for j in range(n)] for i in range(h)]

def naive_matmul(A, B):
    n = len(A)
    C = _zeros(n)
    for i in range(n):
        for j in range(n):
            s = A[i][0] * B[0][j]
            for k in range(1, n):
                s = s + A[i][k] * B[k][j]
            C[i][j] = s
    return C

def _recursive_impl(A, B, leaf):
    n = len(A)
    if n <= leaf:
        return naive_matmul(A, B)
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)
    C11 = _add(_recursive_impl(A11, B11, leaf), _recursive_impl(A12, B21, leaf))
    C12 = _add(_recursive_impl(A11, B12, leaf), _recursive_impl(A12, B22, leaf))
    C21 = _add(_recursive_impl(A21, B11, leaf), _recursive_impl(A22, B21, leaf))
    C22 = _add(_recursive_impl(A21, B12, leaf), _recursive_impl(A22, B22, leaf))
    return _join(C11, C12, C21, C22)

def _strassen_impl(A, B, leaf):
    n = len(A)
    if n <= leaf:
        return naive_matmul(A, B)
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)
    M1 = _strassen_impl(_add(A11, A22), _add(B11, B22), leaf)
    M2 = _strassen_impl(_add(A21, A22), B11, leaf)
    M3 = _strassen_impl(A11, _sub(B12, B22), leaf)
    M4 = _strassen_impl(A22, _sub(B21, B11), leaf)
    M5 = _strassen_impl(_add(A11, A12), B22, leaf)
    M6 = _strassen_impl(_sub(A21, A11), _add(B11, B12), leaf)
    M7 = _strassen_impl(_sub(A12, A22), _add(B21, B22), leaf)
    C11 = _add(_sub(_add(M1, M4), M5), M7)
    C12 = _add(M3, M5)
    C21 = _add(M2, M4)
    C22 = _add(_sub(_add(M1, M3), M2), M6)
    return _join(C11, C12, C21, C22)


# ---------------------------------------------------------------------------
# FLOP counting
# ---------------------------------------------------------------------------

def recursive_flops(n, leaf=1):
    if n <= leaf:
        return n * n * (2 * n - 1)
    h = n // 2
    return 8 * recursive_flops(h, leaf) + 4 * h * h

def strassen_flops(n, leaf=1):
    if n <= leaf:
        return n * n * (2 * n - 1)
    h = n // 2
    return 7 * strassen_flops(h, leaf) + 18 * h * h


# ---------------------------------------------------------------------------
# Measure and extrapolate
# ---------------------------------------------------------------------------

def measure(label, func, n):
    A = np.ones((n, n))
    B = np.ones((n, n))
    print(f"  {label} n={n}...", end="", flush=True)
    t0 = time.time()
    cost = bytedmd(func, (A, B))
    elapsed = time.time() - t0
    print(f" {cost:>14,}  ({elapsed:.1f}s)")
    return cost, elapsed


def fit_power_law(sizes, costs):
    """Fit cost = C * n^alpha. Returns (C, alpha)."""
    log_n = [math.log(n) for n in sizes]
    log_c = [math.log(c) for c in costs]
    # Linear regression on log-log
    n_pts = len(sizes)
    sum_x = sum(log_n)
    sum_y = sum(log_c)
    sum_xy = sum(x * y for x, y in zip(log_n, log_c))
    sum_x2 = sum(x * x for x in log_n)
    alpha = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_x2 - sum_x * sum_x)
    log_C = (sum_y - alpha * sum_x) / n_pts
    return math.exp(log_C), alpha


def main():
    print("=" * 80)
    print("FINDING THE BYTEDMD CROSSOVER POINT: Strassen vs Recursive")
    print("=" * 80)
    print()

    leaf_sizes = [1, 2, 4, 8]
    # Measure up to n=128 (and n=256 if time permits)
    sizes = [4, 8, 16, 32, 64, 128]

    results = {}

    for lf in leaf_sizes:
        print(f"\n--- Leaf size = {lf} ---")
        rec_costs = {}
        str_costs = {}

        for n in sizes:
            if n <= lf:
                # Both degenerate to naive — skip
                continue

            rec_func = lambda A, B, _lf=lf: _recursive_impl(A, B, leaf=_lf)
            str_func = lambda A, B, _lf=lf: _strassen_impl(A, B, leaf=_lf)

            rc, rt = measure(f"Recursive(L={lf})", rec_func, n)
            sc, st = measure(f"Strassen(L={lf}) ", str_func, n)

            rec_costs[n] = rc
            str_costs[n] = sc

            ratio = rc / sc
            print(f"    R/S ratio = {ratio:.4f}  ({'Strassen wins' if ratio > 1 else 'Recursive wins'})")

            # Bail if taking too long (>120s for a single measurement)
            if max(rt, st) > 120:
                print(f"    Skipping larger sizes for L={lf} (too slow)")
                break

        results[lf] = {'rec': rec_costs, 'str': str_costs}

    # -----------------------------------------------------------------------
    # Analysis and extrapolation
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("EXTRAPOLATION ANALYSIS")
    print("=" * 80)

    for lf in leaf_sizes:
        data = results.get(lf)
        if not data:
            continue

        rec_sizes = sorted(data['rec'].keys())
        str_sizes = sorted(data['str'].keys())
        common = sorted(set(rec_sizes) & set(str_sizes))

        if len(common) < 3:
            continue

        rec_vals = [data['rec'][n] for n in common]
        str_vals = [data['str'][n] for n in common]
        ratios = [r / s for r, s in zip(rec_vals, str_vals)]

        # Fit power laws
        C_r, alpha_r = fit_power_law(common, rec_vals)
        C_s, alpha_s = fit_power_law(common, str_vals)

        print(f"\n--- Leaf = {lf} ---")
        print(f"  Recursive: ByteDMD ≈ {C_r:.2f} * n^{alpha_r:.4f}")
        print(f"  Strassen:  ByteDMD ≈ {C_s:.2f} * n^{alpha_s:.4f}")
        print(f"  Exponent gap: {alpha_r - alpha_s:+.4f}")
        print()

        # Print ratio progression
        print(f"  {'n':>6}  {'R/S Ratio':>10}  {'Local exponent gap':>20}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*20}")
        prev = None
        for i, n in enumerate(common):
            r = ratios[i]
            if prev is not None and prev[1] != r:
                local_gap = math.log(ratios[i] / prev[1]) / math.log(n / prev[0])
                print(f"  {n:>6}  {r:>10.4f}  {local_gap:>+20.4f}")
            else:
                print(f"  {n:>6}  {r:>10.4f}  {'':>20}")
            prev = (n, r)

        # Extrapolate crossover
        if alpha_r > alpha_s:
            # Crossover when C_r * n^alpha_r = C_s * n^alpha_s
            # n^(alpha_r - alpha_s) = C_s / C_r
            # n = (C_s / C_r)^(1/(alpha_r - alpha_s))
            gap = alpha_r - alpha_s
            if C_s > C_r:
                crossover_n = (C_s / C_r) ** (1.0 / gap)
                print(f"\n  Extrapolated crossover: n ≈ {crossover_n:,.0f}")
                print(f"    (where {C_r:.2f} * n^{alpha_r:.4f} = {C_s:.2f} * n^{alpha_s:.4f})")
                if crossover_n > 1e9:
                    print(f"    This is astronomically large — effectively NEVER for practical purposes.")
                elif crossover_n > 1e6:
                    print(f"    This is impractically large (millions of elements per side).")
                elif crossover_n > 10000:
                    print(f"    This is very large but theoretically reachable.")
                else:
                    print(f"    This is within practical range!")
            else:
                print(f"\n  C_s ({C_s:.2f}) <= C_r ({C_r:.2f}) and alpha_r > alpha_s")
                print(f"  Strassen already wins (or would at small n). Check the ratio data above.")
        else:
            print(f"\n  Strassen exponent ({alpha_s:.4f}) >= Recursive exponent ({alpha_r:.4f})")
            print(f"  No crossover: Strassen's ByteDMD grows at the same rate or faster.")
            print(f"  Strassen will NEVER win on ByteDMD at this leaf size.")

    # -----------------------------------------------------------------------
    # FLOP crossover comparison
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("FLOP CROSSOVER (for reference)")
    print("=" * 80)

    for lf in leaf_sizes:
        common = sorted(results.get(lf, {}).get('rec', {}).keys())
        if len(common) < 2:
            continue
        print(f"\n  Leaf={lf}:")
        for n in common:
            rf = recursive_flops(n, leaf=lf)
            sf = strassen_flops(n, leaf=lf)
            print(f"    n={n:>5}: FLOP R/S = {rf/sf:.4f}", end="")
            if rf > sf:
                print(f"  (Strassen wins by {(rf/sf - 1)*100:.1f}%)")
            else:
                print(f"  (Recursive wins by {(1 - rf/sf)*100:.1f}%)")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), 'crossover_results.json')
    save_data = {}
    for lf in leaf_sizes:
        data = results.get(lf, {})
        save_data[f'L{lf}'] = {
            'recursive': {str(n): v for n, v in data.get('rec', {}).items()},
            'strassen': {str(n): v for n, v in data.get('str', {}).items()},
        }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
