#!/usr/bin/env python3
"""
Comprehensive benchmark: Strassen vs Recursive matrix multiplication.

Compares FLOP counts and ByteDMD costs across matrix sizes (powers of 2).
Baseline is recursive (divide-and-conquer with 8 subproblems), not naive triple-loop.
Uses the fast Fenwick-tree ByteDMD tracer for larger matrices.
"""

import sys, os, json, time, math

# Use the fast ByteDMD tracer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fast-flash-attention'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import bytedmd_fast as _bfast

# Increase Fenwick tree capacity for large benchmarks
_OrigCtx = _bfast._FastContextFenwick
class _BigContext(_OrigCtx):
    def __init__(self, max_ops=20_000_000):
        super().__init__(max_ops)
_bfast._Context = _BigContext

from bytedmd_fast import bytedmd, traced_eval, trace_to_bytedmd

import numpy as np


# ---------------------------------------------------------------------------
# Matrix helpers (pure-Python lists-of-lists, compatible with ByteDMD tracer)
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


# ---------------------------------------------------------------------------
# Naive matrix multiply (i-j-k) — used as leaf kernel
# ---------------------------------------------------------------------------

def naive_matmul(A, B):
    """Standard O(n^3) matrix multiply."""
    n = len(A)
    C = _zeros(n)
    for i in range(n):
        for j in range(n):
            s = A[i][0] * B[0][j]
            for k in range(1, n):
                s = s + A[i][k] * B[k][j]
            C[i][j] = s
    return C


# ---------------------------------------------------------------------------
# Recursive matrix multiply (8 subproblems — same structure as Strassen but
# without the Strassen trick of reducing to 7 multiplications)
# ---------------------------------------------------------------------------

def _recursive_impl(A, B, leaf):
    n = len(A)
    if n <= leaf:
        return naive_matmul(A, B)
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    C11 = _add(_recursive_impl(A11, B11, leaf), _recursive_impl(A12, B21, leaf))
    C12 = _add(_recursive_impl(A11, B12, leaf), _recursive_impl(A12, B22, leaf))
    C21 = _add(_recursive_impl(A21, B11, leaf), _recursive_impl(A22, B21, leaf))
    C22 = _add(_recursive_impl(A21, B12, leaf), _recursive_impl(A22, B22, leaf))
    return _join(C11, C12, C21, C22)


def recursive_matmul_l1(A, B):
    return _recursive_impl(A, B, leaf=1)

def recursive_matmul_l2(A, B):
    return _recursive_impl(A, B, leaf=2)

def recursive_matmul_l4(A, B):
    return _recursive_impl(A, B, leaf=4)

def recursive_matmul_l8(A, B):
    return _recursive_impl(A, B, leaf=8)


# ---------------------------------------------------------------------------
# Strassen algorithm (7 subproblems + 18 additions)
# ---------------------------------------------------------------------------

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


def strassen_matmul_l1(A, B):
    return _strassen_impl(A, B, leaf=1)

def strassen_matmul_l2(A, B):
    return _strassen_impl(A, B, leaf=2)

def strassen_matmul_l4(A, B):
    return _strassen_impl(A, B, leaf=4)

def strassen_matmul_l8(A, B):
    return _strassen_impl(A, B, leaf=8)


# ---------------------------------------------------------------------------
# FLOP counting (analytical)
# ---------------------------------------------------------------------------

def naive_flops(n):
    """Naive matmul: n^2 dot products, each with n muls and n-1 adds."""
    return n * n * (2 * n - 1)


def recursive_flops(n, leaf=1):
    """Recursive matmul: T(n) = 8*T(n/2) + 4*(n/2)^2 additions for combining."""
    if n <= leaf:
        return naive_flops(n)
    h = n // 2
    return 8 * recursive_flops(h, leaf) + 4 * h * h


def strassen_flops(n, leaf=1):
    """Strassen FLOPs: T(n) = 7*T(n/2) + 18*(n/2)^2 additions for combining."""
    if n <= leaf:
        return naive_flops(n)
    h = n // 2
    return 7 * strassen_flops(h, leaf) + 18 * h * h


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmarks():
    sizes = [2, 4, 8, 16, 32, 64]
    leaf_sizes = [1, 2, 4, 8]

    print("=" * 95)
    print("STRASSEN vs RECURSIVE MATRIX MULTIPLICATION: FLOP & ByteDMD BENCHMARK")
    print("=" * 95)
    print()
    print("Baseline: Recursive matmul (8 subproblems, O(n^3))")
    print("Compared: Strassen matmul  (7 subproblems, O(n^2.807))")
    print("Both use the same divide-and-conquer structure; Strassen trades 1 multiply")
    print("for extra additions at each recursion level.")

    # -----------------------------------------------------------------------
    # Table 1: FLOP Ratios — Recursive / Strassen at each leaf size
    # -----------------------------------------------------------------------
    print()
    print("TABLE 1: FLOP Counts")
    print("-" * 95)
    header = f"{'n':>4}  {'Recursive':>12}  {'Strassen':>12}"
    for lf in leaf_sizes:
        header += f"  {'R/S L=' + str(lf):>10}"
    print(header)
    print("-" * 95)

    for n in sizes:
        row = f"{n:>4}"
        # Show absolute counts at leaf=1
        rf = recursive_flops(n, leaf=1)
        sf = strassen_flops(n, leaf=1)
        row += f"  {rf:>12,}  {sf:>12,}"
        # Ratios at each leaf size
        for lf in leaf_sizes:
            rf_l = recursive_flops(n, leaf=lf) if n > lf else naive_flops(n)
            sf_l = strassen_flops(n, leaf=lf) if n > lf else naive_flops(n)
            ratio = rf_l / sf_l if sf_l > 0 else 1.0
            row += f"  {ratio:>10.3f}"
        print(row)

    print()
    print("  R/S > 1: Strassen does fewer FLOPs.  R/S = 1: identical (leaf dominates).")
    print(f"  Theoretical limit: 8/7 = {8/7:.3f} per recursion level")

    # -----------------------------------------------------------------------
    # Measure ByteDMD
    # -----------------------------------------------------------------------
    print()
    print("Measuring ByteDMD costs...")
    print()

    rec_funcs = {1: recursive_matmul_l1, 2: recursive_matmul_l2,
                 4: recursive_matmul_l4, 8: recursive_matmul_l8}
    str_funcs = {1: strassen_matmul_l1, 2: strassen_matmul_l2,
                 4: strassen_matmul_l4, 8: strassen_matmul_l8}

    dmd_rec = {}   # (n, leaf) -> cost
    dmd_str = {}   # (n, leaf) -> cost

    for n in sizes:
        A = np.ones((n, n))
        B = np.ones((n, n))

        for lf in leaf_sizes:
            if lf >= n:
                # Both degenerate to naive
                if (n, lf) not in dmd_rec:
                    print(f"  n={n:>2} naive (leaf≥n)...", end="", flush=True)
                    t0 = time.time()
                    cost = bytedmd(naive_matmul, (A, B))
                    print(f" {cost:>14,}  ({time.time()-t0:.2f}s)")
                    dmd_rec[(n, lf)] = cost
                    dmd_str[(n, lf)] = cost
                else:
                    dmd_rec[(n, lf)] = dmd_rec[(n, leaf_sizes[0])]
                    dmd_str[(n, lf)] = dmd_str[(n, leaf_sizes[0])]
                continue

            # Recursive
            print(f"  n={n:>2} Recursive(L={lf})...", end="", flush=True)
            t0 = time.time()
            dmd_rec[(n, lf)] = bytedmd(rec_funcs[lf], (A, B))
            print(f" {dmd_rec[(n, lf)]:>14,}  ({time.time()-t0:.2f}s)")

            # Strassen
            print(f"  n={n:>2} Strassen(L={lf})...", end="", flush=True)
            t0 = time.time()
            dmd_str[(n, lf)] = bytedmd(str_funcs[lf], (A, B))
            print(f" {dmd_str[(n, lf)]:>14,}  ({time.time()-t0:.2f}s)")

    # -----------------------------------------------------------------------
    # Table 2: ByteDMD absolute costs
    # -----------------------------------------------------------------------
    print()
    print("TABLE 2: ByteDMD Costs — Absolute")
    print("-" * 95)
    header = f"{'n':>4}  {'':>6}"
    for lf in leaf_sizes:
        header += f"  {'L=' + str(lf):>14}"
    print(header)
    print("-" * 95)

    for n in sizes:
        row_r = f"{n:>4}  {'Recur':>6}"
        row_s = f"{'':>4}  {'Stras':>6}"
        for lf in leaf_sizes:
            row_r += f"  {dmd_rec[(n, lf)]:>14,}"
            row_s += f"  {dmd_str[(n, lf)]:>14,}"
        print(row_r)
        print(row_s)
        print()

    # -----------------------------------------------------------------------
    # Table 3: Combined FLOP and ByteDMD ratios (Recursive / Strassen)
    # -----------------------------------------------------------------------
    print("TABLE 3: Recursive/Strassen Ratios — FLOPs and ByteDMD side by side")
    print("         (>1 = Strassen wins, <1 = Recursive wins)")
    print("-" * 95)
    header = f"{'n':>4}  {'':>6}"
    for lf in leaf_sizes:
        header += f"  {'L=' + str(lf):>10}"
    print(header)
    print("-" * 95)

    for n in sizes:
        # FLOP ratios
        row_f = f"{n:>4}  {'FLOPs':>6}"
        for lf in leaf_sizes:
            rf = recursive_flops(n, leaf=lf) if n > lf else naive_flops(n)
            sf = strassen_flops(n, leaf=lf) if n > lf else naive_flops(n)
            ratio = rf / sf if sf > 0 else 1.0
            row_f += f"  {ratio:>10.3f}"

        # ByteDMD ratios
        row_d = f"{'':>4}  {'DMD':>6}"
        for lf in leaf_sizes:
            rc = dmd_rec[(n, lf)]
            sc = dmd_str[(n, lf)]
            ratio = rc / sc if sc > 0 else 1.0
            row_d += f"  {ratio:>10.3f}"

        print(row_f)
        print(row_d)
        print()

    # -----------------------------------------------------------------------
    # Table 4: Scaling exponents
    # -----------------------------------------------------------------------
    print("TABLE 4: Empirical Scaling Exponents (log2(cost(2n)/cost(n)))")
    print("-" * 85)
    print(f"{'n→2n':>8}  {'R FLOP':>8}  {'S FLOP':>8}  {'R DMD L1':>9}  {'S DMD L1':>9}  {'R DMD L4':>9}  {'S DMD L4':>9}")
    print("-" * 85)

    prev_n = None
    for n in sizes:
        if prev_n is not None:
            log_r = math.log2(n / prev_n)
            rf_exp = math.log2(recursive_flops(n) / recursive_flops(prev_n)) / log_r
            sf_exp = math.log2(strassen_flops(n) / strassen_flops(prev_n)) / log_r

            rd1_exp = math.log2(dmd_rec[(n, 1)] / dmd_rec[(prev_n, 1)]) / log_r if dmd_rec.get((prev_n, 1), 0) > 0 else 0
            sd1_exp = math.log2(dmd_str[(n, 1)] / dmd_str[(prev_n, 1)]) / log_r if dmd_str.get((prev_n, 1), 0) > 0 else 0

            # L=4 exponents (skip if both are same as naive)
            r4 = dmd_rec.get((n, 4), 0)
            r4p = dmd_rec.get((prev_n, 4), 0)
            s4 = dmd_str.get((n, 4), 0)
            s4p = dmd_str.get((prev_n, 4), 0)
            rd4_exp = math.log2(r4 / r4p) / log_r if r4p > 0 and r4 > 0 else 0
            sd4_exp = math.log2(s4 / s4p) / log_r if s4p > 0 and s4 > 0 else 0

            print(f"  {prev_n:>2}→{n:<3} {rf_exp:>8.3f}  {sf_exp:>8.3f}  {rd1_exp:>9.3f}  {sd1_exp:>9.3f}  {rd4_exp:>9.3f}  {sd4_exp:>9.3f}")
        prev_n = n

    print()
    print("  Expected FLOP exponents: Recursive=3.000, Strassen=log2(7)≈2.807")

    # -----------------------------------------------------------------------
    # Table 5: Summary — does Strassen's FLOP advantage translate to ByteDMD?
    # -----------------------------------------------------------------------
    print()
    print("TABLE 5: Does Strassen's FLOP Advantage Translate to ByteDMD Advantage?")
    print("-" * 85)
    header = f"{'n':>4}  {'Leaf':>5}  {'FLOP R/S':>9}  {'DMD R/S':>9}  {'FLOP win':>9}  {'DMD win':>9}  {'Agreement':>10}"
    print(header)
    print("-" * 85)

    for n in sizes:
        for lf in leaf_sizes:
            rf = recursive_flops(n, leaf=lf) if n > lf else naive_flops(n)
            sf = strassen_flops(n, leaf=lf) if n > lf else naive_flops(n)
            rc = dmd_rec[(n, lf)]
            sc = dmd_str[(n, lf)]
            flop_ratio = rf / sf if sf > 0 else 1.0
            dmd_ratio = rc / sc if sc > 0 else 1.0
            fw = "Strassen" if flop_ratio > 1.001 else ("Recur" if flop_ratio < 0.999 else "Tie")
            dw = "Strassen" if dmd_ratio > 1.001 else ("Recur" if dmd_ratio < 0.999 else "Tie")
            agree = "YES" if fw == dw else "NO"
            # Skip trivial ties where leaf >= n
            if lf >= n:
                continue
            print(f"{n:>4}  {'L='+str(lf):>5}  {flop_ratio:>9.3f}  {dmd_ratio:>9.3f}  {fw:>9}  {dw:>9}  {agree:>10}")

    print()
    print("KEY INSIGHTS:")
    print("  - When comparing against recursive (not naive), both algorithms share the same")
    print("    divide-and-conquer overhead (split/join). The only difference is 7 vs 8")
    print("    subproblem multiplications and different addition patterns.")
    print("  - FLOP ratio approaches 8/7 ≈ 1.143 as n grows (one fewer multiply per level).")
    print("  - ByteDMD ratio shows whether the FLOP savings survive the data-movement model.")
    print("  - Agreement column: YES = ByteDMD confirms FLOPs; NO = data movement disagrees.")

    # Save results
    output = {
        'sizes': sizes,
        'leaf_sizes': leaf_sizes,
        'recursive': {},
        'strassen': {},
    }
    for lf in leaf_sizes:
        output['recursive'][f'L{lf}'] = {}
        output['strassen'][f'L{lf}'] = {}
        for n in sizes:
            rf = recursive_flops(n, leaf=lf) if n > lf else naive_flops(n)
            sf = strassen_flops(n, leaf=lf) if n > lf else naive_flops(n)
            output['recursive'][f'L{lf}'][str(n)] = {
                'flops': rf, 'bytedmd': dmd_rec.get((n, lf), None)
            }
            output['strassen'][f'L{lf}'][str(n)] = {
                'flops': sf, 'bytedmd': dmd_str.get((n, lf), None)
            }

    results_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    run_benchmarks()
