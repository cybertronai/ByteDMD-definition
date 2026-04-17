#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Compare the theoretical explicit RMM energy formula against empirical costs.

Theory (from gemini/matmul-explicit-formula.md):
  E_explicit(N) ≈ (20/3 + √3) · N³ log₂ N + 5 · N³

  where (20/3 + √3) ≈ 8.3987

Also compares the static in-place baseline:
  E_static(N) ≈ 2√3 · N⁴ ≈ 3.4641 · N⁴
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import manual_matmul as mm


COEFF_EXPLICIT = 20/3 + math.sqrt(3)  # ≈ 8.3987
COEFF_STATIC = 2 * math.sqrt(3)       # ≈ 3.4641


def E_explicit_theory(N):
    if N < 2:
        return 5
    return COEFF_EXPLICIT * N**3 * math.log2(N) + 5 * N**3


def E_static_theory(N):
    return COEFF_STATIC * N**4


def run_experiments(Ns):
    results = []
    for N in Ns:
        print(f'N={N:>4d} ...', end=' ', flush=True)
        t0 = time.time()

        A = [[1.0] * N for _ in range(N)]
        B = [[1.0] * N for _ in range(N)]

        cost_explicit = mm.matmul_explicit_rmm(A, B)
        cost_naive = mm.matmul_naive_manual(A, B)

        theory_exp = E_explicit_theory(N)
        theory_static = E_static_theory(N)

        ratio_exp = cost_explicit / theory_exp if theory_exp > 0 else float('inf')
        ratio_static = cost_naive / theory_static if theory_static > 0 else float('inf')

        elapsed = time.time() - t0
        print(f'explicit={cost_explicit:>14,}  theory={theory_exp:>14,.0f}  '
              f'ratio={ratio_exp:.4f}  |  naive={cost_naive:>14,}  '
              f'theory_static={theory_static:>14,.0f}  ratio={ratio_static:.4f}  '
              f'({elapsed:.1f}s)')

        results.append({
            'N': N,
            'cost_explicit': cost_explicit,
            'cost_naive': cost_naive,
            'theory_explicit': theory_exp,
            'theory_static': theory_static,
            'ratio_explicit': ratio_exp,
            'ratio_static': ratio_static,
        })
    return results


def plot_absolute(results, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    Ns = np.array([r['N'] for r in results])
    emp_exp = np.array([r['cost_explicit'] for r in results], dtype=float)
    thy_exp = np.array([r['theory_explicit'] for r in results], dtype=float)
    emp_naive = np.array([r['cost_naive'] for r in results], dtype=float)
    thy_static = np.array([r['theory_static'] for r in results], dtype=float)

    # Left panel: absolute costs
    ax1.loglog(Ns, emp_exp, 'o-', color='tab:blue', linewidth=2.2, markersize=8,
               label='Explicit RMM (empirical)')
    ax1.loglog(Ns, thy_exp, 's--', color='tab:cyan', linewidth=1.8, markersize=7,
               label=f'Theory: ({COEFF_EXPLICIT:.4f})·N³log₂N + 5N³')
    ax1.loglog(Ns, emp_naive, 'o-', color='tab:red', linewidth=2.2, markersize=8,
               label='Naive matmul (empirical)')
    ax1.loglog(Ns, thy_static, 's--', color='tab:orange', linewidth=1.8, markersize=7,
               label=f'Theory: ({COEFF_STATIC:.4f})·N⁴')

    ax1.set_xlabel('Matrix size N', fontsize=12)
    ax1.set_ylabel('Total ByteDMD cost', fontsize=12)
    ax1.set_title('Absolute cost: Theory vs Empirical', fontsize=13)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(fontsize=9, loc='upper left')

    # Right panel: ratios (empirical / theory)
    ratio_exp = emp_exp / thy_exp
    ratio_static = emp_naive / thy_static

    ax2.plot(Ns, ratio_exp, 'o-', color='tab:blue', linewidth=2.2, markersize=8,
             label='Explicit RMM: empirical / theory')
    ax2.plot(Ns, ratio_static, 'o-', color='tab:red', linewidth=2.2, markersize=8,
             label='Naive: empirical / theory')
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.6, label='Perfect match (1.0)')

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Matrix size N', fontsize=12)
    ax2.set_ylabel('Empirical / Theoretical', fontsize=12)
    ax2.set_title('Convergence of empirical to theory', fontsize=13)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(fontsize=9)

    fig.suptitle(
        f'Explicit RMM Energy Formula Validation\n'
        f'E_explicit ≈ (20/3+√3)·N³log₂N + 5N³ ≈ {COEFF_EXPLICIT:.4f}·N³log₂N\n'
        f'E_static ≈ 2√3·N⁴ ≈ {COEFF_STATIC:.4f}·N⁴',
        fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def plot_normalized(results, out_path):
    """Plot cost / (N³ log₂ N) to see the coefficient converge."""
    fig, ax = plt.subplots(figsize=(10, 6))

    Ns = np.array([r['N'] for r in results if r['N'] >= 2])
    emp_exp = np.array([r['cost_explicit'] for r in results if r['N'] >= 2], dtype=float)
    emp_naive = np.array([r['cost_naive'] for r in results if r['N'] >= 2], dtype=float)

    norm_exp = emp_exp / (Ns**3 * np.log2(Ns))
    norm_naive = emp_naive / (Ns**3 * np.log2(Ns))

    ax.plot(Ns, norm_exp, 'o-', color='tab:blue', linewidth=2.5, markersize=9,
            label='Explicit RMM: cost / (N³ log₂ N)')
    ax.axhline(y=COEFF_EXPLICIT, color='tab:cyan', linestyle='--', linewidth=1.5,
               label=f'Theory: 20/3 + √3 ≈ {COEFF_EXPLICIT:.4f}')

    ax.plot(Ns, norm_naive, 'o-', color='tab:red', linewidth=2.5, markersize=9,
            label='Naive: cost / (N³ log₂ N)')

    # Also show naive / N⁴ on secondary axis
    ax2 = ax.twinx()
    norm_naive_n4 = emp_naive / Ns.astype(float)**4
    ax2.plot(Ns, norm_naive_n4, 's--', color='tab:orange', linewidth=1.5, markersize=7,
             label=f'Naive: cost / N⁴')
    ax2.axhline(y=COEFF_STATIC, color='tab:orange', linestyle=':', linewidth=1.2,
                label=f'Theory: 2√3 ≈ {COEFF_STATIC:.4f}')
    ax2.set_ylabel('Naive cost / N⁴', fontsize=11, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.legend(fontsize=8, loc='center right')

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Matrix size N', fontsize=12)
    ax.set_ylabel('Cost / (N³ log₂ N)', fontsize=12)
    ax.set_title('Normalized cost: coefficient convergence', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def main():
    Ns_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if Ns_arg:
        Ns = [int(x) for x in Ns_arg.split(',')]
    else:
        Ns = [2, 4, 8, 16, 32, 64, 128]

    print(f'Running formula validation for N = {Ns}')
    print(f'Theory coefficients: explicit = {COEFF_EXPLICIT:.6f}, static = {COEFF_STATIC:.6f}')
    print()

    results = run_experiments(Ns)

    out_dir = os.path.dirname(__file__)
    plot_absolute(results, os.path.join(out_dir, 'formula_vs_empirical.png'))
    plot_normalized(results, os.path.join(out_dir, 'formula_normalized.png'))

    # Summary table
    print(f'\n{"N":>4} | {"Explicit emp":>14} | {"Explicit thy":>14} | '
          f'{"Ratio":>8} | {"Naive emp":>14} | {"Static thy":>14} | {"Ratio":>8}')
    print('-' * 90)
    for r in results:
        print(f'{r["N"]:>4} | {r["cost_explicit"]:>14,} | {r["theory_explicit"]:>14,.0f} | '
              f'{r["ratio_explicit"]:>8.4f} | {r["cost_naive"]:>14,} | '
              f'{r["theory_static"]:>14,.0f} | {r["ratio_static"]:>8.4f}')


if __name__ == '__main__':
    main()
