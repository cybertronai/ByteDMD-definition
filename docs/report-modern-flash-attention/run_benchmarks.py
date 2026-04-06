#!/usr/bin/env python3
"""
Comprehensive benchmark: ByteDMD vs FLOPs for attention variants.

Measures:
  - Naive attention
  - Flash attention v1 (basic tiling)
  - Flash attention v2 (swizzled K/V ordering)
  - Flash attention v3 (double-tiled Q×K/V)

across a range of sequence lengths, head dimensions, and block sizes.
"""
import sys, os, time, json, math

sys.path.insert(0, os.path.dirname(__file__))
from bytedmd_fast import bytedmd, traced_eval, trace_to_bytedmd
from attention_variants import (
    naive_attention, flash_attention_v1, flash_attention_v2, flash_attention_v3,
    naive_flops, flash_flops, flash_v3_flops, make_matrix,
)


def measure_bytedmd(func, args):
    """Measure ByteDMD cost (using the fast tracer, no assert_noescape)."""
    trace, _ = traced_eval(func, args)
    return trace_to_bytedmd(trace, bytes_per_element=1)


def run_single(name, func, args, flops, timings):
    """Run a single measurement, returning result dict."""
    t0 = time.time()
    cost = measure_bytedmd(func, args)
    elapsed = time.time() - t0
    timings[name] = elapsed
    return {
        'method': name,
        'bytedmd': cost,
        'flops': flops,
        'bytedmd_per_flop': round(cost / flops, 4) if flops > 0 else 0,
        'measurement_time_s': round(elapsed, 3),
    }


def run_benchmarks():
    """Run all benchmarks and return results."""
    # Configuration space
    # (seq_len, head_dim)
    size_configs = [
        (4, 2),
        (8, 2),
        (16, 2),
        (32, 2),
        (64, 2),
        (128, 2),
        # Also test d=4 for a subset
        (8, 4),
        (16, 4),
        (32, 4),
        (64, 4),
    ]

    # Block sizes to test for flash variants
    block_sizes = [2, 4, 8, 16]

    all_results = []
    
    print(f"{'Config':<16} {'Method':<28} {'ByteDMD':>10} {'FLOPs':>10} "
          f"{'DMD/FLOP':>10} {'Time(s)':>8}")
    print("=" * 88)

    for N, d in size_configs:
        config = f"N={N},d={d}"
        Q = make_matrix(N, d)
        K = make_matrix(N, d)
        V = make_matrix(N, d)

        nf = naive_flops(N, d)
        timings = {}

        # --- Naive ---
        print(f"\n{config}")
        try:
            r = run_single("naive", naive_attention, (Q, K, V), nf, timings)
            r['N'] = N
            r['d'] = d
            r['Bk'] = None
            r['Bq'] = None
            all_results.append(r)
            print(f"  {'naive':<26} {r['bytedmd']:>10,} {r['flops']:>10,} "
                  f"{r['bytedmd_per_flop']:>10.2f} {r['measurement_time_s']:>8.1f}")
        except Exception as e:
            print(f"  naive: FAILED ({e})")

        # --- Flash v1 ---
        for Bk in block_sizes:
            if Bk > N:
                continue
            name = f"flash_v1(Bk={Bk})"
            ff = flash_flops(N, d, Bk)
            try:
                # Need fresh copies since traced_eval might mutate lists
                Q2 = make_matrix(N, d)
                K2 = make_matrix(N, d)
                V2 = make_matrix(N, d)
                r = run_single(name,
                               lambda Q, K, V, _Bk=Bk: flash_attention_v1(Q, K, V, Bk=_Bk),
                               (Q2, K2, V2), ff, timings)
                r['N'] = N
                r['d'] = d
                r['Bk'] = Bk
                r['Bq'] = None
                all_results.append(r)
                print(f"  {name:<26} {r['bytedmd']:>10,} {r['flops']:>10,} "
                      f"{r['bytedmd_per_flop']:>10.2f} {r['measurement_time_s']:>8.1f}")
            except Exception as e:
                print(f"  {name}: FAILED ({e})")

        # --- Flash v2 (swizzled) ---
        for Bk in block_sizes:
            if Bk > N:
                continue
            name = f"flash_v2(Bk={Bk})"
            ff = flash_flops(N, d, Bk)
            try:
                Q2 = make_matrix(N, d)
                K2 = make_matrix(N, d)
                V2 = make_matrix(N, d)
                r = run_single(name,
                               lambda Q, K, V, _Bk=Bk: flash_attention_v2(Q, K, V, Bk=_Bk),
                               (Q2, K2, V2), ff, timings)
                r['N'] = N
                r['d'] = d
                r['Bk'] = Bk
                r['Bq'] = None
                all_results.append(r)
                print(f"  {name:<26} {r['bytedmd']:>10,} {r['flops']:>10,} "
                      f"{r['bytedmd_per_flop']:>10.2f} {r['measurement_time_s']:>8.1f}")
            except Exception as e:
                print(f"  {name}: FAILED ({e})")

        # --- Flash v3 (double-tiled) ---
        for Bq in [2, 4]:
            for Bk in block_sizes:
                if Bk > N or Bq > N:
                    continue
                name = f"flash_v3(Bq={Bq},Bk={Bk})"
                ff = flash_v3_flops(N, d, Bq, Bk)
                try:
                    Q2 = make_matrix(N, d)
                    K2 = make_matrix(N, d)
                    V2 = make_matrix(N, d)
                    r = run_single(name,
                                   lambda Q, K, V, _Bq=Bq, _Bk=Bk: flash_attention_v3(Q, K, V, Bq=_Bq, Bk=_Bk),
                                   (Q2, K2, V2), ff, timings)
                    r['N'] = N
                    r['d'] = d
                    r['Bk'] = Bk
                    r['Bq'] = Bq
                    all_results.append(r)
                    print(f"  {name:<26} {r['bytedmd']:>10,} {r['flops']:>10,} "
                          f"{r['bytedmd_per_flop']:>10.2f} {r['measurement_time_s']:>8.1f}")
                except Exception as e:
                    print(f"  {name}: FAILED ({e})")

    return all_results


def save_results(results, path):
    """Save results to JSON."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == '__main__':
    results = run_benchmarks()
    
    out_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    save_results(results, out_path)
    
    # Print summary: best flash variant per (N, d)
    print("\n" + "=" * 88)
    print("SUMMARY: Best flash variant vs naive (ByteDMD ratio = naive/flash)")
    print(f"{'Config':<16} {'Best Flash':<28} {'Naive DMD':>10} {'Flash DMD':>10} "
          f"{'DMD Ratio':>10} {'FLOP Ratio':>10}")
    print("-" * 88)
    
    # Group by (N, d)
    from collections import defaultdict
    groups = defaultdict(list)
    naive_costs = {}
    for r in results:
        key = (r['N'], r['d'])
        if r['method'] == 'naive':
            naive_costs[key] = r
        else:
            groups[key].append(r)
    
    for key in sorted(naive_costs.keys()):
        N, d = key
        naive = naive_costs[key]
        if not groups[key]:
            continue
        # Find best flash (lowest ByteDMD)
        best = min(groups[key], key=lambda r: r['bytedmd'])
        dmd_ratio = naive['bytedmd'] / best['bytedmd'] if best['bytedmd'] > 0 else float('inf')
        flop_ratio = naive['flops'] / best['flops'] if best['flops'] > 0 else float('inf')
        
        print(f"N={N:<3},d={d:<3}     {best['method']:<28} {naive['bytedmd']:>10,} "
              f"{best['bytedmd']:>10,} {dmd_ratio:>10.2f}x {flop_ratio:>10.2f}x")
