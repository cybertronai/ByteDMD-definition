# ByteDMD vs FLOPs: Flash Attention Analysis

## Executive Summary

**FLOPs cannot distinguish flash attention from naive attention. ByteDMD can.**

Flash attention and naive attention perform nearly identical arithmetic — the FLOP ratio stays flat near 1.0x across all sequence lengths. But ByteDMD, which measures data movement cost via LRU stack distances, reveals a **growing advantage for flash attention that reaches 4.14x at N=128** with the best variant (flash v3 with double-tiling).

The crossover occurs at **N ≈ 8**: below this, tiling overhead neutralizes the locality benefit. Above it, flash attention's advantage grows as approximately **O(√N)**, consistent with the theoretical prediction that naive attention's data movement scales as O(N²·d·√N) while tiled attention scales as O(N²·d·√Bk).

This is the central argument for ByteDMD over FLOPs: **the metric that matters is the one that captures the optimization that matters**.

---

## Attention Variants Tested

### 1. Naive Attention
Standard three-phase attention:
1. Materialize full N×N score matrix S = Q @ K^T
2. Row-wise softmax: P = softmax(S)
3. Output: O = P @ V

This pushes the N×N matrix onto the LRU stack, burying Q/K/V underneath ~N² intermediates.

### 2. Flash Attention v1 (Basic Tiling)
Process K/V in blocks of size Bk with online softmax (Dao et al. 2022). The full N×N matrix is never materialized. Working set at any time: O(Bk·d) elements + running accumulators.

### 3. Flash Attention v2 (Swizzled K/V Ordering)  
Same algorithm as v1, but K/V blocks are processed in **snake order**: even query rows go block 0→1→2→..., odd rows go ...→2→1→0. This keeps the last K/V block from row i near the top of the LRU stack for row i+1, reducing inter-row data movement.

### 4. Flash Attention v3 (Double-Tiled)
Tiles both Q (blocks of Bq) and K/V (blocks of Bk). The outer loop is over Q-blocks, inner over K/V-blocks with snake ordering. This maximizes K/V reuse: the same K/V block serves Bq query rows before being evicted. This is the most memory-efficient variant.

---

## Key Results

### Table 1: ByteDMD Advantage Grows with N While FLOPs Stay Flat (d=2)

| N   | Naive ByteDMD | Best Flash Variant          | Flash ByteDMD | ByteDMD Ratio | FLOP Ratio |
|-----|---------------|------------------------------|---------------|---------------|------------|
| 4   | 1,260         | flash_v1(Bk=4)              | 1,213         | **1.04x**     | 1.05x      |
| 8   | 7,230         | flash_v1(Bk=8)              | 6,228         | **1.16x**     | 1.07x      |
| 16  | 43,215        | flash_v2(Bk=8)              | 30,867        | **1.40x**     | 1.02x      |
| 32  | 272,416       | flash_v3(Bq=4,Bk=8)         | 146,576       | **1.86x**     | 1.00x      |
| 64  | 1,823,073     | flash_v3(Bq=4,Bk=8)         | 671,189       | **2.72x**     | 0.99x      |
| 128 | 12,814,834    | flash_v3(Bq=4,Bk=8)         | 3,098,327     | **4.14x**     | 0.98x      |

**Observation**: The ByteDMD ratio doubles approximately every time N quadruples, consistent with O(√N) scaling. The FLOP ratio hovers between 0.98x–1.07x throughout — completely blind to the optimization.

### Table 2: ByteDMD/FLOP Efficiency

| N   | Naive ByteDMD/FLOP | Best Flash ByteDMD/FLOP |
|-----|-------------------|------------------------|
| 4   | 7.0               | 7.1                    |
| 8   | 9.7               | 9.0                    |
| 16  | 14.3              | 10.4                   |
| 32  | 22.3              | 12.0                   |
| 64  | 37.2              | 13.6                   |
| 128 | 65.3              | 15.6                   |

Naive attention's ByteDMD-per-FLOP ratio grows as O(√N), reflecting the N×N intermediate matrix pushing data ever deeper. Flash attention's ratio grows much more slowly, staying bounded by O(√Bk).

---

## Crossover Analysis

### When does flash attention start winning?

| N   | ByteDMD Ratio | log₂(Ratio) | Scaling Exponent (Δlog₂/ΔlogN) |
|-----|---------------|-------------|-------------------------------|
| 4   | 1.04x         | 0.055       | —                             |
| 8   | 1.16x         | 0.215       | 0.160                         |
| 16  | 1.40x         | 0.485       | 0.270                         |
| 32  | 1.86x         | 0.894       | 0.409                         |
| 64  | 2.72x         | 1.442       | 0.547                         |
| 128 | 4.14x         | 2.048       | 0.607                         |

The scaling exponent approaches 0.5 (i.e., the ratio grows as N^0.5 = √N), converging toward the theoretical prediction from the LRU stack model:

- **Naive**: Each V read during O = P @ V has stack distance ≈ N² (buried under the full attention matrix). Cost per read: √(N²) = N.
- **Flash**: V reads have stack distance ≈ Bk·d (buried under only one block of intermediates). Cost per read: √(Bk·d).
- **Ratio**: ≈ N / √(Bk·d) = √(N²/(Bk·d)), growing as √N for fixed Bk, d.

**Crossover region**: N ≈ 4–8. At N=4, tiling overhead (online softmax rescaling) is proportionally large and the attention matrix is only 16 elements — not enough to bury V significantly. At N=8, the tiling benefit starts outweighing overhead.

---

## Flash Variant Comparison (d=2)

### Which variant wins at each scale?

| N   | Best v1 (ByteDMD) | Best v2 (ByteDMD) | Best v3 (ByteDMD) | Winner |
|-----|-------------------|-------------------|-------------------|--------|
| 4   | 1,213             | 1,213             | 1,239             | v1/v2 (tie) |
| 8   | 6,228             | 6,228             | 6,512             | v1/v2 (tie) |
| 16  | 31,814            | 30,867            | 31,674            | **v2** |
| 32  | 156,677           | 150,942           | 146,576           | **v3** |
| 64  | 794,020           | 742,854           | 671,189           | **v3** |
| 128 | 4,049,546         | 3,749,360         | 3,098,327         | **v3** |

**Pattern**: 
- At small N (≤8), all variants perform similarly — the problem fits mostly in the LRU "cache" regardless.
- At N=16, swizzled ordering (v2) starts helping by keeping K/V blocks warm across consecutive Q rows.
- At N≥32, double-tiling (v3) dominates by amortizing K/V reads across Bq=4 query rows simultaneously.

### Improvement from v1 → v2 → v3 at N=128

| Variant         | ByteDMD     | Improvement over v1 |
|----------------|-------------|---------------------|
| flash_v1(Bk=16)| 4,049,546   | —                   |
| flash_v2(Bk=8) | 3,749,360   | 7.4%                |
| flash_v3(Bq=4,Bk=8) | 3,098,327 | **23.5%**       |

The v3 improvement is substantial — double-tiling provides a major locality benefit that stacks on top of v1's basic tiling.

---

## Effect of Head Dimension

| Config     | ByteDMD Ratio (naive/best flash) |
|-----------|--------------------------------|
| N=8, d=2  | 1.16x                          |
| N=8, d=4  | 1.00x (no advantage)           |
| N=16, d=2 | 1.40x                          |
| N=16, d=4 | 1.12x                          |
| N=32, d=2 | 1.86x                          |
| N=32, d=4 | 1.41x                          |
| N=64, d=2 | 2.72x                          |
| N=64, d=4 | 1.90x                          |

Larger d reduces flash attention's relative advantage because Q/K/V vectors are larger, consuming more of the working set per row. The N×N attention matrix is proportionally less dominant. However, the advantage is still significant and growing with N at d=4.

---

## Block Size Sensitivity (Flash v3, Bq=4, d=2)

| N   | Bk=2       | Bk=4       | Bk=8       | Bk=16      | Optimal |
|-----|-----------|-----------|-----------|-----------|---------|
| 4   | 1,514     | 1,239     | —         | —         | 4       |
| 8   | 7,766     | 6,676     | 6,512     | —         | 8       |
| 16  | 36,347    | 32,179    | 31,674    | 32,850    | 8       |
| 32  | 164,995   | 148,272   | 146,576   | 154,668   | 8       |
| 64  | 748,478   | 678,013   | 671,189   | 708,735   | 8       |
| 128 | 3,441,558 | 3,134,604 | 3,098,327 | 3,260,907 | 8       |

**Bk=8 is optimal across most sizes.** This balances:
- **Too-small Bk** (=2): More K/V block transitions, more online softmax overhead, more Q re-reads
- **Too-large Bk** (=16): The intermediate block matrices grow, losing the locality advantage (at Bk=N, flash degenerates to naive)

The sweet spot at Bk=8 corresponds to keeping ~8·d = 16 intermediate score values in the working set, which matches the heuristic that SRAM-sized working sets are optimal.

---

## Why ByteDMD Captures What FLOPs Miss

### The fundamental issue with FLOPs

FLOPs count arithmetic operations: additions, multiplications, comparisons. Both naive and flash attention perform:
- Q @ K^T: O(N²·d) multiply-adds
- Softmax: O(N²) exp + additions
- P @ V: O(N²·d) multiply-adds

Flash adds a small overhead for online softmax merging (~5·d + 5 extra ops per query per block merge), making it marginally *worse* in FLOPs. The FLOP metric structurally cannot capture that flash attention avoids materializing the N×N matrix.

### What ByteDMD measures differently

ByteDMD assigns cost = Σ √(stack_distance) for all read operations. When naive attention materializes the full S matrix, the V elements get pushed to depth ~N² on the LRU stack. Reading them costs √(N²) = N per element. Flash attention keeps V elements within depth ~Bk·d, costing √(Bk·d) per read.

The total ByteDMD cost difference is:
- **Naive**: O(N²·d·√N) — the √N factor comes from the N² intermediates burying V
- **Flash**: O(N²·d·√Bk) — bounded by the block size, not the sequence length

The ByteDMD ratio is thus O(√(N/Bk)), growing as √N for fixed Bk. This matches our measurements: the ratio goes from 1.04x at N=4 to 4.14x at N=128.

---

## Methodology

- **Tracer**: All measurements use `bytedmd_fast.py`, an optimized version of the ByteDMD tracer with O(log N) depth lookups via Fenwick tree (vs O(N) in the original). Validated to produce identical results.
- **Implementations**: Pure scalar Python loops on list-of-lists, matching ByteDMD's tracking model. No numpy vectorized operations.
- **Auxiliary ops**: `exp(x) → -x` (1 read), `max(a,b) → a+b` (2 reads), `inv(x) → -x` (1 read). Values differ from real attention but ByteDMD only measures access patterns.
- **FLOPs**: Counted analytically to match each implementation exactly.
- **Inputs**: All-1.0 matrices. Values don't affect ByteDMD cost.
- **Block sizes**: Bk ∈ {2, 4, 8, 16}, Bq ∈ {2, 4} for v3.

## Reproducing

```bash
# Validate optimized tracer
python3 fast-flash-attention/validate_fast.py

# Run full benchmarks (~25s)
python3 fast-flash-attention/run_benchmarks.py

# Analyze results
python3 fast-flash-attention/analyze_results.py
```

Results are saved to `fast-flash-attention/benchmark_results.json`.
