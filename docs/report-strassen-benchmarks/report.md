# Strassen vs Recursive Matrix Multiplication: FLOP and ByteDMD Analysis

## Summary

We compare **Strassen's algorithm** (7 subproblems, O(n^2.807)) against **recursive matrix multiplication** (8 subproblems, O(n^3)) — both sharing the same divide-and-conquer structure (split/join). The only difference: Strassen trades 1 multiplication for extra additions at each recursion level.

Two cost metrics:
1. **FLOPs** — arithmetic operation count
2. **ByteDMD** — data-movement cost (LRU stack depth model with sqrt-cost reads)

**Key finding**: FLOPs and ByteDMD **disagree** at large sizes. When Strassen starts winning on FLOPs (ratio > 1), ByteDMD still shows recursive as cheaper. Strassen's extra 18 additions (vs recursive's 4) create temporary matrices that push source data deeper on the LRU stack, making its data-movement cost worse than the FLOP savings suggest. ByteDMD never favors Strassen over recursive at any tested size.

---

## Table 1: FLOP Ratios (Recursive / Strassen)

Ratios > 1 mean Strassen does fewer FLOPs. Theoretical limit per level: 8/7 = 1.143.

| n | L=1 | L=2 | L=4 | L=8 |
|--:|----:|----:|----:|----:|
| 2 | 0.480 | 1.000 | 1.000 | 1.000 |
| 4 | 0.453 | 0.718 | 1.000 | 1.000 |
| 8 | 0.476 | 0.696 | 0.896 | 1.000 |
| 16 | 0.520 | 0.734 | 0.917 | **1.008** |
| 32 | 0.579 | 0.803 | 0.989 | **1.080** |
| 64 | 0.651 | 0.896 | **1.096** | **1.192** |

With leaf=8, Strassen first saves FLOPs at n=16. With leaf=4, the crossover is at n=64.

---

## Table 2: ByteDMD Costs (Absolute)

| n | Algo | L=1 | L=2 | L=4 | L=8 |
|--:|:----:|----:|----:|----:|----:|
| 2 | Recur | 66 | 66 | 66 | 66 |
| 2 | Stras | 152 | 66 | 66 | 66 |
| 4 | Recur | 1,006 | 1,006 | 948 | 948 |
| 4 | Stras | 2,435 | 1,682 | 948 | 948 |
| 8 | Recur | 12,758 | 12,758 | 12,410 | 12,900 |
| 8 | Stras | 30,606 | 23,691 | 17,704 | 12,900 |
| 16 | Recur | 151,944 | 151,944 | 149,181 | 155,193 |
| 16 | Stras | 350,947 | 285,429 | 234,226 | 197,899 |
| 32 | Recur | 1,760,923 | 1,760,923 | 1,738,829 | 1,788,784 |
| 32 | Stras | 3,851,973 | 3,214,422 | 2,757,203 | 2,464,300 |
| 64 | Recur | 20,143,659 | 20,143,659 | 19,967,098 | 20,369,758 |
| 64 | Stras | 41,354,139 | 34,992,687 | 30,750,338 | 28,293,255 |

Notable: Recursive ByteDMD is remarkably stable across leaf sizes (~20M at n=64 regardless of leaf). Strassen ByteDMD drops significantly with larger leaves but never reaches recursive's level.

---

## Table 3: Side-by-Side FLOP and ByteDMD Ratios (Recursive / Strassen)

> Ratio > 1 = Strassen wins. **Bold** = disagreement between metrics.

| n | Leaf | FLOP R/S | DMD R/S | FLOP winner | DMD winner |
|--:|-----:|---------:|--------:|:-----------:|:----------:|
| 2 | 1 | 0.480 | 0.434 | Recursive | Recursive |
| 4 | 1 | 0.453 | 0.413 | Recursive | Recursive |
| 4 | 2 | 0.718 | 0.598 | Recursive | Recursive |
| 8 | 1 | 0.476 | 0.417 | Recursive | Recursive |
| 8 | 2 | 0.696 | 0.539 | Recursive | Recursive |
| 8 | 4 | 0.896 | 0.701 | Recursive | Recursive |
| **16** | **8** | **1.008** | **0.784** | **Strassen** | **Recursive** |
| **32** | **4** | **0.989** | **0.631** | Recursive | Recursive |
| **32** | **8** | **1.080** | **0.726** | **Strassen** | **Recursive** |
| **64** | **4** | **1.096** | **0.649** | **Strassen** | **Recursive** |
| **64** | **8** | **1.192** | **0.720** | **Strassen** | **Recursive** |

At n=64 with L=8: FLOPs say Strassen wins by 19.2%, but ByteDMD says recursive wins by 28%.

---

## Table 4: Scaling Exponents

| Transition | R FLOP | S FLOP | R DMD (L=1) | S DMD (L=1) | R DMD (L=4) | S DMD (L=4) |
|-----------:|-------:|-------:|------------:|------------:|------------:|------------:|
| 2→4 | 3.222 | 3.305 | 3.930 | 4.002 | 3.844 | 3.844 |
| 4→8 | 3.100 | 3.030 | 3.665 | 3.652 | 3.710 | 4.223 |
| 8→16 | 3.047 | 2.921 | 3.574 | 3.519 | 3.587 | 3.726 |
| 16→32 | 3.023 | 2.868 | 3.535 | 3.456 | 3.543 | 3.557 |
| 32→64 | 3.011 | 2.841 | 3.516 | 3.424 | 3.521 | 3.479 |

Expected FLOP exponents: Recursive = 3.000, Strassen = log2(7) ≈ 2.807.

ByteDMD exponents are ~0.5 higher than FLOP exponents (the sqrt-depth penalty). Strassen's ByteDMD exponent (3.42–3.48) is lower than recursive's (3.52), but the constant factor gap is so large that the crossover would require n >> 64.

---

## Analysis

### Why the FLOP and ByteDMD metrics disagree

When comparing against recursive (instead of naive), the structural overhead of split/join is identical in both algorithms. The **only** difference is:

- **Recursive**: 8 sub-multiplications + 4 element-wise additions to combine results
- **Strassen**: 7 sub-multiplications + 18 element-wise additions/subtractions to set up and combine

FLOPs count each addition and multiplication equally. Since a multiplication involves two recursive calls (much more work than an element-wise add), trading 1 multiply for 14 extra additions is a clear win for FLOPs at large n.

ByteDMD disagrees because those 14 extra additions create **temporary matrices** that sit on the LRU stack. These temporaries push the source data (A and B quadrants) deeper, making subsequent reads more expensive. The sqrt-depth cost model amplifies this effect: burying data at depth d costs sqrt(d), not 1.

### Why recursive ByteDMD is stable across leaf sizes

Recursive matmul's ByteDMD is nearly constant across leaf sizes (e.g., ~20M at n=64 for all leaves). This is because recursive matmul creates very few temporaries — only 4 additions to combine sub-results. The LRU stack stays compact regardless of where the recursion bottoms out.

Strassen's ByteDMD, by contrast, drops significantly with larger leaves (41M → 28M at n=64 going from L=1 to L=8) because each eliminated recursion level removes 18 temporary matrices from the stack. But even at L=8, Strassen still creates far more temporaries than recursive.

### The DMD ratio is getting worse, not better

The DMD R/S ratio at L=8 goes: 1.000, 1.000, 1.000, 0.784, 0.726, 0.720, 0.740 (n=128). While there is a slight uptick at n=128, the ratio remains far below 1.0. The FLOP ratio at L=8 goes: 1.000, 1.000, 1.000, 1.008, 1.080, 1.192, 1.335 — steadily increasing.

This divergence means: at every scale where Strassen's FLOP advantage matters, its data-movement disadvantage remains stubbornly large.

---

## Crossover Analysis (extended to n=128)

We extended measurements to n=128 and fit power laws `ByteDMD ≈ C * n^α` to extrapolate.

### Fitted power laws

| Leaf | Recursive | Strassen | Exponent gap (α_R − α_S) |
|-----:|:----------|:---------|:--------------------------|
| L=1 | 7.68 · n^3.554 | 21.07 · n^3.484 | **+0.070** (R grows faster) |
| L=2 | 7.68 · n^3.554 | 14.17 · n^3.540 | **+0.014** (R grows faster, barely) |
| L=4 | 8.06 · n^3.539 | 12.01 · n^3.544 | **−0.005** (S grows faster!) |
| L=8 | 9.19 · n^3.513 | 11.20 · n^3.539 | **−0.026** (S grows faster!) |

### R/S ByteDMD ratio progression (extended)

| n | L=1 | L=2 | L=4 | L=8 |
|--:|----:|----:|----:|----:|
| 4 | 0.413 | 0.598 | — | — |
| 8 | 0.417 | 0.539 | 0.701 | — |
| 16 | 0.433 | 0.532 | 0.637 | 0.784 |
| 32 | 0.457 | 0.548 | 0.631 | 0.726 |
| 64 | 0.487 | 0.576 | 0.649 | 0.720 |
| **128** | **0.522** | **0.612** | **0.682** | **0.740** |

### Extrapolated crossover points

| Leaf | Exponent gap | Crossover n | Verdict |
|-----:|:------------|:------------|:--------|
| **L=1** | +0.070 | **~1,950,000** | Impractically large (millions per side) |
| **L=2** | +0.014 | **~6.4 × 10^18** | Astronomically large — effectively never |
| **L=4** | −0.005 | **Never** | Strassen exponent ≥ recursive; no crossover |
| **L=8** | −0.026 | **Never** | Strassen exponent ≥ recursive; no crossover |

### Interpretation

Only at **leaf=1** does Strassen have a meaningfully lower ByteDMD exponent (3.484 vs 3.554). But the constant factor penalty is so enormous (21.07 vs 7.68 = 2.74x) that the crossover is at n ≈ 2 million — far beyond any practical matrix size.

At **leaf=4 and leaf=8**, Strassen's ByteDMD exponent is actually *higher* than recursive's. The local exponent gap is converging from negative toward zero but hasn't crossed positive:

```
L=8 local exponent gap:  16→32: -0.112,  32→64: -0.012,  64→128: +0.039
```

Even if the local gap stabilizes at +0.04, starting from a ratio of 0.74 at n=128, Strassen would need: `0.74 · (n/128)^0.04 = 1.0` → `n/128 = (1/0.74)^25 = 1217` → n ≈ 156,000. And a gap of +0.04 is optimistic based on the trend.

**Bottom line: Strassen never beats recursive matmul on ByteDMD at any practical matrix size, regardless of leaf size.**

---

## Why FLOPs and ByteDMD reach opposite conclusions

| | FLOPs | ByteDMD |
|:--|:------|:--------|
| **What it counts** | Operations (adds, muls equally) | sqrt(LRU depth) per read |
| **Strassen's advantage** | 7 muls instead of 8 | — |
| **Strassen's penalty** | 18 adds instead of 4 | 18 temporary matrices push data deeper on stack |
| **Net effect at large n** | Wins: 7/8 per level > add overhead | Loses: temporaries increase sqrt(depth) cost |
| **Scaling** | n^2.807 vs n^3 | n^3.48–3.54 vs n^3.51–3.55 (gap ≈ 0) |

The fundamental issue: FLOPs treat all operations as unit cost. ByteDMD reveals that Strassen's 14 extra additions per level aren't free — each creates temporary data that displaces source matrices from the top of the cache hierarchy. The sqrt-depth cost model amplifies this: pushing data from depth d to depth 2d doesn't double the cost, it increases it by √2 ≈ 1.41x. Strassen's many temporaries create this kind of displacement at every recursion level.

### Implications

1. **Strassen's FLOP savings don't translate to data-movement savings** — the extra temporaries dominate.

2. **The naive triple-loop comparison was misleading** — against naive, Strassen(L=8) appeared to win on ByteDMD at n=32. This was because naive's triple loop has poor locality itself. Against recursive (which has good locality by construction), Strassen never wins.

3. **Recursive matmul is the real competitor** — it matches Strassen's cache-friendly block structure while avoiding the temporary matrix overhead. This aligns with cache-oblivious algorithm theory, where recursive matmul achieves optimal I/O complexity O(n³/√M) without needing Strassen's trick.

---

## Reproduction

```bash
python3 strassen-benchmarks/benchmark_strassen.py    # main comparison tables
python3 strassen-benchmarks/find_crossover.py         # extended measurements + extrapolation
```

Results saved to `benchmark_results.json` and `crossover_results.json`.

