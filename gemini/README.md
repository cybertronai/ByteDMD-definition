# Gemini Research Reports — Chronological Index

Research notes on **ByteDMD** (Byte-level Data Movement Distance): a concrete metric for data movement cost where reading address $d$ costs $\lceil\sqrt{d}\rceil$, modeling 2D VLSI wire-length routing on a spatial chip.

---

## April 7, 2026

| Report | Summary |
|--------|---------|
| [matmul-snake.md](matmul-snake.md) | Closed-form ByteDMD solutions for naive matmul with normal and snake loop ordering |

## April 8, 2026

| Report | Summary |
|--------|---------|
| [rmm.md](rmm.md) | Evaluation of recursive vs stressed matmul methods and suggestions for ByteDMD improvements |
| [bytedmd-instruction-list.md](bytedmd-instruction-list.md) | 24 canonical instruction families and 38 methods with I/O counts for ByteDMD tracer |
| [bytedmd-strassen.md](bytedmd-strassen.md) | Asymptotic analysis comparing ByteDMD costs: Naive, Tiled, Recursive, Strassen vs FLOP costs |
| [cache-eviction-visualization.md](cache-eviction-visualization.md) | Brainstorming cache eviction visualizations with concentric circles and Manhattan geometry |
| [matmul-just-snake.md](matmul-just-snake.md) | Exact ByteDMD cost for naive matmul using snake order: 906 for 4x4, asymptotic O(N^4) |
| [matmul-naive-snake-chatgpt.md](matmul-naive-snake-chatgpt.md) | Closed-form formulas for ByteDMD naive matmul: discrete and continuous bounds |
| [matmul-naive-vanilla.md](matmul-naive-vanilla.md) | ByteDMD exact formulas and tight bounds for naive matrix multiplication algorithms |
| [matmul-regular-snake.md](matmul-regular-snake.md) | Closed-form ByteDMD solutions for matrix multiplication under normal and snake order |
| [matrix-vector.md](matrix-vector.md) | Exact ByteDMD formulas for NxN matrix-vector multiplication with tight asymptotic bounds |
| [matvec+vecmat.md](matvec+vecmat.md) | Matvec and vecmat ByteDMD formulas: vecmat 8% cheaper via pseudo-eviction buffering |

## April 10, 2026

| Report | Summary |
|--------|---------|
| [bytedmd-naive-formulas.md](bytedmd-naive-formulas.md) | Exact ByteDMD formulas separating theoretical model from Python proxy implementation discrepancies |
| [matmul-optimal.md](matmul-optimal.md) | Energy lower bounds and optimal spatial computer designs for NxN matrix-vector multiplication |
| [matvec-exact-formula.md](matvec-exact-formula.md) | Exact matvec/vecmat ByteDMD cost identity under demand-paged liveness analysis |
| [matvec-vecmat.md](matvec-vecmat.md) | Perfect symmetry: matvec and vecmat have identical ByteDMD costs with liveness analysis |
| [renaming-brainstorming.md](renaming-brainstorming.md) | Alternative names for ByteDMD metric emphasizing concrete execution vs asymptotic bounds |
| [tombstone-analysis.md](tombstone-analysis.md) | Exact ByteDMD formulas with liveness analysis for matvec/vecmat from trace simulation |

## April 11, 2026

| Report | Summary |
|--------|---------|
| [at2-lower-bounds.md](at2-lower-bounds.md) | VLSI Area-Time complexity analysis of matmul algorithms under spatial computer model |
| [flash-attention-gc-vs-nogc.md](flash-attention-gc-vs-nogc.md) | Liveness analysis impact on Flash Attention: GC vs no-GC asymptotic complexity bounds |
| [gc-vs-nogc-formulas.md](gc-vs-nogc-formulas.md) | LRU stack physics: how garbage collection changes matmul ByteDMD from O(N^4) to O(N^4) |
| [infinite-graveyard.md](infinite-graveyard.md) | ByteDMD "Infinite Graveyard" model where dead temporaries sink permanently down stack |
| [mathematica-matmul.md](mathematica-matmul.md) | Wolfram Mathematica code for precise ByteDMD trace of Strassen algorithm with liveness |
| [matmul-complexity.md](matmul-complexity.md) | Comprehensive ByteDMD complexity analysis: naive O(N^4), tiled O(N^3.5), recursive O(N^3 log N) |
| [matmul-lower-bound.md](matmul-lower-bound.md) | Absolute lower bound for ByteDMD matmul: O(N^3) only achievable by Strassen with liveness |
| [noam-flash-attention.md](noam-flash-attention.md) | Flash Attention 2 implementation using Noam Shazeer's shape-suffix dimension conventions |
| [strassen-11apr26.md](strassen-11apr26.md) | Precise ByteDMD trace cost of Strassen via Wolfram Mathematica with liveness analysis |
| [vanilla-recursive-11apr26.md](vanilla-recursive-11apr26.md) | ByteDMD of vanilla recursive divide-and-conquer matmul via Wolfram Mathematica |
| [vanilla-recursive2-11apr26.md](vanilla-recursive2-11apr26.md) | Cache-oblivious recursive matmul restricts working sets to O(N^3 log N) ByteDMD cost |
| [vecmat-11apr26.md](vecmat-11apr26.md) | Matvec/vecmat identical ByteDMD costs under liveness: active variable cycling proves symmetry |

## April 12, 2026

| Report | Summary |
|--------|---------|
| [week15recap.md](week15recap.md) | Two-week research recap on ByteDMD metric for 2D silicon: design, formulas, and implementation |

## April 13, 2026

| Report | Summary |
|--------|---------|
| [bytedmd-cold-bug.md](bytedmd-cold-bug.md) | O(N^3) cold-miss formula vs O(N^4) total -- static slots avoid free compaction cost |
| [flash-attention-no-benefit.md](flash-attention-no-benefit.md) | Flash Attention shows no benefit when sequence length and head dimension scale equally |
| [gemini-merged.md](gemini-merged.md) | Chronologically merged Gemini notes on ByteDMD formulas and algorithm analysis |
| [matmul-analytic.md](matmul-analytic.md) | Iterative matmul LRU thrashing comparison: recursive O(N^3 log N) vs standard O(N^4) |
| [matmul-edf.md](matmul-edf.md) | EDF register allocation mirrors Belady's optimal via MWIS, traps O(N^4) squatter anomaly |
| [optimal-hierarchical-memory.md](optimal-hierarchical-memory.md) | Hierarchical scratchpad with arena allocators achieves O(N^3 log N) for recursive matmul |
| [optimal-managed-rmm.md](optimal-managed-rmm.md) | Software DMA with arena allocation strategy for O(N^3 log N) recursive matmul |
| [optimal-managed-strassen.md](optimal-managed-strassen.md) | Strassen O(N^3) vs recursive matmul O(N^3 log N): Strassen absorbs data movement overhead |
| [optimal-register-allocation.md](optimal-register-allocation.md) | Belady's algorithm fails in continuous spatial model; NP-hard Minimum Interval Coloring |
| [strassen-discrepancy.md](strassen-discrepancy.md) | Associativity ordering differences between Mathematica and Python affect ByteDMD cost |
| [vecmat-11apr26-part2.md](vecmat-11apr26-part2.md) | Matvec/vecmat identity: simultaneous pricing makes stack behavior commutative |

## April 15, 2026

| Report | Summary |
|--------|---------|
| [15apr26-dmdlive-analysis.md](15apr26-dmdlive-analysis.md) | DMD vs DMDlive are extremes; Tombstone/Ripple-Shift achieves physically realistic bounds |

## April 16, 2026

| Report | Summary |
|--------|---------|
| [ripple-shift.md](ripple-shift.md) | Ripple Shift cascaded eviction avoids tombstone inflation bug, maintains live footprint |
| [bytedmd_explicit_rmm.md](bytedmd_explicit_rmm.md) | Hierarchical scratchpad allocator proves O(N^3 log N) achievable with deterministic address packing |
| [mwis-bounds.md](mwis-bounds.md) | Linear programming relaxation for optimal register allocation in continuous cache model |
| [unimodular-lp.md](unimodular-lp.md) | Discrete calculus converts nonlinear sqrt cost function to linear program for optimal allocation |
| [unimodular-optimized-impl.md](unimodular-optimized-impl.md) | Profile-guided oracle allocation approaches totally unimodular LP continuous cache bound |

## April 17, 2026

| Report | Summary |
|--------|---------|
| [livedmd-not-a-bound.md](livedmd-not-a-bound.md) | ByteDMDlive is not a lower bound; fixed-address allocators can beat LRU sliding behavior |
| [matmul-explicit-formula.md](matmul-explicit-formula.md) | Exact energy formula E = (20/3+sqrt3) N^3 log2 N for hierarchical RMM, validated empirically |
| [strassen-manual.md](strassen-manual.md) | Strassen with manual memory placement and stack arena semantics, 3N^2 peak memory |
| [strassen-vs-systolic.md](strassen-vs-systolic.md) | Single-processor Strassen and systolic arrays both hit O(N^3) energy floor on 2D silicon |
| [efficient-strassen.md](efficient-strassen.md) | Zero-Allocation Fused Strassen eliminates temporaries via virtual matrices in L1 scratchpad |
| [better-heuristics.md](better-heuristics.md) | Five hardware-agnostic cache-friendliness heuristics: stack distance, arithmetic intensity, liveness, byte-ticks, Belady AUC |
| [simplified-manual-rmm.md](simplified-manual-rmm.md) | Pyramid-and-Forklift RMM walkthrough; connection to HMM (1987) and Hong-Kung; 3 ways to beat the 2D bound |
| [rmm-logn-simplified.md](rmm-logn-simplified.md) | Cleaned-up Pyramid/Forklift RMM code; Master-Theorem proof of Θ(N³ log N) from T(K) = 8T(K/2) + Θ(K³) recurrence |
| [bytedmd-logn-proof.md](bytedmd-logn-proof.md) | Two proofs that RMM under bytedmd_live is Θ(N³ log N): Master Theorem and a Hong–Kung cache-miss integral identity |
| [flops-are-a-lie.md](flops-are-a-lie.md) | Six cross-CS examples where FLOP counts mislead: Fibonacci heaps, sparse matmul, Winograd vs Im2Col, gradient checkpointing, LUTs, quicksort vs heapsort, Flash vs Sparse attention |
