# Research Report: Designing a Data Movement Cost Metric for 2D Silicon

**Period:** March 28 – April 12, 2026  
**Repository:** cybertronai/ByteDMD

## 1. Introduction

This report documents a two-week research effort to design a concrete, computable cost metric for data movement on 2D silicon, inspired by Bill Dally's ACM opinion piece arguing that FLOPs are an outdated measure of computational cost. The central idea: reading a byte at depth $d$ in an LRU stack costs $\lceil\sqrt{d}\rceil$, modeling the physical wire length on a 2D chip. The total cost of a computation is $C = \sum_{b \in \text{reads}} \lceil\sqrt{d(b)}\rceil$.

The work proceeded along three interleaved axes:

1. **Metric design:** How should the LRU stack behave? What happens to dead variables? How are cold misses priced? Each choice changes the asymptotic formulas.
2. **Closed-form analysis:** For each metric variant, derive exact or asymptotic ByteDMD formulas for canonical algorithms (matvec, naive matmul, recursive matmul, Strassen, flash attention).
3. **Implementation and validation:** Build Python tracers and Wolfram Language scripts that agree to the last integer, confirming the formulas.

The core finding is that **memory management strategy matters as much as algorithm choice** — the same algorithm can range from $\Theta(N^{3.5})$ to $\Theta(N^3 \log N)$ depending on whether dead variables are reclaimed. Furthermore, the choice between LRU compaction vs. stationary slots determines whether naive matmul measures as $O(N^{3.7})$ or a clean $O(N^4)$.

Below we group findings by the design axes explored, with links to the relevant Gemini research notes in `gemini/`.

---

## 2. Axis 1: Memory Management Strategies

The single biggest design decision is what happens when a variable dies (is never read again). We explored four strategies, each producing different asymptotic behaviors.

### 2.1 Strategy 1: No GC ("Infinite Graveyard")

**Description:** Dead temporaries remain on the LRU stack permanently. The stack grows without bound.

**Implementation:** Original `bytedmd.py` before April 9.

**Effect on formulas:**
- Dead intermediates bury live data progressively deeper
- Naive matmul: $\Theta(N^4)$ — B elements at depth $\sim 3N^2$; $N^3$ accesses at cost $\sqrt{3N^2} = \sqrt{3}\,N$ each → $\sqrt{3}\,N^4$
- Recursive matmul: degrades to $\Theta(N^{3.5})$ — at level $S$, $O(S^3)$ dead temporaries bury $S^2$ inputs at depth $S^3$; cost $S^2 \cdot \sqrt{S^3} = S^{3.5}$; geometric sum → $25.1\,N^{3.5}$
- Strassen: degrades to $\Theta(N^{3.4})$ — $O(N^{2.81})$ leaked intermediates; base matrices at depth $N^{2.81}$ → $74.7\,N^{3.405}$

**Pros:** Simplest to implement. No liveness analysis needed.  
**Cons:** Artificially inflates costs by polynomial factors. Makes recursive algorithms look worse than they are.

**References:** [infinite-graveyard.md](infinite-graveyard.md), [gc-vs-nogc-formulas.md](gc-vs-nogc-formulas.md), [tombstone-analysis.md](tombstone-analysis.md)

### 2.2 Strategy 2: Traditional GC (Tombstone Holes)

**Description:** Dead variables are replaced by tombstone markers that hold their slot. New allocations recycle the lowest tombstone. Stack size equals the high-water mark, not the live count.

**Implementation:** `bytedmd.py` with refcount GC (April 8–9).

**Effect on formulas:**
- Stack size capped at peak watermark, but holes remain
- Naive matmul: $\Theta(N^4)$ — unchanged (all of B is live throughout the $k$-loop)
- Recursive matmul: $\Theta(N^3 \log N)$ — dead branches evaporate; depth bounded by watermark $\sim 4.25\,S^2$; cost per level $12.3\,S^3$; accumulates over $\log N$ levels
- Strassen: $\Theta(N^3)$ — watermark $4S^2$; cost per level $25\,S^3$; geometric sum factor $= 8$ → $200\,N^3$

**Pros:** Recovers optimal asymptotics. No "free data movement."  
**Cons:** Tombstone recycling introduces a "teleportation" artifact where new values appear at artificially low depths. Leading constants are larger than Strategy 3.

**References:** [tombstone-analysis.md](tombstone-analysis.md), [gc-vs-nogc-formulas.md](gc-vs-nogc-formulas.md)

### 2.3 Strategy 3: Aggressive Compaction (LRU Stack Shrinking)

**Description:** Dead variables are immediately removed from the LRU stack. Remaining items slide up to fill the gap. Stack size always equals the live variable count.

**Implementation:** `bytedmd.py` after April 11 commit `ecedf37`. Uses two-pass liveness analysis: Pass 1 records last-use index for every variable; Pass 2 replays the trace, evicting variables the moment their last use completes.

**Effect on formulas:**
- Same asymptotics as Strategy 2, but smaller constants due to tighter stack
- Naive matmul: $\Theta(N^4)$ — same (B is fully live)
- Recursive matmul: $\Theta(N^3 \log N)$ — depth $\sim 1.5\,S^2$; cost per level $7.3\,S^3$ → $7.3\,N^3 \log_2 N$
- Strassen: $\Theta(N^3)$ — depth $2S^2$; cost per level $17.6\,S^3$ → $140.8\,N^3$

**Pros:** Tightest constants. Clean semantics (stack = working set).  
**Cons:** The "sliding up" is physically free data movement — in real silicon, removing a cache line doesn't move the others. This makes hot hits artificially cheap (see §2.4).

**References:** [gc-vs-nogc-formulas.md](gc-vs-nogc-formulas.md), [vecmat-11apr26.md](vecmat-11apr26.md), [mathematica-matmul.md](mathematica-matmul.md)

### 2.4 Strategy 4: Stationary Slots + Min-Heap Free-List

**Description:** SRAM is an array of numbered slots. Each variable gets a fixed slot when allocated (lowest available via min-heap). Dead variables' slots are returned to the heap. Access cost = $\sqrt{\text{slot number}}$. No data movement on death.

**Implementation:** `bytedmd_cold.py` after the fix from [bytedmd-cold-bug.md](bytedmd-cold-bug.md).

**Effect on formulas:**
- Naive matmul: clean $\Theta(N^4)$ — B occupies slots $1 \ldots N^2$; each access costs $\sqrt{N^2} = N$; total $N^3 \times N = N^4$. Measured `cost/N^4` converges to $\sim 4.0$.
- Hot hit depths are realistic: max hot depth = peak working set (512 for N=16), mean = 287, vs. LRU compaction's max 25, mean 8.5
- Removes the "free sliding" artifact of Strategy 3

**Pros:** Most physically realistic. No free data movement. Clean $O(N^4)$ for naive matmul.  
**Cons:** Greedy (min-heap) allocation is suboptimal — see §2.5. Belady's MIN also fails.

**References:** [bytedmd-cold-bug.md](bytedmd-cold-bug.md), [optimal-register-allocation.md](optimal-register-allocation.md)

### 2.5 Strategy 5: Optimal Register Allocation (Theoretical)

**Description:** A clairvoyant compiler assigns variable lifetimes to slots to minimize total $\sum \sqrt{\text{slot}}$ cost. Maps to the Minimum Sum Interval Coloring problem (NP-hard, Marx 2005).

**Key finding — "No Unsolicited Reads" proof:** Moving a variable proactively always increases cost, so optimal allocation fixes each variable at its initial slot.

**Key finding — Belady fails ("Squatter Anomaly"):** Long-lived variables (matrix corners, outer-loop accumulators) grab cheap slots at $t=0$ and "squat" there. Short-lived inner-loop temporaries are forced into expensive slots. Cost: $O(N \cdot \sqrt{K})$ instead of optimal $O(N + K\sqrt{K})$.

**Tractable approximation:** Iterative Maximum Weighted Independent Set via Earliest-Deadline-First (EDF) interval scheduling, $O(N \log N)$ per slot.

**References:** [optimal-register-allocation.md](optimal-register-allocation.md), [optimal-hierarchical-memory.md](optimal-hierarchical-memory.md)

### 2.6 Strategy 6: Inverted Stack Arenas (Managed Memory)

**Description:** Recursion depth dictates physical address placement. The deepest level (base cases) gets the lowest addresses (closest to CPU). Parent levels copy data down into child arenas; children operate in tight local footprints. Dead arenas are recycled by siblings.

**Implementation:** `experiments/optimal-memory/optimal_rmm.py` and `optimal_strassen.py`.

**Effect on formulas:**
- Recursive matmul: $\Theta(N^3 \log N)$ — $O(M^2)$ elements copied at distance $O(M)$ per level; recurrence $T(N) = 8T(N/2) + \Theta(N^3)$; `cost/(N^3 log N)` converges to $\sim 16$
- Strassen: $\Theta(N^3)$ — same arena structure but 7 branches; recurrence $T(N) = 7T(N/2) + \Theta(N^3)$; root-dominated → `cost/N^3` stabilizing at $\sim 85$

**Key insight:** Standard RMM's 8-way branching perfectly resonates with $2^3$ (the cube dimension), making every cache level contribute equally ($N^3$ per level × $\log N$ levels). Strassen's 7-way branching breaks this resonance: per-level cost decays geometrically by $7/8$, summing to a bounded constant.

**References:** [optimal-managed-rmm.md](optimal-managed-rmm.md), [optimal-managed-strassen.md](optimal-managed-strassen.md)

---

## 3. Axis 2: Cold Miss Pricing

Orthogonal to GC strategy is how to price the first access to an input element.

### 3.1 Eager Initialization (Stack-based)

**Description:** Arguments are pushed to the LRU stack on function entry. First reads are "free" (they're already at the top).

**Effect:** Small functions are artificially cheap. A function with 100 arguments paying $\sqrt{100} = 10$ per read even if only 2 are used.

### 3.2 Demand-Paged Initialization

**Description:** Arguments start in "DRAM." First access is a cold miss priced at `len(stack) + cold_counter`.

**Implementation:** `bytedmd.py` after April 10 commit `176000d`.

**Effect:** Unused arguments are free. Cold misses scale with current stack size.

### 3.3 Global Peak Working Set Pricing

**Description:** Cold misses are priced at `peak_working_set + global_cold_counter` on a monotonically increasing tape. Later cold misses are progressively more expensive.

**Implementation:** `bytedmd_cold.py` (before the stationary-slot fix).

**Effect:** Algorithms with bloated peak footprints pay more. Matvec N=2: 29 (vs. regular's 21). The cold tape never resets.

**References:** [tombstone-analysis.md](tombstone-analysis.md) §"Cold miss motivation"

---

## 4. Axis 3: Operand Pricing Semantics

### 4.1 Sequential Pricing

**Description:** Each operand in `a + b` is priced against the stack state *after* the previous operand was moved to top.

**Problem:** `b + c` ≠ `c + b` — violates commutativity. Column-major traversal (vecmat) is rewarded for cache thrashing.

### 4.2 Simultaneous Pricing

**Description:** All operands in a READ_BATCH are priced against the *same* pre-instruction snapshot.

**Implementation:** `bytedmd.py` after April 10 commit `176000d`.

**Effect:** `b + c` = `c + b`. Eliminates the artificial vecmat advantage.

**References:** [tombstone-analysis.md](tombstone-analysis.md) §"Sequential pricing breaks commutativity"

---

## 5. Master Formula Table

All formulas below assume $w=1$ byte per element and aggressive compaction unless noted.

### 5.1 Matrix-Vector Multiply ($y = Ax$, $N \times N$)

| Metric variant | Formula | N=4 |
|---|---|---|
| No GC (infinite graveyard) | $\frac{2\sqrt{3}}{3} N^3 + 2N^{2.5}$ | 177 |
| Aggressive compaction | $\frac{2}{3} N^3 + N^{2.5}$ | 124 |
| Demand-paged + aggressive | same | 124 |
| Cold (peak WS tape) | inflated cold misses | 164 |
| Stationary slots | $O(N^{2.5})$ | 178 |

**References:** [matrix-vector.md](matrix-vector.md), [matvec-exact-formula.md](matvec-exact-formula.md), [vecmat-11apr26.md](vecmat-11apr26.md), [vecmat-11apr26-part2.md](vecmat-11apr26-part2.md)

### 5.2 Naive Matrix Multiply ($C = AB$, i-j-k loop)

| Metric variant | Asymptotic | Leading constant | N=4 |
|---|---|---|---|
| No GC | $\sqrt{3}\,N^4$ | 1.732 | 948 |
| Aggressive compaction | $N^4 + \sqrt{2}\,N^{3.5}$ | ~1.0 | 676 |
| Snake order (no GC) | $\frac{2\sqrt{5}}{3}\,N^4$ | 1.491 | 906 |
| Stationary slots | $\sim 4.0\,N^4$ | ~4.0 | 1035 |

**References:** [matmul-naive-snake-chatgpt.md](matmul-naive-snake-chatgpt.md), [matmul-naive-vanilla.md](matmul-naive-vanilla.md), [matmul-analytic.md](matmul-analytic.md), [bytedmd-naive-formulas.md](bytedmd-naive-formulas.md), [bytedmd-cold-bug.md](bytedmd-cold-bug.md)

### 5.3 Recursive Matrix Multiply (8-way D&C)

| Metric variant | Asymptotic | Constant |
|---|---|---|
| No GC | $25.1\,N^{3.5}$ | 25.1 |
| Traditional GC | $12.3\,N^3 \log_2 N$ | 12.3 |
| Aggressive compaction | $7.3\,N^3 \log_2 N$ | 7.3 |
| Gray-code swizzled | $5.20\,N^3 \log_2 N$ | 5.20 |
| Inverted arena | $\sim 16\,N^3 \log_2 N$ | ~16 |

**Lower bound** (Hong-Kung): $\Omega(N^3 \log N)$ for any $O(N^3)$-FLOP algorithm.

**References:** [matmul-snake.md](matmul-snake.md), [gc-vs-nogc-formulas.md](gc-vs-nogc-formulas.md), [vanilla-recursive-11apr26.md](vanilla-recursive-11apr26.md), [vanilla-recursive2-11apr26.md](vanilla-recursive2-11apr26.md), [optimal-managed-rmm.md](optimal-managed-rmm.md)

### 5.4 Strassen (7-way D&C)

| Metric variant | Asymptotic | Constant |
|---|---|---|
| No GC | $74.7\,N^{3.405}$ | 74.7 |
| Traditional GC | $200\,N^3$ | 200 |
| Aggressive compaction | $140.8\,N^3$ | 140.8 |
| Strassen-Winograd | $107.7\,N^3$ | 107.7 |
| Inverted arena | $\sim 85\,N^3$ | ~85 |

**Lower bound:** $\Omega(N^3)$ from cold-miss initialization: $\sum_{d=1}^{O(N^2)} \sqrt{d} = \Theta(N^3)$.

**Key insight:** Strassen annihilates the $\log N$ penalty. Its arithmetic complexity $O(N^{2.81})$ is masked by the $O(N^3)$ physical routing floor, but it still beats standard matmul asymptotically.

**References:** [bytedmd-strassen.md](bytedmd-strassen.md), [strassen-11apr26.md](strassen-11apr26.md), [strassen-discrepancy.md](strassen-discrepancy.md), [optimal-managed-strassen.md](optimal-managed-strassen.md)

### 5.5 Flash Attention

| Variant | ByteDMD | Ratio vs. naive |
|---|---|---|
| Naive attention, N=128, d=2 | 13,705,802 | 1.00x |
| Flash attention, N=128, d=2, Bk=8 | 4,221,808 | **3.25x** |

FLOPs are nearly identical (0.96x ratio), but ByteDMD diverges to 3.25x at N=128. Flash attention's advantage comes from keeping the working set small (no $N \times N$ score matrix materialized), which ByteDMD captures but FLOPs cannot.

**Caveat:** When L=D=K=N and block sizes scale with N (Br=Bc=N/2), flash attention shows **no benefit** because the projections dominate and blocks exceed SRAM capacity.

**References:** [flash-attention-gc-vs-nogc.md](flash-attention-gc-vs-nogc.md), [noam-flash-attention.md](noam-flash-attention.md), [flash-attention-no-benefit.md](flash-attention-no-benefit.md)

---

## 6. Chronological Development Log

### Phase 1: Initial Metric and Formulas (Apr 4–7)

- Built flash attention benchmark showing 2.38x (N=64) to 3.25x (N=128) advantage under ByteDMD
- Derived first closed-form formulas for naive matmul: $\sqrt{3}\,N^4$ (standard) vs. $\frac{2\sqrt{5}}{3}\,N^4$ (snake)
- Computed exact matvec/vecmat costs (194/191 for N=4)
- Identified that Strassen/Winograd score *worse* than naive at N=4 due to intermediate matrix thrashing

### Phase 2: Memory Management Exploration (Apr 8–10)

- Implemented three memory strategies in `experiments/memory_management/`
- Discovered the "Infinite Graveyard" problem: dead temporaries degrade recursive matmul from $N^3 \log N$ to $N^{3.5}$
- Identified four tracer bugs: sequential pricing (non-commutative), argument order bias, zero-accumulator penalty, dead variable leaking
- Implemented fixes: simultaneous pricing, demand-paged initialization, natural LRU aging
- Derived the tombstone analysis: tombstones correctly preserve the $2N^{2.5}$ L1 thrashing penalty while preventing the "software teleportation" cheat of aggressive compaction

### Phase 3: Aggressive Compaction and Liveness Analysis (Apr 11)

- Switched `bytedmd.py` to two-pass aggressive liveness compaction
- Derived exact closed-form for matvec under compaction: $\text{Cost}(N) = N \cdot F(2N) + F(2N-1) - (N-2) \cdot F(N+2) + \ldots$; scaling $O(N^{2.5})$
- Derived exact analytical formula for naive matmul: $C(N) = N^4 + \sqrt{2}\,N^{3.5} + \Theta(N^3)$
- Built Wolfram Language scripts matching Python tracer exactly for all algorithms
- Fixed Strassen C22 associativity mismatch (left-associative saves reads by accessing M2 while hot)

### Phase 4: Cold Miss Pricing and Stationary Slots (Apr 12)

- Created `bytedmd_cold.py` with global peak working set cold-miss pricing
- Discovered the "free sliding" bug: LRU compaction makes hot hits artificially cheap; naive matmul scales as $\sim N^{3.7}$ instead of clean $N^4$
- Fixed by replacing LRU stack with stationary slots + min-heap free-list
- After fix: `cost/N^4` converges to $\sim 4.0$; effective exponent is 4.0 at N=8→16

### Phase 5: Optimal Arena Allocation (Apr 12)

- Implemented inverted stack arena for recursive matmul: base cases at lowest addresses
- Verified $O(N^3 \log N)$ scaling: `cost/(N^3 \log N)` converges to ~16
- Implemented arena-based Strassen: verified $O(N^3)$ scaling; `cost/N^3` stabilizing
- At N=64, Strassen arena cost (25M) approaches standard RMM arena cost (24M); crossover imminent
- Explored optimal register allocation theory: problem is NP-hard; Belady fails due to "Squatter Anomaly"; EDF approximation is tractable

---

## 7. Design Decision Summary

| Design Choice | Options Explored | Recommendation |
|---|---|---|
| **Dead variable handling** | No GC, tombstones, compaction, stationary slots | Stationary slots (most physical) |
| **Cold miss pricing** | Eager init, demand-paged, peak WS tape | Demand-paged + monotone tape |
| **Operand pricing** | Sequential, simultaneous | Simultaneous (preserves commutativity) |
| **Slot allocation** | Min-heap greedy, Belady, optimal (NP-hard) | Min-heap greedy (tractable, correct asymptotics) |
| **Arena layout** | Flat, inverted stack | Inverted stack (for managed memory experiments) |

---

## 8. Open Questions

1. **Optimal allocation tractability:** The minimum-sum interval coloring is NP-hard. Is the greedy min-heap approximation tight enough that it preserves asymptotic complexity classes? The EDF approximation from [optimal-register-allocation.md](optimal-register-allocation.md) suggests yes.

2. **Strassen crossover:** Under the inverted arena model, Strassen (O(N^3)) doesn't beat standard RMM (O(N^3 log N)) until very large N due to the constant factor (~85 vs ~16). Where exactly is the crossover?

3. **Competition metric:** For a practical competition metric, which strategy strikes the best balance between physical realism and analytical tractability? Stationary slots are the most honest but require liveness analysis and heap management. Aggressive compaction gives cleaner formulas but includes free data movement.

4. **Multi-byte elements:** All formulas above use $w=1$. The `bytes_per_element` scaling uses the closed-form $U(m) = M(6m - 2M^2 + 3M - 1)/6$ where $M = \lceil\sqrt{m}\rceil$. How do the crossover points shift for realistic 4-byte or 8-byte elements?
