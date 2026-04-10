# matmul-regular-snake

**Source:** https://gemini.google.com/share/89111fbb7879
**Created with:** Gemini 3 Deep Think, April 7, 2026
**Topic:** Closed-form ByteDMD solutions for matrix multiplication under normal and snake order, recursive matmul, garbage collection effects

---

## Prompt 1
Closed-form solution for ByteDMD for matrix multiplication under normal and snake order, and for recursive matrix multiplication (without reuse).

### 1. Normal Order (i-j-k) Continuous Cost
Between reads of B_{k,j} in row i-1 and row i, the loop sweeps N^2 steps, pushing N^2 unique B elements plus 2N^2 dead temporaries. Stack depth stabilizes at d ~ 3N^2.

**C_normal(N) ~ sqrt(3) * N^4**

### 2. Snake Order Continuous Cost
Reversing j-loop on alternating rows creates a triangular depth profile: d ~ 5Nx, where x is distance from the turn-around point.

**C_snake(N) ~ (sqrt(5)/(3*sqrt(2))) * N^4**

Snake-ordering drops the routing constant from ~1.732 to ~1.491, a strict **~14% reduction**.

### 3. Recursive Matmul (Without Reuse) — Exact Counts
| N   | Exact ByteDMD Cost |
|-----|--------------------|
| 1   | 3                  |
| 2   | 66                 |
| 4   | 1,006              |
| 8   | 12,758             |
| 16  | 151,944            |
| 32  | 1,760,923          |

Recursive Morton/Z-order traversal bounds cache distances to O(N), so the recursive layout scales at O(N^3.5), avoiding catastrophic N^4 penalties.

---

## Prompt 2
Exact formulas for ByteDMD of naive matmul, tight bounds, discrepancies.

### Closed-Form Continuous Solutions

**Normal Order:** C_normal(N) ~ sqrt(3) * N^4 + alpha * N^3.5 + 3N^3, where alpha = 2 + (2*sqrt(2))/3 ~ 2.943

**Snake Order:** C_snake(N) ~ (sqrt(5)/(3*sqrt(2))) * N^4 + alpha * N^3.5 + 3N^3

### Algebraic Bounds
- Lower: L(N) = sqrt(3)*N^4 + alpha*N^3.5 - 2N^3
- Upper: U(N) = sqrt(3)*N^4 + alpha*N^3.5 + 5N^3

For N=4: 692 <= 948 <= 1140.

### Discrepancies
1. **AST Left-to-Right Evaluation:** Python evaluates A*B first, pushing to stack before s is priced. Generates phantom 2N^3 cost.
2. **Memory Leak (No GC):** _Context.allocate() only appends, never pops. 2N^3 dead temporaries permanently buried in LRU stack.
3. **Untracked Array Indexing:** List traversal (__getitem__) bypasses metric — zero-cost memory teleportation.
4. **Sequential Unrolling:** s = A[i][0]*B[0][j] initialization dodges N^2 phantom zero reads.

---

## Prompt 3
Does memory management matter for naive matmul?

**Yes, massively.** Adding garbage collection drops the routing constant:
- Without deletion: depth = 3N^2, cost ~ sqrt(3) * N^4 (routing constant 1.732)
- With deletion: depth = N^2, cost ~ 1.0 * N^4 (routing constant 1.0)

GC instantly drops data movement penalty by **~42%**.

---

## Prompt 4
Mathematica code to generate read depth over time graphs for Naive and Snake Order with GC.

### Key Visual Insights
1. **N^2 Plateau (Naive):** With GC, Matrix B depth jumps to exactly N^2 (not 3N^2), proving cost drops to 1.0*N^4
2. **Hot Cache Floor (Matrix A):** Stays at depth ~2N in both algorithms, near-zero routing penalty
3. **Sawtooth Wave (Snake):** B's flat band becomes a triangular wave — depth plunges to 0 at turn-arounds and climbs to N^2. The area under the sawtooth integrates to 0.66*N^4

Full Mathematica simulation code provided for N=16 with garbage-collected LRU cache simulation.
