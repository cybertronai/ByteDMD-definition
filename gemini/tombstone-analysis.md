# tombstone-analysis

**Source:** https://gemini.google.com/share/6eacee1a7249
**Created with:** Gemini 3 Deep Think, April 9, 2026
**Topic:** ByteDMD formulas with tombstone/liveness-based LRU stack, model fixes, cold miss pricing

---

## Prompt 1
Exact ByteDMD formulas for matvec and vecmat with liveness-based implementation.

Reference scores: matvec=187, vecmat=181 for $N=4$. Requires fully unrolled left-to-right AST evaluation (not standard accumulator loop which would give 193/201).

### 1. Exact ByteDMD Cost Formulas ($N \geq 4$)

To capture the ceilings exactly, we define the exact closed-form algebraic sum of square roots (mirroring `_sum_usqrt` logic):

$$S(M) = \sum_{k=1}^{M} \lceil \sqrt{k} \rceil = \frac{K(6M - 2K^2 + 3K - 1)}{6} \quad \text{where } K = \lfloor \sqrt{M-1} \rfloor + 1$$

Both matvec and vecmat incur costs from three components: reading the matrix $A$, intermediate arithmetic additions, and reading the vector $x$. Because the AST evaluates left-to-right, every scalar addition is priced at $\lceil\sqrt{4}\rceil + \lceil\sqrt{2}\rceil = 4$.

#### A. Matrix-Vector (`matvec`, loop order i-j)

Because matvec evaluates $A$ sequentially by row, elements of $A$ are always fetched from the absolute bottom of the active LRU stack (depth $N^2 + N$) and then immediately replaced by tombstones. This steadily rotates elements of $x$, shifting them down until they hit a hard equilibrium depth of exactly $4N$.

- **Matrix $A$ Cost:** $N^2 \lceil\sqrt{N^2 + N}\rceil = N^3 + N^2$
- **Addition Cost:** $4N(N-1)$
- **Vector $x$ Cost:** $S(4N-1) - S(N) + \lceil\sqrt{3N}\rceil + (N-3)\lceil\sqrt{4N-1}\rceil + (N^2 - 4N + 3)\lceil\sqrt{4N}\rceil$

**Exact Formula:**

$$C_{\text{matvec}}(N) = N^3 + 5N^2 - 4N + S(4N-1) - S(N) + \lceil\sqrt{3N}\rceil + (N-3)\lceil\sqrt{4N-1}\rceil + (N^2 - 4N + 3)\lceil\sqrt{4N}\rceil$$

(Verification: At $N=4$, $80 + 48 + 59 = \mathbf{187}$)

#### B. Vector-Matrix (`vecmat`, loop order j-i)

In vecmat, $A$ is traversed column-by-column, meaning we stride diagonally through the row-major memory stack. The depth of element $A_{i,j}$ physically shrinks as earlier columns are evaluated.

**Matrix $A$ Cost:**

$$\text{Cost}_A = \sum_{j=0}^{N-1} \sum_{i=0}^{N-1} \lceil\sqrt{N^2 + N - i(N - 1 - j)}\rceil$$

**Addition Cost:** $4N(N-1)$

**Vector $x$ Cost:**

$$\text{Cost}_x = S(4N-2) - S(N-1) + \lceil\sqrt{3N-1}\rceil + \lceil\sqrt{4N-2}\rceil + (N-3)\lceil\sqrt{4N-1}\rceil + (N^2 - 4N + 2)\lceil\sqrt{4N}\rceil$$

**Exact Formula:**

$$C_{\text{vecmat}}(N) = \text{Cost}_A(N) + 4N(N-1) + \text{Cost}_x(N)$$

(Verification: At $N=4$, $76 + 48 + 57 = \mathbf{181}$)

### 2. Tight Continuous Bounds & Analytic Constants

By bounding the ceiling function $\sqrt{z} \leq \lceil\sqrt{z}\rceil < \sqrt{z} + 1$, we extract tight asymptotic continuous envelopes.

**Matvec Bounds:**

$$N^3 + 2N^{5/2} + 5N^2 - \frac{10}{3}N^{3/2} - 4N \leq C_{\text{matvec}}(N) \leq N^3 + 2N^{5/2} + 6N^2 - \frac{10}{3}N^{3/2}$$

**Vecmat Bounds:**

The matrix $A$ cost traverses the stack out-of-order. Integrating the dominant sum over the continuous 2D surface yields an exact geometric constant:

$$\mathcal{K} = \frac{16}{9} - \frac{4}{3}\ln 2 \approx 0.85358$$

$$\mathcal{K}N^3 + 2N^{5/2} + 4N^2 - \frac{10}{3}N^{3/2} - 4N \leq C_{\text{vecmat}}(N) \leq \mathcal{K}N^3 + 2N^{5/2} + 6N^2$$

**Algorithmic Conclusion:** vecmat scales structurally better ($\approx 0.853N^3$ vs $1.0N^3$). Striding out-of-order frequently pulls values that have floated upward via tombstones, whereas row-major matvec monopolistically draws from the deepest possible block of the stack.

### 3. Discrepancies and Inconsistencies in the Tracer

1. **Sequential Pricing Violates "Simultaneous" Instruction Pricing:** The `_Context.read()` method evaluates and LRU-bumps inputs sequentially mid-instruction. For `a + a` where `a` sits at depth 5, the first read is priced at 5, brings `a` to the top (depth 1), and the second read is priced at 1. Total: $\lceil\sqrt{5}\rceil + \lceil\sqrt{1}\rceil = 4$, rather than the documented simultaneous evaluation $\lceil\sqrt{5}\rceil + \lceil\sqrt{5}\rceil = 6$. Furthermore, `b+c` vs `c+b` yield different costs, breaking operator commutativity.

2. **The "Leapfrog" Anomaly (Rewarding Cache Thrashing):** Accessing a matrix column-wise leapfrogs over unread row elements. Those bypassed elements don't get evicted at the scalar level — they just hang around at the bottom. ByteDMD calculates column-major access as $\approx 15\%$ cheaper than contiguous row-major movement, unintentionally rewarding the destruction of 2D spatial locality.

3. **Argument Ordering Hack:** vecmat costs 181 only with `vecmat(A, x)`. The mathematical standard signature `vecmat(x, A)` for $y = x^T A$ traps $x$ underneath the matrix, inflating the trace cost to 189. The benchmark subtly favored vecmat by preserving matvec's optimal $(A, x)$ push order.

4. **Tombstone Teleportation:** `STORE` allocations lazily `.remove(None)` from the first tombstone found (bottom-most). This teleports dead memory directly from the absolute bottom of the cache to the top instantly, without physically shifting the live intermediate elements down, contradicting the physical VLSI routing premise.

---

## Prompt 2
Suggestions for how to fix this model while keeping formulas analytically compact.

### Four Architectural Fixes

**1. Fix "Simultaneous" Instruction Pricing (Restores Commutativity)**

Price all inputs against the pre-instruction stack, then batch-move uniquely accessed values to the top deterministically.

```python
def read(self, keys):
    valid = list(dict.fromkeys(k for k in keys if k is not None))
    if not valid:
        return []
    # 1. Price simultaneously against pre-instruction stack
    depths, cold_misses = [], []
    for k in valid:
        if k in self.stack:
            depths.append(len(self.stack) - self.stack.index(k))
        else:
            cold_misses.append(k)
            depths.append(len(self.stack) + len(cold_misses))
    self.trace.extend(depths)
    self.stack.extend(cold_misses)
    # 2. Batch-update LRU to MRU position
    for k in valid:
        self.stack.remove(k)
        self.stack.append(k)
    return depths
```

**2. Cache-Line Chunking (Fixes the Locality / Thrashing Anomaly)**

Group contiguous memory into logical blocks of size $B$ (e.g., $B=16$ scalars). Map elements to blocks: `block_id = base_addr + index // B`. The LRU stack tracks `block_ids`, meaning a read to any element pulls the entire block to MRU position.

**3. Demand-Paged Initialization (Removes Signature Bias)**

Start the tracer with an empty stack. When a block is accessed for the first time, treat it as a "Cold Miss" appended to the stack organically, achieving a symmetric baseline independent of Python parameter order.

**4. Natural LRU Aging (Removes Tombstone Teleportation)**

Delete the Pass 2 `_replay_with_liveness` logic entirely. Let dead variables sink naturally to the bottom as they age out. The stack bounds organically anchor to the algorithm's exact working-set size.

### Resulting Analytically Compact Formulas (with block size $B$)

**Corrected Matrix-Vector (matvec, loop order i-j):**

Because rows are contiguous, matvec exhibits perfect spatial locality. A block of $A$ stream-hits at depth 1. The vector $x$ rotates in a shallow working set of size $N/B$ blocks.

$$C_{\text{matvec}}(N) = \underbrace{N^2(1 - \frac{1}{B}) \cdot 1}_{\text{A stream hits}} + \underbrace{N^2 \cdot \sqrt{\frac{N}{B}}}_{\text{x working set hits}}$$

**Leading Order:** $O\!\left(\frac{N^{2.5}}{\sqrt{B}}\right)$

**Corrected Vector-Matrix (vecmat, loop order j-i):**

vecmat strides vertically, requesting a scalar from a completely different block on every read. It must access $N$ distinct blocks before returning to the first column.

$$C_{\text{vecmat}}(N) = \underbrace{N^2(1 - \frac{1}{B}) \cdot \sqrt{N}}_{\text{A thrashing hits}} + \underbrace{N^2 \cdot \sqrt{N}}_{\text{x thrashing hits}}$$

**Leading Order:** $O(2N^{2.5})$

**Conclusion:** The penalty ratio vecmat/matvec resolves to $\approx 2\sqrt{B}$. For $B=16$, vecmat is **8 times more expensive**, correctly reflecting real-world spatial locality paradigms.

---

## Prompt 3
Cold miss pricing and preventing nesting exploits.

### Physical Motivation: DRAM is a 2D Grid

The ALU sits at the center (Depth 0). The LRU stack represents L1/L2 cache expanding outward in concentric rings. Before you read an element for the first time, it lives in off-chip Main Memory (DRAM). If an algorithm requires a $1000 \times 1000$ matrix, it physically occupies 1,000,000 slots of 2D area. The $M$-th element is at spatial distance $\approx \sqrt{M}$ from the processor.

A Cold Miss (reading an element for the first time) represents routing a wire across the DRAM grid. Therefore, **compulsory cache misses cannot be cheap.**

### The Exploit: The "Nesting / Streaming" Cheat

The routing cost $\sqrt{z}$ is strictly sub-additive: $\sqrt{A+B} < \sqrt{A} + \sqrt{B}$.

A programmer wants to process a 1,000,000-element array. In one monolithic loop, initialization costs:

$$\sum_{k=1}^{1{,}000{,}000} \sqrt{k} \approx 666{,}000{,}000$$

To cheat, the programmer streams through a nested helper that resets context or keeps the LRU stack at size 1. Every fetch costs $\lceil\sqrt{1}\rceil = 1$. Total drops to 1,000,000. The programmer mathematically pretended a million-element matrix occupies 1 unit of physical space.

### Solution: Dual-Tracked Spatial Routing

**LRU Stack (L1 Cache):** Tracks active, "hot" variables. Can stay small if dead variables are dropped.

**Monotonic DRAM Frontier:** A single global integer counting the total number of unique elements the program has ever fetched. **It never decreases**, even when variables die.

$$\text{Total Cold Miss Cost} = \sum_{k=1}^{M} \lceil\sqrt{k}\rceil \approx \int_0^M z^{1/2}\,dz = \frac{2}{3}M^{3/2}$$

If the developer splits the algorithm into nested functions $A$ and $B$, the definite integrals chain seamlessly:

$$\underbrace{\int_0^A \sqrt{z}\,dz}_{\text{Nested Func A}} + \underbrace{\int_A^{A+B} \sqrt{z}\,dz}_{\text{Nested Func B}} = \int_0^{A+B} \sqrt{z}\,dz$$

The abstraction boundary costs exactly zero, but it hides nothing. The programmer pays the $O(M^{1.5})$ physical spatial tax dictated by raw input size, no matter how they structure their loops.

### Implementation Rules

**Rule A: The Global Context.** The `_Context` object must be a singleton initialized once at the start of `traced_eval`. It is passed seamlessly into all nested functions. Do not clear the context or LRU stack when scopes change.

**Rule B: Lazy Input Paging.** Remove eager `_wrap` pushing of all arguments at function entry. Assign tracking IDs to arguments but leave them "in DRAM." An element only enters the chip when an instruction explicitly evaluates it.

**Rule C: The Pricing Engine.** Update `_Context.read()` to evaluate Hot vs Cold independently:

```python
class _Context:
    def __init__(self):
        self.stack = []          # Active L1 cache (can recycle tombstones)
        self.dram_frontier = 0   # Monotonic global spatial boundary

    def read(self, keys):
        valid_keys = list(dict.fromkeys(k for k in keys if k is not None))
        if not valid_keys:
            return []
        depths = []
        for k in valid_keys:
            if k in self.stack:
                # HOT HIT: Element is on-chip. Find depth in active L1 cache.
                depths.append(len(self.stack) - self.stack.index(k))
            else:
                # COLD MISS: Element is fetched from off-chip DRAM grid.
                self.dram_frontier += 1
                depths.append(self.dram_frontier)
                # Bring into L1 Cache (recycle a tombstone if available)
                try:
                    self.stack.remove(None)
                except ValueError:
                    pass
                self.stack.append(k)
        self.trace.extend(depths)
        # LRU BUMP: Batch move all accessed values to the ALU (top of stack)
        for k in valid_keys:
            self.stack.remove(k)
            self.stack.append(k)
        return depths
```

(Note: Newly computed intermediates like `c = a + b` are generated directly at the ALU. They are pushed to `self.stack` at Depth 1 via `allocate`. Because they are not fetched from Main Memory, they **do not** increment `dram_frontier`.)
