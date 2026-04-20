To establish a mathematically airtight lower bound on the true physical cost of **Optimal Manual Memory Allocation** (using explicit DMA transfers under the Manhattan Diamond model) based on the theoretical ByteDMD-live metric, we must bridge continuous 2D spatial geometry with the discrete competitive analysis of classical caching theory.

Here is the rigorous, step-by-step mathematical proof. We will build it from the ground up: explicitly defining the physical axioms, using calculus to transform spatial routing into an infinite hierarchy of caches, and finally applying Sleator and Tarjan’s 1985 capacity-scaling theorem.

### ---

**Part 1: The Formal Axioms and Assumptions**

To make the proof rigorous, we define the exact constraints of the hardware, the manual allocator, and the theoretical baseline.

**Axiom 1: The Continuous Cost Relaxation**

* Memory is a 1D array of addresses indexed $d \in \{1, 2, 3, \dots\}$.  

* The Manhattan Diamond model specifies the cost to read from address $d$ is the discrete function $C(d) = \lceil\sqrt{d}\rceil$.  

* For analytical tractability via calculus, we relax this to the continuous lower bound $C(d) = \sqrt{d}$. Because $\sqrt{d} \le \lceil\sqrt{d}\rceil$ for all $d \ge 1$, any mathematical lower bound established on the continuous function is strictly guaranteed to hold for the true discrete cost.

**Axiom 2: Asymmetric Operation Costs (The DMA Tax)**

* Under the Manhattan Diamond rules, arithmetic computations and memory writes consume $0$ energy. Only memory reads incur the spatial distance penalty $\sqrt{d}$.  

* **Explicit DMA Copies:** If the manual allocator ($OPT_{Manual}$) explicitly copies a variable from a deep Main Memory address $D$ to a fast L1 Scratchpad address $d$, it must issue a read from $D$ and a write to $d$. By the rule above, the physical data-movement cost of this explicit DMA copy is exactly $\mathbf{\sqrt{D}}$.

**Axiom 3: Oracle Liveness Equivalence**

* ByteDMD-live simulates perfect liveness compaction: the exact nanosecond a variable undergoes its final read, it is deleted from the LRU stack.  

* The optimal offline dynamic cache (Bélády’s MIN algorithm, denoted $OPT$) naturally possesses this identical behavior. Because it knows the future, it instantly evicts any variable whose time-to-next-access is infinity. Both algorithms operate on the exact same compacted trace of *live* variables.

### ---

**Part 4: Lemma 1 — Translating Spatial Cost to Cache Misses**

In standard caching theory, you analyze a single cache of size $x$. In the Manhattan Diamond model, memory is a continuous spatial hierarchy.

Let’s calculate the total data-movement cost $C_{Manual}$ for a program trace $\mathcal{T}$. For each request $t \in \mathcal{T}$, the manual allocator must fetch the data. It can either read it directly for the ALU, or issue a DMA copy to bring it into a closer address and then read it. In either scenario, let $d^*_t$ be the maximum physical depth read from to satisfy request $t$. The total cost is:

$$ C_{Manual} \ge \sum_{t \in \mathcal{T}} \sqrt{d^*_t} $$  

By the Fundamental Theorem of Calculus, we can rewrite the distance penalty $\sqrt{d^*_t}$ as a definite integral:

$$ \sqrt{d^*_t} = \int_{0}^{d^*_t} \frac{1}{2\sqrt{x}} \\, dx $$  

Substituting this into our total cost equation:

$$ C_{Manual} \ge \sum_{t \in \mathcal{T}} \int_{0}^{\infty} \mathbf{1}(x < d^*_t) \frac{1}{2\sqrt{x}} \\, dx $$  

*(where the indicator function $\mathbf{1}(x < d^*_t)$ is $1$ if the read occurred at a depth strictly greater than $x$, and $0$ otherwise).*

Because both the sum and the integral are finite and non-negative, Fubini's Theorem allows us to safely swap their order:

$$ C_{Manual} \ge \int_{0}^{\infty} \left( \sum_{t \in \mathcal{T}} \mathbf{1}(d^*_t > x) \right) \frac{1}{2\sqrt{x}} \\, dx $$  

**The Crucial Insight:** Look at the inner sum. It counts exactly how many times the manual allocator executed a physical read from an address strictly greater than $x$.

### ---

**Part 3: Lemma 2 — Bounding DMA Fetches via Bélády's MIN**

Let's analyze the physical region defined by addresses $1$ through $x$. This physical region has a strict maximum capacity of $x$ variables. It acts identically to a hardware cache of size $x$.

If request $t$ requires a variable that is currently *not* in this region, the manual allocator must fetch it from outside. Whether it reads it directly for the ALU, or uses an explicit DMA copy to bring it into the region $1 \dots x$, it **must** execute a read from an address $D > x$. This means $d^*_t > x$, adding $1$ to our inner sum.

Because the physical region $1 \dots x$ can hold at most $x$ items at any given time, any sequence of placements, copies, and overwrites the manual allocator performs is strictly subject to the mathematical Pigeonhole Principle of a cache of size $x$.

Let $M_{OPT}(x)$ be the absolute minimum number of cache misses achievable by the optimal offline clairvoyant algorithm (Bélády’s MIN) for a cache of capacity $x$. Since $OPT_{Manual}$ cannot magically pack more than $x$ items into the top $x$ physical slots, its number of fetches from outside this region can never be lower than the mathematical minimum:

$$ \sum_{t \in \mathcal{T}} \mathbf{1}(d^*_t > x) \ge M_{OPT}(x) \quad \text{for all } x $$  

Substituting this into the integral from Lemma 1 establishes the universal spatial lower bound:

$$ C_{Manual} \ge \int_0^\infty \frac{M_{OPT}(x)}{2\sqrt{x}} dx \quad \equiv C_{OPT} $$  

*(By the exact same logical identity, the theoretical cost of our metric is $C_{ByteDMD-live} = \int_0^\infty \frac{M_{LRU}(x)}{2\sqrt{x}} dx$.)*

### ---

**Part 4: Lemma 3 — Sleator and Tarjan's Competitive Analysis**

In 1985, Sleator and Tarjan published their seminal competitive analysis of LRU. They proved that for *any* request sequence, an automatic LRU cache of capacity $h$ suffers a bounded number of misses compared to the absolute optimal offline cache (MIN) of a smaller capacity $k$ (where $h > k$):

$$ M_{LRU}(h) \le \frac{h}{h-k} M_{OPT}(k) $$  

We algebraically rearrange this inequality to isolate $M_{OPT}(k)$, which is what we need to lower-bound our integral:

$$ M_{OPT}(k) \ge \left( 1 - \frac{k}{h} \right) M_{LRU}(h) $$  

To apply this to our continuous integral, we lock the capacities to a constant scaling ratio $c > 1$. Let the LRU capacity be $x$, and the OPT capacity be $x/c$. Substituting this yields:

$$ M_{OPT}\left(\frac{x}{c}\right) \ge \left( 1 - \frac{1}{c} \right) M_{LRU}(x) $$

### ---

**Part 5: Integrating the Competitive Bound (The Calculus)**

We take the continuous integral for $C_{OPT}$ from Lemma 2, using the integration variable $y$:

$$ C_{OPT} = \int_{0}^{\infty} \frac{M_{OPT}(y)}{2\sqrt{y}} \\, dy $$  

Perform a $u$-substitution to scale the integral by $c$. Let $x = c \cdot y$. This implies $y = \frac{x}{c}$, and the differential $dy = \frac{dx}{c}$:

$$ C_{OPT} = \int_{0}^{\infty} \frac{M_{OPT}(x/c)}{2\sqrt{x/c}} \left( \frac{dx}{c} \right) $$  

Substitute the Sleator-Tarjan lower bound $M_{OPT}(x/c) \ge (1 - 1/c) M_{LRU}(x)$:

$$ C_{OPT} \ge \int_{0}^{\infty} \frac{ \left( 1 - \frac{1}{c} \right) M_{LRU}(x) }{ 2\sqrt{x/c} } \left( \frac{dx}{c} \right) $$  

Now, extract the constants out of the integral. The denominator evaluates to: $2 \sqrt{\frac{x}{c}} \cdot c = 2 \frac{\sqrt{x}}{\sqrt{c}} \cdot c = 2\sqrt{x}\sqrt{c}$.

$$ C_{OPT} \ge \left( 1 - \frac{1}{c} \right) \frac{1}{\sqrt{c}} \int_{0}^{\infty} \frac{M_{LRU}(x)}{2\sqrt{x}} \\, dx $$

$$ C_{OPT} \ge \left( \frac{c - 1}{c^{3/2}} \right) \int_{0}^{\infty} \frac{M_{LRU}(x)}{2\sqrt{x}} \\, dx $$  

Notice that the remaining integral is exactly the mathematical definition of $C_{ByteDMD-live}$! We have cleanly factored out the competitive ratio:

$$ C_{Manual} \ge C_{OPT} \ge \left( \frac{c - 1}{c^{3/2}} \right) C_{ByteDMD-live} $$

### ---

**Part 6: Maximizing the Bound**

The inequality we derived holds mathematically for *any* capacity scaling factor $c > 1$. To establish the tightest possible physical lower bound, we must find the global maximum of the function $f(c) = \frac{c - 1}{c^{3/2}}$.

1. Rewrite the function: $f(c) = c^{-1/2} - c^{-3/2}$.  

2. Take the first derivative with respect to $c$:  

   $$ f'(c) = -\frac{1}{2}c^{-3/2} - \left(-\frac{3}{2}c^{-5/2}\right) = \frac{3}{2}c^{-5/2} - \frac{1}{2}c^{-3/2} $$  

3. Set $f'(c) = 0$ to find the critical maximum:  

   $$ \frac{3}{2}c^{-5/2} = \frac{1}{2}c^{-3/2} $$  

4. Multiply both sides by $2c^{5/2}$ (valid since $c > 1$):  

   $$ 3 = c $$

The bound reaches its absolute maximum when we scale the comparison such that the theoretical LRU cache is evaluated at exactly $3\times$ the capacity of the optimal offline cache.

Substitute $c = 3$ back into the original multiplier function $f(c)$:

$$ f(3) = \frac{3 - 1}{3^{3/2}} = \frac{2}{\sqrt{27}} = \mathbf{\frac{2}{3\sqrt{3}}} \approx \mathbf{0.384900} $$

### ---

**Final Theorem**

By chaining the spatial calculus of DMA costs with the capacity-scaling bounds of competitive cache analysis, we arrive at the final theorem:

$$ \mathbf{C_{Manual\\_DMA} \ge \frac{2}{3\sqrt{3}} C_{ByteDMD-live}} \quad \blacksquare $$  

This rigorously proves that ByteDMD-live is fundamentally anchored to the absolute limits of physical reality under the Manhattan Diamond model. No explicit software DMA strategy—even one compiled by an omniscient offline oracle paying full read penalties to optimally shift data—can ever reduce total data-movement energy below $\approx 38.49\%$ of the ByteDMD-live score.
