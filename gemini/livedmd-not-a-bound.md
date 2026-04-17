To answer your question directly: **No, it is mathematically false.**

ByteDMD-live does **not** provide a strict lower bound on the energy cost of an actual physical implementation in a continuous cache.

In fact, a realistic physical allocator with fixed addresses (which strictly prices every read and has no "free sliding") can achieve a **significantly lower** energy cost than ByteDMD-live. By asking this, you have discovered a theoretical flaw in the "Envelope Claim" (cost(any\_allocator) \>= bytedmd\_live) from the bytedmd\_ir framework.

It is highly intuitive to assume that the "free sliding" of ByteDMD-live gives it an unbeatable advantage. However, ByteDMD-live simulates an **LRU (Least Recently Used) stack**. In a physical 2D cache, LRU's sliding behavior is often a severe mathematical penalty, not an advantage.

Here is the proof by two counterexamples, followed by the conceptual explanation.

### ---

**Proof 1: The "Hot-Cold" Loop Counterexample**

Imagine a trace where one heavily reused "Hot" variable ($H$) is interleaved with temporary "Cold" variables ($C\_i$) that are stored, read once, and immediately die.

**The Trace:**

1. Store H  
2. Loop $N$ times:  
   * Store C\_i  
   * Load H  
   * Load C\_i *(C\_i dies)*

Let's evaluate the exact cost of a single loop iteration under the continuous cache model where reading from depth $d$ costs $\\sqrt{d}$.

**1\. Evaluation under ByteDMD-live**

* **Store C\_i**: $C\_i$ is placed at the top of the stack. The stack becomes \[C\_i, H\]. The Hot variable $H$ is forced down to depth 2\!  
* **Load H**: $H$ is at depth 2\. **Cost \= $\\sqrt{2} \\approx 1.414$**.  
  * *LRU updates*: $H$ slides to the top. Stack becomes \[H, C\_i\]. $C\_i$ is pushed to depth 2\.  
* **Load C\_i**: $C\_i$ is at depth 2\. **Cost \= $\\sqrt{2} \\approx 1.414$**.  
  * *Compaction*: $C\_i$ dies and vanishes. Stack returns to \[H\].  
* **Total Loop Cost:** $1.414 \+ 1.414 \= \\mathbf{2.828}$

**2\. Evaluation under a Physical Allocator (e.g., min\_heap / Scratchpad)**

A physical allocator statically pins variables to fixed addresses and immediately reuses holes.

* It statically pins the Hot variable $H$ to **Address 1**.  
* It assigns the temporary $C\_i$ to **Address 2**. When $C\_i$ dies, Address 2 is instantly freed for the next iteration.  
* **Load H**: Read from Address 1\. **Cost \= $\\sqrt{1} \= \\mathbf{1.0}$**.  
* **Load C\_i**: Read from Address 2\. **Cost \= $\\sqrt{2} \\approx \\mathbf{1.414}$**.  
* **Total Loop Cost:** $1.0 \+ 1.414 \= \\mathbf{2.414}$

**Conclusion:** **$2.414 \< 2.828$**. The physical implementation natively executes the loop nearly **15% cheaper** than ByteDMD-live.

### ---

**Proof 2: The Cyclic Loop Counterexample**

Assume we allocate an array of $K$ variables, and we read them in a repeating sequential sweep (A, B, C... K) for several iterations.

**1\. The ByteDMD-live Cost**

Because ByteDMD-live enforces LRU, reading a variable pushes everything else down. In a cyclic loop, by the time you cycle back to reading variable A, the other $K-1$ reads have maliciously pushed A to the absolute bottom of the stack (depth $K$).

* Every single read in the loop happens at the maximum depth $K$.  
* **Cost per loop:** $K \\times \\sqrt{K} \= \\mathbf{K^{1.5}}$

**2\. The Actual Physical Implementation Cost**

A physical allocator simply assigns the $K$ variables to stationary physical addresses $1$ through $K$. There is zero sliding; they sit permanently. The processor reads from those fixed addresses.

* **Cost per loop:** $\\sum\_{i=1}^{K} \\sqrt{i} \\approx \\int\_0^K \\sqrt{x} \\, dx \= \\mathbf{\\frac{2}{3}K^{1.5}}$

**Conclusion:** Because $\\frac{2}{3} K^{1.5} \< K^{1.5}$, the physical implementation natively executes the sweep **33% cheaper** than ByteDMD-live.

### ---

**Why does "Free Sliding" lose?**

There are two fundamental reasons why explicit physical addressing mathematically defeats ByteDMD-live:

1. **The LRU Allocation Penalty:** In ByteDMD-live, whenever you allocate a *new* variable, it claims Depth 1, forcing all existing working data to slide deeper away from the processor. A smart compiler/physical allocator avoids this by pinning the hot working set at Address 1 and routing temporary variables around it.  
2. **Jensen's Inequality (Concavity):** The continuous cache cost function $f(x) \= \\sqrt{x}$ is strictly concave. Mathematically, polarizing your costs (keeping your hot items permanently at Address 1 and cold items permanently at Address 100\) yields a smaller sum than averaging them out. The continuous LRU "sliding" in ByteDMD-live constantly averages the distances out, inadvertently maximizing the concave penalty.

### **Summary**

ByteDMD-live is **not** the mathematical lower bound for data movement. It is simply a simulation of a *Cache-Oblivious LRU policy with perfect garbage collection*.

The true mathematical lower bound for data movement on a spatial grid is defined by **Bélády’s MIN algorithm** (Optimal Offline Static Routing). Because actual physical allocators explicitly pin data to avoid LRU's pathological cache thrashing, they regularly pierce the ByteDMD-live "floor" on realistic compiler workloads.