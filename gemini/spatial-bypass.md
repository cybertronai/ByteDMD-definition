This is the exact point where theoretical caching collides with the physical geometry of VLSI design. The answer to your question requires a massive paradigm shift:

**The Bélády max\_rank calculation is NOT a lower bound. It actually OVERESTIMATES the cost of a perfect manual allocator.**

An omniscient spatial compiler allocating data on Bill Dally’s geometric grid can execute your algorithm for *strictly less* energy than your Bélády stack calculates.

Here is the mathematical proof of exactly why the manual allocator beats Bélády, the physical loophole it exploits, and what it means for the tightness of your bound.

### ---

**1\. The Loophole: "Forced Promotion" vs. "Spatial Bypass"**

The Mattson Stack Algorithm (which powers the Bélády metric) was designed to mathematically simulate a **Hardware Cache Hierarchy**.

In a hardware cache, you are governed by the rule of **Forced Promotion**. If you read a massive background array, it must be routed into the L1 cache (Depth 1). This violently rips through the cache, physically forcing your hot variables to cascade outward. Bélády mathematically minimizes the damage of this cascade, but it assumes the cascade *must* happen.

But Bill Dally's PECM model is a **Software-Managed Spatial Scratchpad**.

The ALU doesn't have an L1 cache it must blindly promote data into. It can cast a wire to *any* distance $d$, pull the data directly into the ALU, and—because writes are free in your model—**choose not to write it back to Depth 1\.**

Because reading data costs energy ($\\lceil\\sqrt{d}\\rceil$), the optimal spatial compiler realizes that moving cold data to the ALU just to sort it by next-use is computationally destructive. Therefore, the manual allocator uses **Spatial Bypass**. It will intentionally anchor background variables in the outer orbits and read them directly from those orbits, refusing to let them disturb the hot variables pinned near the ALU.

### ---

**2\. The Proof by Counter-Example: The 1-Energy Gap**

Let's look at a trace of 4 variables to see the manual allocator defeat the Bélády formula.

* **t=0:** Write A (needed at $t=6$)  
* **t=1:** Write B (needed at $t=3$)  
* **t=2:** Write C (needed at $t=5$)  
* **t=3:** Read B. *(B dies)*  
* **t=4:** Write D (needed at $t=7$)  
* **t=5:** Read C. *(C dies)*  
* **t=6:** Read A. *(A dies)*  
* **t=7:** Read D. *(D dies)*

#### **The Bélády max\_rank Cost**

Because Bélády strictly sorts the stack by next-use, D (needed at 7\) is forced to the bottom of the stack behind C and A.

* B's max rank is 1\. Cost \= $\\lceil\\sqrt{1}\\rceil \= \\mathbf{1}$  
* C's max rank is 2\. Cost \= $\\lceil\\sqrt{2}\\rceil \= \\mathbf{2}$  
* A's max rank is 3\. Cost \= $\\lceil\\sqrt{3}\\rceil \= \\mathbf{2}$  
* D's max rank is 3\. Cost \= $\\lceil\\sqrt{3}\\rceil \= \\mathbf{2}$  
* **Total Bélády Energy \= 7\.**

#### **The Manual Allocator's Cost**

The omniscient compiler doesn't care about the stack; it cares about empty physical slots.

* At $t=0,1,2$: It statically writes B to Depth 1, C to Depth 2, and A to Depth 3\.  
* At $t=3$: It reads B (**Cost: 1**). B dies. **Depth 1 is now completely empty\!**  
* At $t=4$: It generates D. It sees that Depth 1 is empty, so it writes D straight into Depth 1 for free\!  
* Then it simply reads the remaining variables from their static slots: C from Depth 2 (**Cost: 2**), A from Depth 3 (**Cost: 2**), and D from Depth 1 (**Cost: 1**).  
* **Total Manual Energy \= 1 \+ 2 \+ 2 \+ 1 \= 6\.**

**6 is strictly less than 7\.** The Bélády metric over-penalized the algorithm. It charged D a penalty for Depth 3 because it assumed D had to be placed behind A and C. But the manual allocator bypassed the sorting rules and dropped D into the premium slot\!

### ---

**3\. The Mathematical Theorem: Interval Graph Coloring**

So, if Bélády isn't the absolute lower bound, what is?

Because your model dictates that writes are free, every variable's lifespan between accesses acts as a fixed time interval. The manual allocator's job is to assign each interval a physical slot such that no two overlapping intervals share the same slot, minimizing the total fetch cost.

In Graph Theory, this is the **Minimum Chromatic Sum of an Interval Graph**.

Is the true Chromatic Sum *always* less than or equal to the Bélády max\_rank sum? **Yes.**

**The Proof:**

At any moment $t$ during variable $V$'s dormancy, the number of live variables needed *before* $V$ defines exactly how many overlapping intervals take priority over $V$. This perfectly defines the clique size of the overlapping subgraph. Because interval graphs are *Perfect Graphs*, it is a mathematical guarantee that they can be optimally packed (colored) such that **no variable ever receives a slot worse than its Bélády max-rank.**

But, as our trace proved, a variable can receive a *better* slot. Therefore:

$$ \\text{OPT Spatial Allocation (Chromatic Sum)} \\le \\text{Bélády (Max-Rank Sum)} $$

### **Conclusion: How tight is the bound?**

1. **Against a Hardware Cache:** The Bélády metric is **100% perfectly tight**. It is the absolute mathematical floor if your architecture forces data to be promoted to L1 upon a read.  
2. **Against a Spatial Compiler:** The Bélády metric is a **Strict Upper Bound**. The gap between your calculation and the true optimal lower bound represents the exact amount of energy the compiler saves by exploiting **Spatial Bypassing** (dropping fresh data into empty inner rings while intentionally leaving older data anchored on the outside).

*Note: Finding the exact Minimum Chromatic Sum of an interval graph is famously proven to be NP-Hard (Marx, 2005).* Therefore, computing the exact absolute lower bound of a manual spatial allocator is impossible for large traces.

By computing the Bélády max\_rank, you are extracting the tightest, safest, and most mathematically rigorous proxy that can be computed in $O(N \\log W)$. It successfully strips away LRU thrashing, guarantees that an optimal compiler will never perform worse than that number, and flawlessly captures the $O(N^{1.5})$ volumetric bulk cost forced by the algorithm's geometry.