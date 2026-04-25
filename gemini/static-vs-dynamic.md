You have just empirically discovered one of the most fundamental theorems in compiler optimization: **A purely static memory allocator is completely blind to temporal locality (loop reordering).**

To answer your question directly: **Yes, there is a better lower bound that automatically accounts for optimal copying, and you already have it in your script.** It is **bytedmd\_opt** (Bélády's MIN).

Here is the exact mathematical explanation of why static\_opt\_lb is blind to loop tiling, why "inserting copies" transforms static theory into dynamic theory, and why bytedmd\_opt is the ultimate lower bound you are looking for.

### ---

**1\. The "Monolithic Rock" Problem (static\_opt\_lb)**

Why do Naive Matmul and Tiled Matmul get the exact same lower bound under static\_opt\_lb?

Because static\_opt\_lb calculates the access density of a variable over its **global lifespan** ($\\text{Total Reads} / \\text{Total Lifespan}$).

In both algorithms, Matrix A and Matrix B are born at $t=0$, and their final reads occur at the very end of the program. They live for $\\Theta(N^3)$ cycles, and are read $\\Theta(N^3)$ times.

Because static\_opt\_lb assumes a variable is a monolithic rock permanently pinned to a single physical address, the order in which the reads happen is completely invisible to it. The LP floor evaluates the exact same set of variables with the exact same global densities, resulting in the exact same physical cost.

### **2\. Why Explicit Copies Drop the Cost**

When you manually insert a copy (e.g., copying a tile of A into a scratchpad variable sA), you fundamentally change the physics of the trace.

You slice that giant, low-density lifespan into tiny, ultra-dense **Bursts**. The new temporary variable sA lives for only a few microseconds but absorbs massive read volume. static\_opt\_lb instantly recognizes this astronomical local density, assigns sA to Address 1, and your bound plummets. The original Matrix A is relegated to deep memory, but you only pay the penalty to read from it *once* to populate sA.

### **3\. The Isomorphism: "Explicit Copies" \= "Dynamic Caching"**

What you are describing as an algorithm designer—*"creating copies of frequently used parts, moving them into lower rank memory, using them, and discarding them"*—is the exact physical definition of a **Dynamic Hardware Cache**.

A hardware LRU cache fetching a cache-line into SRAM is functionally identical to a software DMA engine allocating a temporary L1\_Copy variable.

Therefore, the absolute theoretical lower bound for an algorithm designer inserting *optimal manual copies* into a static scratchpad is exactly equal to the absolute theoretical lower bound of an *optimal dynamic cache*.

### **4\. Why bytedmd\_opt is the Ultimate Arbitrator**

Because dynamic caches only keep data in fast memory during active "bursts" of reuse, they are highly sensitive to loop reordering.

Your bytedmd\_opt metric evaluates the **Bélády MIN stack distance**. It looks at the temporal gap between reuses:

* **Naive Matmul:** The inner loop reads A\[0\]\[0\], then sweeps the entire rest of the matrix before returning to A\[0\]\[0\]. The temporal gap is massive. bytedmd\_opt recognizes that any dynamic allocator will be forced to evict it, charging a massive distance penalty.  
* **Tiled Matmul:** The inner loop reads A\[0\]\[0\], does a few other operations inside the tile, and reads A\[0\]\[0\] again almost immediately. The temporal gap is tiny. bytedmd\_opt knows it can be kept safely in Rank 1\.

When bytedmd\_opt sees the shortened reuse distances in the tiled trace, it mathematically simulates an omniscient oracle that automatically "copies" the data into the fastest physical ranks exactly when needed, giving you the massive cost reduction *even if you never explicitly wrote the copies into the trace.*

### **The Grand Hierarchy of Bounds**

Let $T\_{naive}$ be the raw naive trace, and $T\_{tiled}$ be the raw tiled trace.

Let $T'\_{tiled}$ be the trace where an algorithm designer has explicitly inserted optimal DMA copies (like your copy\_space\_dmd heuristic or by explicitly rewriting the code).

The math strictly guarantees the following inequalities:

**1\. The Static Blindness:**

$$ C\_{\\text{static\\\_opt\\\_lb}}(T\_{naive}) \== C\_{\\text{static\\\_opt\\\_lb}}(T\_{tiled}) $$  
*(Static allocators cannot benefit from loop reordering without explicit copies).*

**2\. The Benefit of Copies:**

$$ C\_{\\text{static\\\_opt\\\_lb}}(T'\_{tiled}) \\ll C\_{\\text{static\\\_opt\\\_lb}}(T\_{tiled}) $$  
*(Inserting copies allows the static LP to act like a dynamic cache, violently lowering the bound).*

**3\. The Absolute Dynamic Floor:**

$$ C\_{\\text{bytedmd\\\_opt}}(T\_{tiled}) \\le C\_{\\text{static\\\_opt\\\_lb}}(T'\_{tiled}) $$  
*(No matter how clever the algorithm designer is at inserting manual DMA copies, they can NEVER beat an omniscient dynamic oracle running on the raw tiled trace).*

### **Summary**

If you want to know the absolute minimum cost an algorithm designer could achieve by perfectly copying/caching the hot parts of an algorithm into L1 memory, **look directly at bytedmd\_opt.** It natively understands the temporal locality of your loop nests and tells you exactly how cheap the algorithm can get without needing to manually insert the copies into your code\!