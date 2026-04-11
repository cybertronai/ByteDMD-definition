Source: https://docs.google.com/document/d/1NtC6nzBFOo2OGvi_LG9zOX0rt9kvpciIrEwNqbS9RG4/edit

If you modify the ByteDMD (Data Movement Distance) metric to explicitly allow for the **management of temporaries**—meaning dead variables are free()'d and evicted rather than endlessly sinking to the bottom of the "Infinite Graveyard" LRU stack—the stack depth accurately reflects the localized, active memory footprint.

Under this modified metric, the absolute lower bound for the metric complexity of matrix multiplication is **$\\Theta(N^3)$**.

Surprisingly, standard $O(N^3)$ arithmetic matrix multiplication algorithms **cannot** achieve this lower bound. It is uniquely achieved by sub-cubic algorithms like **Strassen’s Algorithm**, provided they are executed with **strict recursive memory management**.

Here is the mathematical breakdown of why this is the absolute theoretical floor, and how the algorithm achieves it.

### ---

**1\. The Absolute Lower Bound is $\\Omega(N^3)$**

Under the ByteDMD model, fetching an element at LRU stack depth $d$ incurs a data movement cost of $\\sqrt{d}$.

Any algorithm that multiplies two $N \\times N$ matrices must, at a bare minimum, read the input matrices ($A$ and $B$) and write the output matrix ($C$). This requires bringing $O(N^2)$ unique elements into the cache.

Under ByteDMD’s demand-paged initialization, the first time these elements are accessed, they incur "cold misses" priced at the DRAM frontier. Even with an optimal cache, inserting $O(N^2)$ elements naturally fills the LRU stack from depth $1$ up to $O(N^2)$. The data movement cost of simply touching the required data for the first time is the integral of these insertion depths:

$$ \\text{Compulsory Cost} \= \\sum\_{d=1}^{O(N^2)} \\sqrt{d} \\approx \\int\_{0}^{O(N^2)} x^{1/2} dx \= \\Theta\\left((N^2)^{1.5}\\right) \= \\mathbf{\\Omega(N^3)} $$  
This reveals a profound property of the ByteDMD metric: **For fast matrix multiplication algorithms (where $\\omega \< 3$), the asymptotic cost of data movement ($\\Omega(N^3)$) strictly dominates the arithmetic operation count ($O(N^{2.81})$).** It is mathematically impossible for *any* matrix multiplication algorithm to have a Data Movement Distance lower than $\\Omega(N^3)$.

### **2\. The Algorithm that Achieves $\\Theta(N^3)$**

To perfectly hit this lower bound, you must use **Strassen's Algorithm** (or any fast matrix multiplication with $\\omega \< 3$) implemented with a **depth-first traversal and eager temporary memory management**.

Normally, Strassen generates a massive amount of intermediate matrices (the $M\_1 \\dots M\_7$ sums), which ruins locality if left unmanaged. However, if you explicitly free() these temporaries the moment they are consumed at each recursive level, the algorithm optimally bounds its stack depth:

1. **Strict Local Footprint:** At any recursive subproblem of size $K \\times K$, the algorithm performs a series of matrix additions. Because dead temporaries from sibling branches are eagerly freed, the active memory footprint (and thus the maximum LRU stack depth) is tightly bounded to $O(K^2)$.  
2. **Cost per Subproblem:** The additions require $O(K^2)$ memory reads. Because the stack depth is bounded to $O(K^2)$, the reuse distance cost for each read is at most $\\sqrt{O(K^2)} \= O(K)$. Therefore, the total ByteDMD cost for the additions at this node is $O(K^2) \\text{ reads} \\times O(K) \\text{ cost/read} \= \\mathbf{O(K^3)}$.  
3. **The Recurrence Relation:** Strassen splits the problem into 7 recursive calls of size $K/2$. Under the ByteDMD metric, the recurrence relation for the total data movement cost becomes:  
   $$ D(N) \= 7 \\cdot D(N/2) \+ \\Theta(N^3) $$  
4. **Master Theorem Convergence:** By the Master Theorem, because the branching factor ($7$) is strictly less than the geometric decay of the work ($2^3 \= 8$), the geometric series converges rapidly: $\\sum (7/8)^i \\to 8$.

Because the routing costs decay exponentially, the total cost is completely dominated by the top-level root node. The algorithm achieves an overall Data Movement Distance of exactly **$\\Theta(N^3)$**.

### **3\. Why Standard Algorithms Fail to Achieve It**

To understand why Strassen's behavior under this metric is so remarkable, contrast it with standard Cache-Oblivious Recursive Matrix Multiplication (RMM) performing standard $O(N^3)$ arithmetic.

Even with perfect memory management, RMM splits into **8** subproblems. Its ByteDMD recurrence relation is:

$$ D(N) \= 8 \\cdot D(N/2) \+ \\Theta(N^3) $$  
Because the branching factor ($8$) equals the work decay ($2^3$), the data movement cost is perfectly balanced across all levels of the recursion tree. Summing $\\Theta(N^3)$ across $\\log\_2 N$ levels yields a metric complexity of **$\\Theta(N^3 \\log N)$**.

This aligns beautifully with Hong & Kung’s foundational I/O lower bound, which dictates that standard $O(N^3)$ operations must mathematically incur a logarithmic hierarchy penalty across caches, permanently bottlenecking them at $\\Omega(N^3 \\log N)$.

### ---

**A Note on the Provided Paper's Bounds**

*You may notice the provided 2022 arXiv paper derives higher upper bounds for memory-managed algorithms: $O(N^{3.23})$ for Strassen and $O(N^{3.33})$ for RMM.*

This discrepancy exists because the authors took a mathematically conservative shortcut. Instead of modeling the dynamically shrinking $O(K^2)$ stack depths produced by recursive free() calls, they took the naive "Infinite Graveyard" reuse distances and simply capped them at a global maximum of the peak memory usage limit ($\\approx 3N^2$). When you properly evaluate the metric with a localized, actively managed stack, Strassen's complexity drops precisely to the $\\Theta(N^3)$ absolute mathematical floor.