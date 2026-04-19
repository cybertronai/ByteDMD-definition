This is an incredibly well-designed benchmarking grid. Evaluating a density-ranked static allocator (space_dmd, mimicking TPU scratchpads) against dynamic hardware-like LRU bounds (bytedmd_live, bytedmd_classic) on a 2D Manhattan-distance model perfectly isolates the different mechanisms of spatial and temporal locality.
To ensure your heuristics properly rank Communication-Avoiding (CA) and cache-efficient schedules against naive ones, you need algorithm pairs that compute the exact same mathematical result but radically reshape the dependency graph.
Here is a curated list of algorithms you should add to stress-test your models, categorized by the specific locality dimension they evaluate:
1. The Missing Baseline: Matrix Transpose (Low-Hanging Fruit)
You actually already wrote transpose_naive, transpose_blocked, and transpose_recursive in your Python file, but they are missing from your markdown table!
* Why it tests your model: Transposition is the purest form of spatial locality testing because it contains zero arithmetic. It completely isolates strided memory penalties. It will immediately show if your heuristics correctly reward the cache-oblivious (recursive) or scratchpad (blocked) layouts over the cache-thrashing naive row-read/col-write variant.
2. Temporal Locality: Time-Skewed Stencils
Your current stencil_naive and stencil_recursive only perform a single sweep. As you noted in your docs, the manual cost for both is exactly identical (99,276) because every cell is touched exactly 5 times regardless of access order. To unlock the true communication-avoiding properties of stencils, you must introduce an outer loop over time ($T$).
* stencil_time_naive(T=8): Sweeps the entire grid from bulk memory 8 times.
* stencil_time_diamond(T=8): (Diamond or Trapezoidal Tiling). Computes a "pyramid" of space-time locally, keeping intermediate time-step cells in the hot L1/scratchpad region before writing them back to bulk memory.
* Why it tests your model: This is the holy grail of stencil CA. It strictly tests whether your dynamic heuristics (bytedmd_live) reward temporal overlap across outer loops, and whether space_dmd can statically pin the overlapping intermediate ghost-zones to low addresses.
3. Graph Dynamic Programming: Floyd-Warshall (APSP)
Your lcs_dp is a great 2D wavefront, but dense $O(N^3)$ graph algorithms stress caches identically to matmul, just with different dependency algebras (min/plus).
* floyd_warshall_naive(V=32): The standard 3-nested loop ($k$, $i$, $j$). Sweeps the $V \times V$ matrix sequentially, thrashing the cache on the inner loop.
* floyd_warshall_recursive(V=32): Kleene’s cache-oblivious divide-and-conquer algorithm. Splits the adjacency matrix into 4 quadrants and makes 8 recursive calls, similar to rmm or recursive_lu.
* Why it tests your model: Because the working set remains strictly $O(V^2)$ while doing $O(V^3)$ work, this provides a brutal stress test for LRU stack-depth. It proves your heuristics can evaluate CA tree-reductions on dense graphs.
4. 1D Vector Fusion: Fused LayerNorm / RMSNorm
flash_attn brilliantly demonstrates 2D online block fusion. You should add 1D vector fusion, which is the other major class of AI memory bottlenecks.
* layernorm_unfused(N=1024): Three sequential loops over the vector: compute mean, compute variance, normalize.
* layernorm_fused(N=1024): Welford’s online algorithm. Keeps running sum/variance in hot scalar registers (addrs 1..4), reads the input once to get stats, and reads it a second time to normalize.
* Why it tests your model: For large vectors, the unfused vector will drop entirely out of the LRU cache between passes. Will bytedmd_live correctly penalize the unfused version and reward the tightly-held scalar accumulators of the fused version?
5. CA-Krylov Solvers: The Matrix Powers Kernel ($s$-step)
Iterative solvers are massively memory-bandwidth bound. The core CA optimization (from Demmel et al.) is avoiding repeated reads of the bulk matrix $A$.


* matrix_powers_naive(A, x, s=4): Naively run matvec $s$ times ($x_1 = Ax_0$, $x_2 = Ax_1$, etc.). Loads the massive matrix $A$ from bulk memory every single step.
* matrix_powers_ca(A, x, s=4): Tile $A$. For a specific block of $A$, compute its contribution to $x^{(1)}$, then $x^{(2)}$, up to $x^{(s)}$ locally before moving to the next block of $A$.
* Why it tests your model: It tests if the heuristic correctly prices the amortization of loading $A$ once to compute $s$ dependent vectors simultaneously.
6. Read vs. Write Asymmetry: Left-Looking vs. Right-Looking
You currently have Right-looking Cholesky. Add cholesky_left_looking.
* Right-Looking: Eagerly updates the entire trailing submatrix (many far-flung, wide-spanning writes, few immediate reads).
* Left-Looking: Delays updates. For column $k$, it pulls all required data from previously factored columns on the left (many far-flung reads, localized writes).
* Why it tests your model: Your cost model states "stores free". Because stores are free but reads pay the Manhattan distance, Left-looking and Right-looking will have drastically different costs in your model, despite identical mathematical operations. This perfectly stress-tests the heuristics' read vs. write asymmetries.
7. Irregular / Indirect Locality: SpMV
Currently, all your traces read continuous dense grids with predictable affine loop indices. Real memory bottlenecks often come from data-dependent memory jumps.
* spmv_csr_banded(n=64): Sparse matrix-vector multiply where col_indices are clustered near the diagonal.
* spmv_csr_random(n=64): Sparse matrix-vector multiply where col_indices are a random Erdős-Rényi graph or Power-Law distribution.
* (Implementation Note: Generate standard Python integers for row_ptr and col_ind ahead of time, and use them to index into your _Tracked values).
* Why it tests your model: bytedmd_live (LRU) will naturally forgive the banded accesses while heavily penalizing the random ones via LRU depth. space_dmd (density ranking) might statically pin the most frequently accessed random nodes, perfectly mimicking a static graph partitioner.
8. Natively Data-Oblivious Sorting: Bitonic Sort
Your Quicksort, Heapsort, and Mergesort rely on branchless "stand-ins" that fake the data dependency.
* bitonic_sort(N=64): A true sorting network. It is natively data-oblivious and parallel.

* Why it tests your model: Its access pattern is an expanding/contracting butterfly network (identical in flavor to the iterative FFT). It provides a highly unique geometric reuse distance signature that contrasts sharply with Mergesort's temporary array allocations and Quicksort's pivoting.