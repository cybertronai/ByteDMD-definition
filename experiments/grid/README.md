# Heuristic Grid for ByteDMD-Style Metrics

This experiment compares a concrete no-free-compaction 2D cost against SpaceDMD and the two abstract ByteDMD heuristics on a small suite of workloads.

Every traced metric cell finished under 4.967 seconds on this run.

## Algorithms

Rows are grouped to follow the dev-branch-style ordering: matmul, attention/vector fusion, matvec/traversal/sparse, FFT, stencil, convolution, sorting/DP/APSP, dense solve, LU, Cholesky, and QR.

| Algorithm | Workload | Implementation |
| --- | --- | --- |
| Naive Matmul | 16x16 | standard i-j-k triple loop |
| Tiled Matmul | 16x16, tile=4 | one explicit blocking level |
| Recursive Matmul | 16x16 | 8-way cache-oblivious recursion |
| Recursive In-Place (Lex) | 16x16 | manual in-place schedule, lexicographic order |
| Recursive In-Place (Gray) | 16x16 | manual in-place schedule, Gray-code order |
| Strassen | 16x16 | leaf size 1 to expose temporary traffic |
| Fused Strassen | 16x16, leaf=8 | zero-allocation virtual sums with direct accumulation into C |
| Naive Attention (d=2) | N=32, d=2 | materializes the full score matrix |
| Flash Attention (Bk=8) | N=32, d=2, Bq=8, Bk=8 | double-tiled Q/KV blocks with wider KV tiles |
| Naive Attention (d=4) | N=32, d=4 | materializes the full score matrix |
| Flash Attention | N=32, d=4, Bq=8, Bk=4 | double-tiled Q/KV blocks with snake KV order |
| LayerNorm (Unfused) | N=1024 | three full vector passes for mean, variance, and normalize |
| LayerNorm (Fused) | N=1024 | Welford statistics fused into two vector passes |
| Matvec | 32x32 by 32 | row-wise matrix-vector baseline |
| Vecmat | 32 by 32x32 | column-oriented access order |
| Matvec Row | 64x64 by 64 | row-major matrix-vector multiply |
| Matvec Column | 64 by 64x64 | column-major vector-matrix multiply |
| Matrix Powers (Naive) | 32x32, s=4 | re-reads A for each successive dense matvec |
| Matrix Powers (CA) | 32x32, s=4, block=4 | row-blocked communication-avoiding proxy for chained matvecs |
| SpMV CSR (Banded) | N=64, bandwidth=3 | sparse matrix-vector multiply with clustered indirect reads |
| SpMV CSR (Random) | N=64, nnz/row=7 | sparse matrix-vector multiply with randomized indirect reads |
| Row Scan | 64x64 | row-major traversal sum |
| Column Scan | 64x64 | column-major traversal sum |
| Transpose (Naive) | 32x32 | direct row-major transpose copy |
| Transpose (Blocked) | 32x32, block=8 | blocked transpose copy |
| Transpose (Recursive) | 32x32, leaf=8 | cache-oblivious recursive transpose |
| FFT (Iterative) | N=1024 | iterative radix-2 Cooley-Tukey |
| FFT (Recursive) | N=1024 | recursive radix-2 Cooley-Tukey |
| Stencil (Naive) | 32x32, one sweep | row-major Jacobi stencil |
| Stencil (Recursive) | 32x32, one sweep, leaf=8 | tile-recursive Jacobi stencil |
| Stencil (Time-Naive) | 32x32, T=4 | four full Jacobi sweeps with fresh intermediate buffers |
| Stencil (Time-Diamond) | 32x32, T=4, block=8 | space-time tiled Jacobi proxy with per-tile halo reuse |
| Spatial Conv (2D, 16x16) | 16x16, kernel=5x5 | same-size zero-padded spatial convolution |
| Spatial Conv (2D, 32x32) | 32x32, kernel=5x5 | same-size zero-padded spatial convolution |
| Regular Conv | 16x16, kernel=3x3, Cin=4, Cout=4 | direct same-size convolution over 4 input/output channels |
| FFT Conv (1D) | N=32 | circular 1D convolution via recursive FFT |
| FFT Conv (2D) | 16x16, kernel=5x5, pad=32x32 | same-size convolution via zero-padded recursive 2D FFT |
| Mergesort | N=64 | top-down mergesort with tracked comparisons |
| Bitonic Sort | N=64 | data-oblivious sorting network with butterfly compare-swaps |
| LCS DP | 32x32 | dynamic programming longest common subsequence |
| Floyd-Warshall (Naive) | V=32 | standard k-i-j all-pairs shortest paths |
| Floyd-Warshall (Recursive) | V=32, leaf=8 | recursive blocked APSP over k/i/j ranges |
| Gaussian Elimination | N=24 | dense solve without pivoting |
| Gauss-Jordan Inverse | N=16 | dense matrix inverse without pivoting |
| LU (No Pivot) | N=24 | Doolittle LU without row swaps |
| LU (Blocked) | N=24, block=4 | panel/TRSM/trailing-update LU without pivoting |
| LU (Recursive) | N=24, leaf=6 | recursive block LU without pivoting |
| LU (Partial Pivot) | N=24 | partial pivoting with row-copy traffic |
| Cholesky | N=24 | lower-triangular Cholesky factorization |
| Cholesky (Blocked) | N=24, block=4 | tile-oriented Cholesky factorization |
| Cholesky (Recursive) | N=24, leaf=6 | recursive block Cholesky factorization |
| Cholesky (Right-Looking) | N=24 | eager trailing-update Cholesky for read/write asymmetry comparison |
| Householder QR | 48x12 | unblocked Householder QR returning R |
| Blocked QR | 48x12, block=4 | panel-blocked Householder QR with delayed trailing updates |
| TSQR | 48x12, leaf_rows=12 | tall-skinny recursive QR returning the final R |

## Measures

- `SpaceDMD`: density-ranked spatial liveness, now with inputs first read from a separate argument stack and only later re-read from the geometric stack.
- `ByteDMD-live`: aggressive live-only compaction on the geometric stack, with the same separate argument-stack first-touch rule.
- `Manual-2D`: hand-scheduled fixed-address implementations with separate scratch and argument/output regions under the 2D `ceil(sqrt(addr))` cost model.
- `ByteDMD-classic`: graveyard model with no reclamation on the geometric stack, again after the first-touch argument-stack read.

All four columns now include a terminal readback of the full returned value, so the table prices both computation and the final result extraction.

SpaceDMD globally ranks geometric-stack variables by access density (`access_count / lifespan`) and then charges each read by that variable's rank among the currently live variables; untouched inputs are priced separately on the argument stack until their first promotion.

## Interpretation Notes

- The trace models now have an explicit first-touch boundary: inputs are priced on an argument stack on first use, then promoted into the geometric stack for later re-use. Manual kernels mirror this with separate scratch and argument/output regions.
- SpaceDMD is intentionally order-blind once data is in the geometric stack: pure permutations with the same multiset of reads, such as `Matvec` vs `Vecmat` or `Row Scan` vs `Column Scan`, can collapse to identical SpaceDMD costs even when `Manual-2D` separates them strongly.
- Single-touch kernels such as the transpose trio are a deliberate failure mode for SpaceDMD. When every cell is read once, the metric collapses to the read count (`n^2` here) rather than the physical `ceil(sqrt(addr))` placement cost.
- The blocked LU and blocked QR rows are panel-update variants, not cosmetic loop chunking. If they still land close to their unblocked counterparts, that should be read as an empirical result rather than a placeholder implementation.
- `Recursive LU` and `Recursive Cholesky` here are copy-based block decompositions built out of `_slice_copy`, triangular solves, and Schur complements. Their costs therefore include explicit materialization traffic and should not be read as in-place communication-optimal factorizations.
- `Matrix Powers (CA)` and `Stencil (Time-Diamond)` are locality proxies rather than full communication-optimal solvers. They preserve the intended block-local dataflow but should be read as stress tests for the heuristics, not numerically tuned production kernels.
- These numbers are implementation-specific to this branch. Comparing them directly to other branches that use different schedules, such as right-looking versus left-looking factorizations or different Strassen fusions, can change the measured locality substantially even when the math is the same.
- SpaceDMD can mis-rank virtual/intermediate-heavy traces such as `Strassen` versus `Fused Strassen`, because it scores density-ranked liveness rather than concrete placement.
- The ranking table has a split verdict: `ByteDMD-live` has the best rank correlation while `ByteDMD-classic` has the best scaled MAPE. In other words, the heuristic that orders rows best is not the same one that matches magnitudes best.

Attention uses proxy `max`, `exp`, and reciprocal operators with the same read arity as the real kernels, so the table focuses on data movement rather than numerical fidelity.

## Results Grid

| Algorithm | SpaceDMD | ByteDMD-live | Manual-2D | ByteDMD-classic |
| --- | --- | --- | --- | --- |
| Naive Matmul | 75,573 | 119,088 | 138,486 | 178,319 |
| Tiled Matmul | 90,626 | 89,445 | 82,574 | 141,169 |
| Recursive Matmul | 104,162 | 95,371 | 102,056 | 149,081 |
| Recursive In-Place (Lex) | 99,017 | 83,216 | 239,777 | 131,240 |
| Recursive In-Place (Gray) | 87,471 | 78,313 | 239,777 | 124,441 |
| Strassen | 129,252 | 202,785 | 210,953 | 341,157 |
| Fused Strassen | 177,235 | 183,970 | 147,360 | 310,614 |
| Naive Attention (d=2) | 121,175 | 182,223 | 419,566 | 273,037 |
| Flash Attention (Bk=8) | 79,155 | 94,106 | 117,651 | 143,791 |
| Naive Attention (d=4) | 227,596 | 298,016 | 804,056 | 462,791 |
| Flash Attention | 183,674 | 195,469 | 286,628 | 317,851 |
| LayerNorm (Unfused) | 81,500 | 153,618 | 122,384 | 206,144 |
| LayerNorm (Fused) | 95,319 | 134,068 | 123,398 | 178,185 |
| Matvec | 29,911 | 36,134 | 29,890 | 39,572 |
| Vecmat | 23,195 | 29,418 | 29,890 | 32,856 |
| Matvec Row | 213,297 | 244,553 | 213,156 | 262,597 |
| Matvec Column | 157,421 | 188,677 | 213,156 | 206,721 |
| Matrix Powers (Naive) | 107,592 | 178,637 | 237,120 | 263,206 |
| Matrix Powers (CA) | 116,600 | 144,919 | 211,932 | 197,497 |
| SpMV CSR (Banded) | 9,602 | 11,410 | 17,869 | 12,270 |
| SpMV CSR (Random) | 11,005 | 14,116 | 18,518 | 15,979 |
| Row Scan | 180,896 | 180,896 | 180,960 | 180,896 |
| Column Scan | 125,024 | 125,024 | 180,960 | 125,024 |
| Transpose (Naive) | 32,980 | 58,989 | 62,813 | 58,989 |
| Transpose (Blocked) | 32,324 | 58,522 | 62,813 | 58,522 |
| Transpose (Recursive) | 32,268 | 58,464 | 62,813 | 58,464 |
| FFT (Iterative) | 266,902 | 410,458 | 467,423 | 602,976 |
| FFT (Recursive) | 136,833 | 213,586 | 467,423 | 311,419 |
| Stencil (Naive) | 64,584 | 91,862 | 142,215 | 118,313 |
| Stencil (Recursive) | 53,942 | 83,339 | 142,215 | 108,221 |
| Stencil (Time-Naive) | 290,346 | 512,219 | 877,793 | 719,588 |
| Stencil (Time-Diamond) | 510,495 | 854,037 | 2,442,625 | 1,441,257 |
| Spatial Conv (2D, 16x16) | 83,083 | 112,862 | 165,557 | 162,940 |
| Spatial Conv (2D, 32x32) | 421,216 | 542,936 | 1,367,491 | 849,599 |
| Regular Conv | 893,512 | 994,160 | 1,958,775 | 1,548,723 |
| FFT Conv (1D) | 3,148 | 5,071 | 5,924 | 6,263 |
| FFT Conv (2D) | 193,770 | 395,662 | 2,812,578 | 644,543 |
| Mergesort | 2,157 | 3,410 | 8,574 | 3,977 |
| Bitonic Sort | 10,283 | 13,095 | 8,842 | 15,899 |
| LCS DP | 23,238 | 30,392 | 138,668 | 30,392 |
| Floyd-Warshall (Naive) | 1,296,932 | 1,600,250 | 2,208,605 | 2,467,170 |
| Floyd-Warshall (Recursive) | 1,289,599 | 1,528,176 | 2,208,605 | 2,371,997 |
| Gaussian Elimination | 144,330 | 174,839 | 149,098 | 264,399 |
| Gauss-Jordan Inverse | 138,241 | 291,004 | 197,639 | 447,568 |
| LU (No Pivot) | 176,646 | 183,748 | 152,609 | 285,388 |
| LU (Blocked) | 168,823 | 205,254 | 152,320 | 292,112 |
| LU (Recursive) | 143,081 | 154,745 | 170,273 | 242,879 |
| LU (Partial Pivot) | 211,252 | 211,749 | 196,063 | 333,458 |
| Cholesky | 58,449 | 60,233 | 103,898 | 87,161 |
| Cholesky (Blocked) | 59,135 | 65,048 | 113,956 | 99,660 |
| Cholesky (Recursive) | 81,590 | 97,705 | 106,249 | 149,742 |
| Cholesky (Right-Looking) | 70,155 | 76,858 | 246,125 | 111,975 |
| Householder QR | 191,725 | 235,920 | 210,511 | 372,359 |
| Blocked QR | 199,250 | 231,744 | 210,847 | 370,913 |
| TSQR | 200,413 | 251,200 | 308,395 | 393,651 |

## Heuristic Ranking Against Manual-2D

| Heuristic | Spearman rho | Scaled MAPE |
| --- | --- | --- |
| SpaceDMD | 0.848 | 51.4% |
| ByteDMD-live | 0.853 | 55.7% |
| ByteDMD-classic | 0.846 | 46.3% |

## Runtime

| Algorithm | Max traced cell (s) | Total traced time (s) |
| --- | --- | --- |
| Naive Matmul | 0.212 | 0.346 |
| Tiled Matmul | 0.181 | 0.293 |
| Recursive Matmul | 0.189 | 0.307 |
| Recursive In-Place (Lex) | 0.141 | 0.255 |
| Recursive In-Place (Gray) | 0.137 | 0.246 |
| Strassen | 0.563 | 0.804 |
| Fused Strassen | 0.488 | 0.699 |
| Naive Attention (d=2) | 0.528 | 0.862 |
| Flash Attention (Bk=8) | 0.146 | 0.273 |
| Naive Attention (d=4) | 0.886 | 1.354 |
| Flash Attention | 0.437 | 0.694 |
| LayerNorm (Unfused) | 0.433 | 0.702 |
| LayerNorm (Fused) | 0.379 | 0.629 |
| Matvec | 0.032 | 0.092 |
| Vecmat | 0.031 | 0.076 |
| Matvec Row | 0.379 | 1.075 |
| Matvec Column | 0.396 | 0.851 |
| Matrix Powers (Naive) | 0.474 | 0.734 |
| Matrix Powers (CA) | 0.188 | 0.394 |
| SpMV CSR (Banded) | 0.011 | 0.032 |
| SpMV CSR (Random) | 0.017 | 0.044 |
| Row Scan | 0.321 | 0.946 |
| Column Scan | 0.351 | 0.708 |
| Transpose (Naive) | 0.082 | 0.186 |
| Transpose (Blocked) | 0.081 | 0.183 |
| Transpose (Recursive) | 0.082 | 0.183 |
| FFT (Iterative) | 1.234 | 1.886 |
| FFT (Recursive) | 0.596 | 1.013 |
| Stencil (Naive) | 0.196 | 0.350 |
| Stencil (Recursive) | 0.195 | 0.417 |
| Stencil (Time-Naive) | 1.576 | 2.924 |
| Stencil (Time-Diamond) | 4.108 | 5.776 |
| Spatial Conv (2D, 16x16) | 0.150 | 0.268 |
| Spatial Conv (2D, 32x32) | 1.630 | 2.304 |
| Regular Conv | 3.080 | 4.235 |
| FFT Conv (1D) | 0.005 | 0.014 |
| FFT Conv (2D) | 1.949 | 3.031 |
| Mergesort | 0.003 | 0.010 |
| Bitonic Sort | 0.009 | 0.024 |
| LCS DP | 0.075 | 0.111 |
| Floyd-Warshall (Naive) | 4.628 | 6.992 |
| Floyd-Warshall (Recursive) | 4.967 | 7.529 |
| Gaussian Elimination | 0.345 | 0.561 |
| Gauss-Jordan Inverse | 0.585 | 0.930 |
| LU (No Pivot) | 0.463 | 0.696 |
| LU (Blocked) | 0.468 | 0.844 |
| LU (Recursive) | 0.381 | 0.593 |
| LU (Partial Pivot) | 0.516 | 0.775 |
| Cholesky | 0.081 | 0.151 |
| Cholesky (Blocked) | 0.113 | 0.210 |
| Cholesky (Recursive) | 0.187 | 0.313 |
| Cholesky (Right-Looking) | 0.119 | 0.231 |
| Householder QR | 0.473 | 0.742 |
| Blocked QR | 0.493 | 0.759 |
| TSQR | 0.434 | 0.722 |

Run the experiment with:

```bash
uv run experiments/grid/run_experiment.py
```