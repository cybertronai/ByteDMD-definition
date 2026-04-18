# Heuristic Grid for ByteDMD-Style Metrics

This experiment compares a concrete no-free-compaction 2D cost against SpaceDMD and the two abstract ByteDMD heuristics on a small suite of workloads.

Every traced metric cell finished under 2.980 seconds on this run.

## Algorithms

Rows are grouped to follow the dev-branch grid ordering: matmul, attention, matvec/traversal, FFT, stencil, convolution, sorting/DP, dense solve, LU, Cholesky, and QR.

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
| Matvec | 32x32 by 32 | row-wise matrix-vector baseline |
| Vecmat | 32 by 32x32 | column-oriented access order |
| Matvec Row | 64x64 by 64 | row-major matrix-vector multiply |
| Matvec Column | 64 by 64x64 | column-major vector-matrix multiply |
| Row Scan | 64x64 | row-major traversal sum |
| Column Scan | 64x64 | column-major traversal sum |
| Transpose (Naive) | 32x32 | direct row-major transpose copy |
| Transpose (Blocked) | 32x32, block=8 | blocked transpose copy |
| Transpose (Recursive) | 32x32, leaf=8 | cache-oblivious recursive transpose |
| FFT (Iterative) | N=1024 | iterative radix-2 Cooley-Tukey |
| FFT (Recursive) | N=1024 | recursive radix-2 Cooley-Tukey |
| Stencil (Naive) | 32x32, one sweep | row-major Jacobi stencil |
| Stencil (Recursive) | 32x32, one sweep, leaf=8 | tile-recursive Jacobi stencil |
| Spatial Conv (2D, 16x16) | 16x16, kernel=5x5 | same-size zero-padded spatial convolution |
| Spatial Conv (2D, 32x32) | 32x32, kernel=5x5 | same-size zero-padded spatial convolution |
| Regular Conv | 16x16, kernel=3x3, Cin=4, Cout=4 | direct same-size convolution over 4 input/output channels |
| FFT Conv (1D) | N=32 | circular 1D convolution via recursive FFT |
| FFT Conv (2D) | 16x16, kernel=5x5, pad=32x32 | same-size convolution via zero-padded recursive 2D FFT |
| Mergesort | N=64 | top-down mergesort with tracked comparisons |
| LCS DP | 32x32 | dynamic programming longest common subsequence |
| Gaussian Elimination | N=24 | dense solve without pivoting |
| Gauss-Jordan Inverse | N=16 | dense matrix inverse without pivoting |
| LU (No Pivot) | N=24 | Doolittle LU without row swaps |
| LU (Blocked) | N=24, block=4 | panel/TRSM/trailing-update LU without pivoting |
| LU (Recursive) | N=24, leaf=6 | recursive block LU without pivoting |
| LU (Partial Pivot) | N=24 | partial pivoting with row-copy traffic |
| Cholesky | N=24 | lower-triangular Cholesky factorization |
| Cholesky (Blocked) | N=24, block=4 | tile-oriented Cholesky factorization |
| Cholesky (Recursive) | N=24, leaf=6 | recursive block Cholesky factorization |
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
| Matvec | 29,911 | 36,134 | 29,890 | 39,572 |
| Vecmat | 23,195 | 29,418 | 29,890 | 32,856 |
| Matvec Row | 213,297 | 244,553 | 213,156 | 262,597 |
| Matvec Column | 157,421 | 188,677 | 213,156 | 206,721 |
| Row Scan | 180,896 | 180,896 | 180,960 | 180,896 |
| Column Scan | 125,024 | 125,024 | 180,960 | 125,024 |
| Transpose (Naive) | 32,980 | 58,989 | 62,813 | 58,989 |
| Transpose (Blocked) | 32,324 | 58,522 | 62,813 | 58,522 |
| Transpose (Recursive) | 32,268 | 58,464 | 62,813 | 58,464 |
| FFT (Iterative) | 266,902 | 410,458 | 467,423 | 602,976 |
| FFT (Recursive) | 136,833 | 213,586 | 467,423 | 311,419 |
| Stencil (Naive) | 64,584 | 91,862 | 142,215 | 118,313 |
| Stencil (Recursive) | 53,942 | 83,339 | 142,215 | 108,221 |
| Spatial Conv (2D, 16x16) | 83,083 | 112,862 | 165,557 | 162,940 |
| Spatial Conv (2D, 32x32) | 421,216 | 542,936 | 1,367,491 | 849,599 |
| Regular Conv | 893,512 | 994,160 | 1,958,775 | 1,548,723 |
| FFT Conv (1D) | 3,148 | 5,071 | 5,924 | 6,263 |
| FFT Conv (2D) | 193,770 | 395,662 | 2,812,578 | 644,543 |
| Mergesort | 2,157 | 3,410 | 8,574 | 3,977 |
| LCS DP | 23,238 | 30,392 | 138,668 | 30,392 |
| Gaussian Elimination | 144,330 | 174,839 | 149,098 | 264,399 |
| Gauss-Jordan Inverse | 138,241 | 291,004 | 197,639 | 447,568 |
| LU (No Pivot) | 176,646 | 183,748 | 152,609 | 285,388 |
| LU (Blocked) | 168,823 | 205,254 | 152,320 | 292,112 |
| LU (Recursive) | 143,081 | 154,745 | 170,273 | 242,879 |
| LU (Partial Pivot) | 211,252 | 211,749 | 196,063 | 333,458 |
| Cholesky | 58,449 | 60,233 | 103,898 | 87,161 |
| Cholesky (Blocked) | 59,135 | 65,048 | 113,956 | 99,660 |
| Cholesky (Recursive) | 81,590 | 97,705 | 106,249 | 149,742 |
| Householder QR | 191,725 | 235,920 | 210,511 | 372,359 |
| Blocked QR | 199,250 | 231,744 | 210,847 | 370,913 |
| TSQR | 200,413 | 251,200 | 308,395 | 393,651 |

## Heuristic Ranking Against Manual-2D

| Heuristic | Spearman rho | Scaled MAPE |
| --- | --- | --- |
| SpaceDMD | 0.824 | 66.8% |
| ByteDMD-live | 0.844 | 77.3% |
| ByteDMD-classic | 0.829 | 66.7% |

## Runtime

| Algorithm | Max traced cell (s) | Total traced time (s) |
| --- | --- | --- |
| Naive Matmul | 0.233 | 0.377 |
| Tiled Matmul | 0.194 | 0.307 |
| Recursive Matmul | 0.199 | 0.320 |
| Recursive In-Place (Lex) | 0.145 | 0.259 |
| Recursive In-Place (Gray) | 0.139 | 0.249 |
| Strassen | 0.547 | 0.790 |
| Fused Strassen | 0.499 | 0.712 |
| Naive Attention (d=2) | 0.538 | 0.889 |
| Flash Attention (Bk=8) | 0.172 | 0.306 |
| Naive Attention (d=4) | 0.911 | 1.417 |
| Flash Attention | 0.498 | 0.823 |
| Matvec | 0.032 | 0.092 |
| Vecmat | 0.031 | 0.077 |
| Matvec Row | 0.386 | 1.092 |
| Matvec Column | 0.396 | 0.858 |
| Row Scan | 0.322 | 0.956 |
| Column Scan | 0.363 | 0.723 |
| Transpose (Naive) | 0.079 | 0.185 |
| Transpose (Blocked) | 0.082 | 0.186 |
| Transpose (Recursive) | 0.083 | 0.186 |
| FFT (Iterative) | 1.344 | 2.001 |
| FFT (Recursive) | 0.596 | 1.012 |
| Stencil (Naive) | 0.201 | 0.362 |
| Stencil (Recursive) | 0.193 | 0.404 |
| Spatial Conv (2D, 16x16) | 0.152 | 0.271 |
| Spatial Conv (2D, 32x32) | 1.673 | 2.319 |
| Regular Conv | 2.980 | 4.133 |
| FFT Conv (1D) | 0.005 | 0.015 |
| FFT Conv (2D) | 1.999 | 3.100 |
| Mergesort | 0.004 | 0.010 |
| LCS DP | 0.074 | 0.110 |
| Gaussian Elimination | 0.349 | 0.567 |
| Gauss-Jordan Inverse | 0.609 | 0.951 |
| LU (No Pivot) | 0.440 | 0.670 |
| LU (Blocked) | 0.464 | 0.841 |
| LU (Recursive) | 0.392 | 0.605 |
| LU (Partial Pivot) | 0.526 | 0.808 |
| Cholesky | 0.088 | 0.163 |
| Cholesky (Blocked) | 0.120 | 0.222 |
| Cholesky (Recursive) | 0.192 | 0.325 |
| Householder QR | 0.478 | 0.752 |
| Blocked QR | 0.503 | 0.773 |
| TSQR | 0.437 | 0.726 |

Run the experiment with:

```bash
uv run experiments/grid/run_experiment.py
```