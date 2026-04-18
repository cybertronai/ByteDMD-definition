# Heuristic Grid for ByteDMD-Style Metrics

This experiment compares a concrete no-free-compaction 2D cost against SpaceDMD and the two abstract ByteDMD heuristics on a small suite of workloads.

Every traced metric cell finished under 4.790 seconds on this run.

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

## Trace Diagnostics

These follow the dev-branch style plots for the current `ByteDMD-live` path: each algorithm gets a reuse-distance-per-load scatter plot and a working-set-size-over-time step plot under [`diagnostics/`](./diagnostics/).

A tab-separated summary is also saved as [`diagnostics/diagnostics_summary.tsv`](./diagnostics/diagnostics_summary.tsv).

| Algorithm | Peak live | Max reuse | Median reuse | Working-set plot | Reuse-distance plot |
| --- | --- | --- | --- | --- | --- |
| Naive Matmul | 770 | 768 | 34 | [link](diagnostics/naive-matmul-16_liveset.png) | [link](diagnostics/naive-matmul-16_reuse_distance.png) |
| Tiled Matmul | 771 | 768 | 11 | [link](diagnostics/tiled-matmul-16_liveset.png) | [link](diagnostics/tiled-matmul-16_reuse_distance.png) |
| Recursive Matmul | 896 | 768 | 11 | [link](diagnostics/rmm-16_liveset.png) | [link](diagnostics/rmm-16_reuse_distance.png) |
| Recursive In-Place (Lex) | 770 | 768 | 6 | [link](diagnostics/rmm-lex-16_liveset.png) | [link](diagnostics/rmm-lex-16_reuse_distance.png) |
| Recursive In-Place (Gray) | 770 | 768 | 7 | [link](diagnostics/rmm-gray-16_liveset.png) | [link](diagnostics/rmm-gray-16_reuse_distance.png) |
| Strassen | 1,194 | 1,023 | 13 | [link](diagnostics/strassen-16_liveset.png) | [link](diagnostics/strassen-16_reuse_distance.png) |
| Fused Strassen | 774 | 773 | 8 | [link](diagnostics/fused-strassen-16_liveset.png) | [link](diagnostics/fused-strassen-16_reuse_distance.png) |
| Naive Attention (d=2) | 2,309 | 2,082 | 4 | [link](diagnostics/naive-attention-32x2_liveset.png) | [link](diagnostics/naive-attention-32x2_reuse_distance.png) |
| Flash Attention (Bk=8) | 409 | 343 | 4 | [link](diagnostics/flash-attention-32x2-b8_liveset.png) | [link](diagnostics/flash-attention-32x2-b8_reuse_distance.png) |
| Naive Attention (d=4) | 2,565 | 2,146 | 4 | [link](diagnostics/regular-attention-32x4_liveset.png) | [link](diagnostics/regular-attention-32x4_reuse_distance.png) |
| Flash Attention | 723 | 593 | 5 | [link](diagnostics/flash-attention-32x4_liveset.png) | [link](diagnostics/flash-attention-32x4_reuse_distance.png) |
| LayerNorm (Unfused) | 2,057 | 2,054 | 4 | [link](diagnostics/layernorm-unfused-1024_liveset.png) | [link](diagnostics/layernorm-unfused-1024_reuse_distance.png) |
| LayerNorm (Fused) | 2,058 | 2,055 | 4 | [link](diagnostics/layernorm-fused-1024_liveset.png) | [link](diagnostics/layernorm-fused-1024_reuse_distance.png) |
| Matvec | 1,090 | 1,056 | 25 | [link](diagnostics/matvec-32_liveset.png) | [link](diagnostics/matvec-32_reuse_distance.png) |
| Vecmat | 1,090 | 1,056 | 13 | [link](diagnostics/vecmat-32_liveset.png) | [link](diagnostics/vecmat-32_reuse_distance.png) |
| Matvec Row | 4,226 | 4,160 | 49 | [link](diagnostics/matvec-row-64_liveset.png) | [link](diagnostics/matvec-row-64_reuse_distance.png) |
| Matvec Column | 4,226 | 4,160 | 23 | [link](diagnostics/matvec-col-64_liveset.png) | [link](diagnostics/matvec-col-64_reuse_distance.png) |
| Matrix Powers (Naive) | 1,122 | 1,088 | 65 | [link](diagnostics/matrix-powers-naive-32-s4_liveset.png) | [link](diagnostics/matrix-powers-naive-32-s4_reuse_distance.png) |
| Matrix Powers (CA) | 1,473 | 1,024 | 66 | [link](diagnostics/matrix-powers-ca-32-s4_liveset.png) | [link](diagnostics/matrix-powers-ca-32-s4_reuse_distance.png) |
| SpMV CSR (Banded) | 567 | 560 | 14 | [link](diagnostics/spmv-csr-banded-64_liveset.png) | [link](diagnostics/spmv-csr-banded-64_reuse_distance.png) |
| SpMV CSR (Random) | 579 | 569 | 28 | [link](diagnostics/spmv-csr-random-64_liveset.png) | [link](diagnostics/spmv-csr-random-64_reuse_distance.png) |
| Row Scan | 4,098 | 4,096 | 1 | [link](diagnostics/scan-row-64_liveset.png) | [link](diagnostics/scan-row-64_reuse_distance.png) |
| Column Scan | 4,098 | 4,096 | 1 | [link](diagnostics/scan-column-64_liveset.png) | [link](diagnostics/scan-column-64_reuse_distance.png) |
| Transpose (Naive) | 2,048 | 2,047 | 836 | [link](diagnostics/transpose-naive-32_liveset.png) | [link](diagnostics/transpose-naive-32_reuse_distance.png) |
| Transpose (Blocked) | 2,048 | 2,047 | 821 | [link](diagnostics/transpose-blocked-32_liveset.png) | [link](diagnostics/transpose-blocked-32_reuse_distance.png) |
| Transpose (Recursive) | 2,048 | 2,047 | 832 | [link](diagnostics/transpose-recursive-32_liveset.png) | [link](diagnostics/transpose-recursive-32_reuse_distance.png) |
| FFT (Iterative) | 2,051 | 2,048 | 3 | [link](diagnostics/fft-iterative-1024_liveset.png) | [link](diagnostics/fft-iterative-1024_reuse_distance.png) |
| FFT (Recursive) | 3,073 | 2,559 | 3 | [link](diagnostics/fft-recursive-1024_liveset.png) | [link](diagnostics/fft-recursive-1024_reuse_distance.png) |
| Stencil (Naive) | 2,049 | 2,047 | 9 | [link](diagnostics/jacobi-naive-32_liveset.png) | [link](diagnostics/jacobi-naive-32_reuse_distance.png) |
| Stencil (Recursive) | 2,049 | 2,047 | 9 | [link](diagnostics/jacobi-recursive-32_liveset.png) | [link](diagnostics/jacobi-recursive-32_reuse_distance.png) |
| Stencil (Time-Naive) | 3,074 | 2,048 | 9 | [link](diagnostics/stencil-time-naive-32-t4_liveset.png) | [link](diagnostics/stencil-time-naive-32-t4_reuse_distance.png) |
| Stencil (Time-Diamond) | 2,817 | 2,385 | 9 | [link](diagnostics/stencil-time-diamond-32-t4_liveset.png) | [link](diagnostics/stencil-time-diamond-32-t4_reuse_distance.png) |
| Spatial Conv (2D, 16x16) | 539 | 537 | 33 | [link](diagnostics/conv2d-spatial-16x16-k5_liveset.png) | [link](diagnostics/conv2d-spatial-16x16-k5_reuse_distance.png) |
| Spatial Conv (2D, 32x32) | 2,075 | 2,073 | 34 | [link](diagnostics/spatial-conv-32x32-k5_liveset.png) | [link](diagnostics/spatial-conv-32x32-k5_reuse_distance.png) |
| Regular Conv | 2,194 | 2,192 | 51 | [link](diagnostics/regular-conv-16x16-k3-c4_liveset.png) | [link](diagnostics/regular-conv-16x16-k3-c4_reuse_distance.png) |
| FFT Conv (1D) | 225 | 127 | 3 | [link](diagnostics/fft-conv-32_liveset.png) | [link](diagnostics/fft-conv-32_reuse_distance.png) |
| FFT Conv (2D) | 5,715 | 3,096 | 3 | [link](diagnostics/conv2d-fft-16x16-k5_liveset.png) | [link](diagnostics/conv2d-fft-16x16-k5_reuse_distance.png) |
| Mergesort | 192 | 159 | 3 | [link](diagnostics/mergesort-64_liveset.png) | [link](diagnostics/mergesort-64_reuse_distance.png) |
| Bitonic Sort | 130 | 127 | 61 | [link](diagnostics/bitonic-sort-64_liveset.png) | [link](diagnostics/bitonic-sort-64_reuse_distance.png) |
| LCS DP | 1,155 | 1,152 | 4 | [link](diagnostics/lcs-dp-32x32_liveset.png) | [link](diagnostics/lcs-dp-32x32_reuse_distance.png) |
| Floyd-Warshall (Naive) | 2,050 | 2,047 | 4 | [link](diagnostics/floyd-warshall-naive-32_liveset.png) | [link](diagnostics/floyd-warshall-naive-32_reuse_distance.png) |
| Floyd-Warshall (Recursive) | 2,050 | 2,047 | 4 | [link](diagnostics/floyd-warshall-recursive-32_liveset.png) | [link](diagnostics/floyd-warshall-recursive-32_reuse_distance.png) |
| Gaussian Elimination | 1,228 | 1,200 | 25 | [link](diagnostics/gaussian-elimination-24_liveset.png) | [link](diagnostics/gaussian-elimination-24_reuse_distance.png) |
| Gauss-Jordan Inverse | 772 | 767 | 44 | [link](diagnostics/gauss-jordan-inverse-16_liveset.png) | [link](diagnostics/gauss-jordan-inverse-16_reuse_distance.png) |
| LU (No Pivot) | 1,431 | 1,152 | 29 | [link](diagnostics/lu-no-pivot-24_liveset.png) | [link](diagnostics/lu-no-pivot-24_reuse_distance.png) |
| LU (Blocked) | 2,710 | 1,409 | 10 | [link](diagnostics/blocked-lu-24_liveset.png) | [link](diagnostics/blocked-lu-24_reuse_distance.png) |
| LU (Recursive) | 2,115 | 1,325 | 15 | [link](diagnostics/recursive-lu-24_liveset.png) | [link](diagnostics/recursive-lu-24_reuse_distance.png) |
| LU (Partial Pivot) | 1,431 | 1,151 | 37 | [link](diagnostics/lu-partial-pivot-24_liveset.png) | [link](diagnostics/lu-partial-pivot-24_reuse_distance.png) |
| Cholesky | 604 | 599 | 4 | [link](diagnostics/cholesky-24_liveset.png) | [link](diagnostics/cholesky-24_reuse_distance.png) |
| Cholesky (Blocked) | 1,266 | 1,151 | 9 | [link](diagnostics/blocked-cholesky-24_liveset.png) | [link](diagnostics/blocked-cholesky-24_reuse_distance.png) |
| Cholesky (Recursive) | 1,467 | 905 | 13 | [link](diagnostics/recursive-cholesky-24_liveset.png) | [link](diagnostics/recursive-cholesky-24_reuse_distance.png) |
| Cholesky (Right-Looking) | 1,454 | 1,151 | 13 | [link](diagnostics/cholesky-right-looking-24_liveset.png) | [link](diagnostics/cholesky-right-looking-24_reuse_distance.png) |
| Householder QR | 1,256 | 1,201 | 4 | [link](diagnostics/householder-qr-48x12_liveset.png) | [link](diagnostics/householder-qr-48x12_reuse_distance.png) |
| Blocked QR | 1,349 | 1,338 | 4 | [link](diagnostics/blocked-qr-48x12_liveset.png) | [link](diagnostics/blocked-qr-48x12_reuse_distance.png) |
| TSQR | 1,208 | 719 | 4 | [link](diagnostics/tsqr-48x12_liveset.png) | [link](diagnostics/tsqr-48x12_reuse_distance.png) |

## Runtime

| Algorithm | Max traced cell (s) | Total traced time (s) |
| --- | --- | --- |
| Naive Matmul | 0.225 | 0.372 |
| Tiled Matmul | 0.187 | 0.302 |
| Recursive Matmul | 0.191 | 0.319 |
| Recursive In-Place (Lex) | 0.148 | 0.266 |
| Recursive In-Place (Gray) | 0.147 | 0.260 |
| Strassen | 0.546 | 0.791 |
| Fused Strassen | 0.548 | 0.795 |
| Naive Attention (d=2) | 0.532 | 0.858 |
| Flash Attention (Bk=8) | 0.155 | 0.287 |
| Naive Attention (d=4) | 0.873 | 1.343 |
| Flash Attention | 0.512 | 0.776 |
| LayerNorm (Unfused) | 0.453 | 0.735 |
| LayerNorm (Fused) | 0.395 | 0.657 |
| Matvec | 0.035 | 0.100 |
| Vecmat | 0.032 | 0.079 |
| Matvec Row | 0.420 | 1.185 |
| Matvec Column | 0.425 | 0.939 |
| Matrix Powers (Naive) | 0.505 | 0.794 |
| Matrix Powers (CA) | 0.192 | 0.388 |
| SpMV CSR (Banded) | 0.009 | 0.027 |
| SpMV CSR (Random) | 0.014 | 0.037 |
| Row Scan | 0.355 | 1.040 |
| Column Scan | 0.418 | 0.786 |
| Transpose (Naive) | 0.094 | 0.203 |
| Transpose (Blocked) | 0.117 | 0.269 |
| Transpose (Recursive) | 0.088 | 0.204 |
| FFT (Iterative) | 1.354 | 2.068 |
| FFT (Recursive) | 0.644 | 1.075 |
| Stencil (Naive) | 0.197 | 0.352 |
| Stencil (Recursive) | 0.190 | 0.400 |
| Stencil (Time-Naive) | 1.729 | 3.154 |
| Stencil (Time-Diamond) | 4.287 | 6.105 |
| Spatial Conv (2D, 16x16) | 0.153 | 0.272 |
| Spatial Conv (2D, 32x32) | 1.804 | 2.496 |
| Regular Conv | 2.947 | 4.090 |
| FFT Conv (1D) | 0.005 | 0.014 |
| FFT Conv (2D) | 1.865 | 2.922 |
| Mergesort | 0.004 | 0.010 |
| Bitonic Sort | 0.010 | 0.027 |
| LCS DP | 0.074 | 0.113 |
| Floyd-Warshall (Naive) | 4.790 | 7.218 |
| Floyd-Warshall (Recursive) | 4.744 | 7.157 |
| Gaussian Elimination | 0.369 | 0.599 |
| Gauss-Jordan Inverse | 0.633 | 1.000 |
| LU (No Pivot) | 0.502 | 0.742 |
| LU (Blocked) | 0.525 | 0.913 |
| LU (Recursive) | 0.427 | 0.643 |
| LU (Partial Pivot) | 0.555 | 0.822 |
| Cholesky | 0.082 | 0.154 |
| Cholesky (Blocked) | 0.115 | 0.214 |
| Cholesky (Recursive) | 0.192 | 0.324 |
| Cholesky (Right-Looking) | 0.115 | 0.223 |
| Householder QR | 0.503 | 0.792 |
| Blocked QR | 0.498 | 0.768 |
| TSQR | 0.451 | 0.752 |

Run the experiment with:

```bash
uv run experiments/grid/run_experiment.py
```