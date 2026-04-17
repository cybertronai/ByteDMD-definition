# Heuristic Grid for ByteDMD-Style Metrics

This experiment compares a concrete no-free-compaction 2D cost against SpaceDMD and the two abstract ByteDMD heuristics on a small suite of workloads.

Every traced metric cell finished under 2.832 seconds on this run.

## Algorithms

| Algorithm | Workload | Implementation |
| --- | --- | --- |
| Matvec | 32x32 by 32 | row-wise matrix-vector baseline |
| Vecmat | 32 by 32x32 | column-oriented access order |
| Matvec Row | 64x64 by 64 | row-major matrix-vector multiply |
| Matvec Column | 64 by 64x64 | column-major vector-matrix multiply |
| Transpose (Naive) | 32x32 | direct row-major transpose copy |
| Transpose (Blocked) | 32x32, block=8 | blocked transpose copy |
| Transpose (Recursive) | 32x32, leaf=8 | cache-oblivious recursive transpose |
| Row Scan | 64x64 | row-major traversal sum |
| Column Scan | 64x64 | column-major traversal sum |
| Naive Matmul | 16x16 | standard i-j-k triple loop |
| Tiled Matmul | 16x16, tile=4 | one explicit blocking level |
| Recursive Matmul | 16x16 | 8-way cache-oblivious recursion |
| Recursive In-Place (Lex) | 16x16 | manual in-place schedule, lexicographic order |
| Recursive In-Place (Gray) | 16x16 | manual in-place schedule, Gray-code order |
| Strassen | 16x16 | leaf size 1 to expose temporary traffic |
| Fused Strassen | 16x16, leaf=8 | zero-allocation virtual sums with direct accumulation into C |
| Gaussian Elimination | N=24 | dense solve without pivoting |
| Gauss-Jordan Inverse | N=16 | dense matrix inverse without pivoting |
| LU (No Pivot) | N=24 | Doolittle LU without row swaps |
| LU (Blocked) | N=24, block=4 | tile-oriented LU without pivoting |
| LU (Recursive) | N=24, leaf=6 | recursive block LU without pivoting |
| LU (Partial Pivot) | N=24 | partial pivoting with row-copy traffic |
| Cholesky | N=24 | lower-triangular Cholesky factorization |
| Cholesky (Blocked) | N=24, block=4 | tile-oriented Cholesky factorization |
| Cholesky (Recursive) | N=24, leaf=6 | recursive block Cholesky factorization |
| Householder QR | 48x12 | unblocked Householder QR returning R |
| Blocked QR | 48x12, block=4 | column-blocked Householder QR returning R |
| TSQR | 48x12, leaf_rows=12 | tall-skinny recursive QR returning the final R |
| FFT (Iterative) | N=1024 | iterative radix-2 Cooley-Tukey |
| FFT (Recursive) | N=1024 | recursive radix-2 Cooley-Tukey |
| 2D Convolution (Spatial) | 16x16, kernel=5x5 | same-size zero-padded spatial convolution |
| Spatial Conv | 32x32, kernel=5x5 | same-size zero-padded spatial convolution |
| Regular Conv | 16x16, kernel=3x3, Cin=4, Cout=4 | direct same-size convolution over 4 input/output channels |
| 2D Convolution (FFT) | 16x16, kernel=5x5, pad=32x32 | same-size convolution via zero-padded recursive 2D FFT |
| FFT Conv | N=32 | circular 1D convolution via recursive FFT |
| Stencil (Naive) | 32x32, one sweep | row-major Jacobi stencil |
| Stencil (Recursive) | 32x32, one sweep, leaf=8 | tile-recursive Jacobi stencil |
| Regular Attention | N=32, d=4 | materializes the full score matrix |
| Naive Attention | N=32, d=2 | materializes the full score matrix |
| Flash Attention | N=32, d=4, Bq=8, Bk=4 | double-tiled Q/KV blocks with snake KV order |
| Flash Attention (Bk=8) | N=32, d=2, Bq=8, Bk=8 | double-tiled Q/KV blocks with wider KV tiles |
| Mergesort | N=64 | top-down mergesort with tracked comparisons |
| LCS DP | 32x32 | dynamic programming longest common subsequence |

## Measures

- `SpaceDMD`: density-ranked spatial liveness, following the April 17, 2026 gist heuristic for ahead-of-time static pinning.
- `ByteDMD-live`: aggressive live-only compaction.
- `Manual-2D`: hand-scheduled fixed-address implementations under the 2D `ceil(sqrt(addr))` cost model.
- `ByteDMD-classic`: graveyard model with no reclamation.

The `Manual-2D` column uses explicit fixed-address kernels rather than the tombstone allocator. Traversal-only variants can collapse when they read the same fixed addresses exactly once; scratch-heavy kernels separate much more strongly.

SpaceDMD globally ranks variables by access density (`access_count / lifespan`) and then charges each read by that variable's rank among the currently live variables.

Attention uses proxy `max`, `exp`, and reciprocal operators with the same read arity as the real kernels, so the table focuses on data movement rather than numerical fidelity.

## Results Grid

| Algorithm | SpaceDMD | ByteDMD-live | Manual-2D | ByteDMD-classic |
| --- | --- | --- | --- | --- |
| Matvec | 13,432 | 46,926 | 28,834 | 62,694 |
| Vecmat | 13,432 | 42,795 | 28,834 | 59,331 |
| Matvec Row | 72,525 | 331,413 | 208,996 | 450,939 |
| Matvec Column | 72,525 | 295,841 | 208,996 | 422,866 |
| Transpose (Naive) | 1,024 | 40,447 | 22,352 | 40,447 |
| Transpose (Blocked) | 1,024 | 39,806 | 22,352 | 39,806 |
| Transpose (Recursive) | 1,024 | 39,737 | 22,352 | 39,737 |
| Row Scan | 12,286 | 270,334 | 180,959 | 325,675 |
| Column Scan | 12,286 | 231,311 | 180,959 | 294,101 |
| Naive Matmul | 89,314 | 117,935 | 131,888 | 178,324 |
| Tiled Matmul | 98,001 | 88,687 | 75,740 | 143,280 |
| Recursive Matmul | 107,846 | 95,462 | 95,222 | 154,251 |
| Recursive In-Place (Lex) | 102,392 | 91,212 | 233,184 | 162,049 |
| Recursive In-Place (Gray) | 100,802 | 86,402 | 233,184 | 155,454 |
| Strassen | 133,981 | 204,752 | 186,661 | 353,207 |
| Fused Strassen | 180,690 | 183,684 | 140,526 | 313,340 |
| Gaussian Elimination | 135,387 | 182,828 | 148,474 | 272,313 |
| Gauss-Jordan Inverse | 130,771 | 289,170 | 192,294 | 442,482 |
| LU (No Pivot) | 114,163 | 168,584 | 132,511 | 239,981 |
| LU (Blocked) | 114,164 | 168,585 | 139,796 | 239,982 |
| LU (Recursive) | 99,038 | 147,668 | 150,175 | 214,757 |
| LU (Partial Pivot) | 158,915 | 196,126 | 176,369 | 296,079 |
| Cholesky | 39,149 | 57,473 | 94,351 | 83,916 |
| Cholesky (Blocked) | 39,374 | 67,613 | 104,409 | 93,979 |
| Cholesky (Recursive) | 68,960 | 97,702 | 96,702 | 141,392 |
| Householder QR | 184,272 | 242,342 | 208,880 | 366,748 |
| Blocked QR | 184,272 | 242,342 | 208,880 | 366,748 |
| TSQR | 194,425 | 257,309 | 306,764 | 409,914 |
| FFT (Iterative) | 234,873 | 400,915 | 426,962 | 582,525 |
| FFT (Recursive) | 108,940 | 204,043 | 426,962 | 338,459 |
| 2D Convolution (Spatial) | 81,089 | 109,710 | 160,267 | 157,550 |
| Spatial Conv | 395,661 | 519,078 | 1,326,692 | 815,690 |
| Regular Conv | 893,095 | 971,149 | 1,916,445 | 1,514,678 |
| 2D Convolution (FFT) | 188,853 | 394,161 | 2,847,679 | 642,020 |
| FFT Conv | 3,085 | 5,184 | 6,915 | 6,766 |
| Stencil (Naive) | 30,921 | 68,807 | 101,754 | 94,490 |
| Stencil (Recursive) | 28,204 | 62,206 | 101,754 | 86,431 |
| Regular Attention | 242,252 | 303,850 | 413,470 | 474,581 |
| Naive Attention | 132,071 | 185,350 | 203,582 | 277,567 |
| Flash Attention | 205,036 | 197,629 | 283,297 | 335,704 |
| Flash Attention (Bk=8) | 86,433 | 95,000 | 116,392 | 150,088 |
| Mergesort | 1,645 | 3,180 | 4,493 | 3,911 |
| LCS DP | 28,568 | 32,265 | 85,929 | 32,486 |

## Heuristic Ranking Against Manual-2D

| Heuristic | Spearman rho | Scaled MAPE |
| --- | --- | --- |
| ByteDMD-classic | 0.913 | 82.3% |
| ByteDMD-live | 0.879 | 94.2% |
| SpaceDMD | 0.786 | 61.5% |

## Runtime

| Algorithm | Max traced cell (s) | Total traced time (s) |
| --- | --- | --- |
| Matvec | 0.102 | 0.172 |
| Vecmat | 0.092 | 0.162 |
| Matvec Row | 1.530 | 2.414 |
| Matvec Column | 1.443 | 2.378 |
| Transpose (Naive) | 0.103 | 0.178 |
| Transpose (Blocked) | 0.102 | 0.176 |
| Transpose (Recursive) | 0.100 | 0.173 |
| Row Scan | 1.113 | 1.876 |
| Column Scan | 0.945 | 1.734 |
| Naive Matmul | 0.229 | 0.366 |
| Tiled Matmul | 0.212 | 0.326 |
| Recursive Matmul | 0.224 | 0.355 |
| Recursive In-Place (Lex) | 0.265 | 0.389 |
| Recursive In-Place (Gray) | 0.258 | 0.377 |
| Strassen | 0.615 | 0.856 |
| Fused Strassen | 0.533 | 0.762 |
| Gaussian Elimination | 0.387 | 0.618 |
| Gauss-Jordan Inverse | 0.616 | 0.943 |
| LU (No Pivot) | 0.285 | 0.508 |
| LU (Blocked) | 0.321 | 0.540 |
| LU (Recursive) | 0.296 | 0.504 |
| LU (Partial Pivot) | 0.431 | 0.676 |
| Cholesky | 0.085 | 0.150 |
| Cholesky (Blocked) | 0.103 | 0.202 |
| Cholesky (Recursive) | 0.169 | 0.297 |
| Householder QR | 0.454 | 0.732 |
| Blocked QR | 0.445 | 0.719 |
| TSQR | 0.533 | 0.814 |
| FFT (Iterative) | 1.245 | 1.908 |
| FFT (Recursive) | 0.875 | 1.290 |
| 2D Convolution (Spatial) | 0.135 | 0.243 |
| Spatial Conv | 1.496 | 2.104 |
| Regular Conv | 2.832 | 3.883 |
| 2D Convolution (FFT) | 1.951 | 2.981 |
| FFT Conv | 0.005 | 0.014 |
| Stencil (Naive) | 0.176 | 0.323 |
| Stencil (Recursive) | 0.172 | 0.326 |
| Regular Attention | 0.951 | 1.429 |
| Naive Attention | 0.570 | 0.904 |
| Flash Attention | 0.513 | 0.747 |
| Flash Attention (Bk=8) | 0.162 | 0.276 |
| Mergesort | 0.003 | 0.008 |
| LCS DP | 0.078 | 0.113 |

Run the experiment with:

```bash
uv run experiments/grid/run_experiment.py
```