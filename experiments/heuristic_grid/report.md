# Heuristic Grid for ByteDMD-Style Metrics

This experiment compares a concrete no-free-compaction 2D cost against several fast heuristics on a small suite of workloads.

Every traced metric cell finished under 1.070 seconds on this run.

## Algorithms

| Algorithm | Workload | Implementation |
| --- | --- | --- |
| Matvec | 32x32 by 32 | row-wise matrix-vector baseline |
| Vecmat | 32 by 32x32 | column-oriented access order |
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
| FFT (Iterative) | N=32 | iterative radix-2 Cooley-Tukey |
| FFT (Recursive) | N=32 | recursive radix-2 Cooley-Tukey |
| Stencil (Naive) | 32x32, one sweep | row-major Jacobi stencil |
| Stencil (Recursive) | 32x32, one sweep, leaf=8 | tile-recursive Jacobi stencil |
| Regular Attention | N=32, d=4 | materializes the full score matrix |
| Flash Attention | N=32, d=4, Bq=8, Bk=4 | double-tiled Q/KV blocks with snake KV order |

## Measures

- `Manual-2D`: the concrete tombstone/no-compaction 2D cost used as the target.
- `ByteDMD-classic`: graveyard model with no reclamation.
- `ByteDMD-live`: aggressive live-only compaction.
- `Reads×sqrt(Peak)`: `reads * ceil(sqrt(peak_live))`, a bandwidth-times-footprint proxy.
- `Reads`: total tracked reads.
- `Peak live slots`: peak active footprint under the live policy.
- `FLOPs`: arithmetic work count.

Attention uses proxy `max`, `exp`, and reciprocal operators with the same read arity as the real kernels, so the table focuses on data movement rather than numerical fidelity.

## Results Grid

| Algorithm | Manual-2D | ByteDMD-classic | ByteDMD-live | Reads×sqrt(Peak) | Reads | Peak live slots | FLOPs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Matvec | 47,951 | 62,694 | 46,926 | 137,088 | 4,032 | 1,090 | 2,016 |
| Vecmat | 43,860 | 59,331 | 42,795 | 137,088 | 4,032 | 1,090 | 2,016 |
| Transpose (Naive) | 40,447 | 40,447 | 40,447 | 47,104 | 1,024 | 2,048 | 1,024 |
| Transpose (Blocked) | 39,806 | 39,806 | 39,806 | 47,104 | 1,024 | 2,048 | 1,024 |
| Transpose (Recursive) | 39,737 | 39,737 | 39,737 | 47,104 | 1,024 | 2,048 | 1,024 |
| Row Scan | 274,427 | 325,675 | 270,334 | 532,415 | 8,191 | 4,098 | 4,095 |
| Column Scan | 235,441 | 294,101 | 231,311 | 532,415 | 8,191 | 4,098 | 4,095 |
| Naive Matmul | 121,869 | 178,324 | 117,935 | 444,416 | 15,872 | 770 | 7,936 |
| Tiled Matmul | 96,306 | 143,280 | 88,687 | 444,416 | 15,872 | 771 | 7,936 |
| Recursive Matmul | 106,395 | 154,251 | 95,462 | 476,160 | 15,872 | 896 | 7,936 |
| Recursive In-Place (Lex) | 96,130 | 162,049 | 91,212 | 458,752 | 16,384 | 770 | 7,936 |
| Recursive In-Place (Gray) | 89,378 | 155,454 | 86,402 | 458,752 | 16,384 | 770 | 7,936 |
| Strassen | 250,051 | 353,207 | 204,752 | 1,068,970 | 30,542 | 1,194 | 15,271 |
| FFT (Iterative) | 1,773 | 2,139 | 1,691 | 3,888 | 432 | 67 | 240 |
| FFT (Recursive) | 1,522 | 1,708 | 1,366 | 4,320 | 432 | 97 | 240 |
| Stencil (Naive) | 73,287 | 94,490 | 68,807 | 378,304 | 8,224 | 2,049 | 5,524 |
| Stencil (Recursive) | 66,583 | 86,431 | 62,206 | 378,304 | 8,224 | 2,049 | 5,524 |
| Regular Attention | 320,318 | 474,581 | 303,850 | 2,072,640 | 40,640 | 2,565 | 20,320 |
| Flash Attention | 234,485 | 335,704 | 197,629 | 1,218,240 | 45,120 | 723 | 22,560 |

## Heuristic Ranking Against Manual-2D

| Heuristic | Spearman rho | Scaled MAPE |
| --- | --- | --- |
| ByteDMD-live | 0.996 | 4.7% |
| ByteDMD-classic | 0.977 | 12.5% |
| Reads×sqrt(Peak) | 0.961 | 33.6% |
| FLOPs | 0.818 | 42.6% |
| Reads | 0.799 | 43.7% |
| Peak live slots | 0.342 | 99.3% |

## Runtime

| Algorithm | Max traced cell (s) | Total traced time (s) |
| --- | --- | --- |
| Matvec | 0.101 | 0.225 |
| Vecmat | 0.092 | 0.208 |
| Transpose (Naive) | 0.116 | 0.291 |
| Transpose (Blocked) | 0.112 | 0.279 |
| Transpose (Recursive) | 0.111 | 0.278 |
| Row Scan | 1.070 | 2.699 |
| Column Scan | 0.926 | 2.420 |
| Naive Matmul | 0.207 | 0.386 |
| Tiled Matmul | 0.181 | 0.306 |
| Recursive Matmul | 0.204 | 0.360 |
| Recursive In-Place (Lex) | 0.245 | 0.387 |
| Recursive In-Place (Gray) | 0.239 | 0.376 |
| Strassen | 0.589 | 0.949 |
| FFT (Iterative) | 0.002 | 0.004 |
| FFT (Recursive) | 0.001 | 0.004 |
| Stencil (Naive) | 0.168 | 0.410 |
| Stencil (Recursive) | 0.164 | 0.417 |
| Regular Attention | 0.923 | 1.661 |
| Flash Attention | 0.491 | 0.744 |

Run the experiment with:

```bash
uv run experiments/heuristic_grid/run_experiment.py
```