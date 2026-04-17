# grid — heuristics × algorithms

A table of cache-energy heuristics evaluated across algorithms with
contrasting locality profiles.

## Model

The **2D cache model**: memory forms a Manhattan-distance disc. Distance 1
holds 1 cell, distance 2 holds 3 cells, distance 3 holds 5 cells, …; the
disc of radius r holds r² cells. A load of a cell at address `a` (1-indexed)
costs `ceil(sqrt(a))`. Every heuristic returns an integer cost that
approximates the total energy `∑ ceil(sqrt(addr))` over all memory touches.

## Axes

**Rows — algorithms** (20 total):

| family       | variants                                                     |
|--------------|--------------------------------------------------------------|
| matmul       | naive, tiled, rmm (cache-oblivious), fused_strassen (ZAFS)   |
| attention    | naive, flash (Bk-block online softmax)                       |
| transpose    | naive (column-major read), blocked, recursive (CO)           |
| matvec       | row-major, column-major                                      |
| FFT          | iterative (in-place), recursive (out-of-place)               |
| stencil      | naive row-major sweep, tile-recursive (leaf=8)               |
| convolution  | spatial (single-channel 2D), regular (multi-channel CNN)     |
| FFT-conv     | N-point circular convolution via two FFTs + pointwise + IFFT |
| sort         | mergesort (data-oblivious merge stand-in)                    |
| DP           | LCS dynamic programming (branch-free recurrence)             |

Only `fused_strassen` (Zero-Allocation Fused Strassen / ZAFS) is shown for
the Strassen family — its abstract arithmetic DAG is identical to standard
Strassen's, so the heuristic columns would be the same either way; only
the `manual` column differs (M₁..M₇ are never materialized; their
sub-additions are folded directly into the L1 tile loads).

**Columns — metrics** (ordered lower-envelope → upper-envelope):

| column            | meaning                                                           |
|-------------------|-------------------------------------------------------------------|
| `bytedmd_live`    | LRU with liveness compaction (lower-envelope heuristic)           |
| `manual`          | hand-placed schedule — **gold standard** per-algorithm reference  |
| `bytedmd_classic` | Mattson LRU stack-depth estimate, no liveness (upper-envelope)    |

## Manual placement

Each algorithm in `manual.py` is re-implemented with an explicit
bump-pointer allocator: hot scalars and scratchpads occupy the lowest
addresses, bulk data lives farther out, and recursion uses `push/pop` so
intermediates unwind cleanly. Every memory touch is charged
`ceil(sqrt(addr))`; stores are free (matching `bytedmd_ir.cost`).

**MAC convention** for the matmul family (naive/tiled/rmm/fused_strassen):
accumulator is read once per (i,j) outside the k-loop, then 2 reads (A, B)
per k-iter. This matches the `strassen_trace.py` / `efficient_strassen_trace.py`
reference implementations so the numbers are directly comparable. RMM and
fused_strassen reproduce those scripts' outputs exactly (95,222 and 140,526
respectively at n=16, T=4).

## Run

    ./run_grid.py          # tabulate: writes grid.csv, grid.md
    ./generate_traces.py   # visualize: writes traces/<slug>.png per algorithm

See **[REPORT.md](REPORT.md)** for the narrated per-algorithm writeup —
descriptions, manual allocation strategies, and embedded memory traces.

## Notes

- **Transpose is degenerate in the fixed-address Manhattan model**: every
  A-cell is touched exactly once, so naive, blocked, and recursive all
  produce the same manual cost (22352 at n=32). The *recency-aware*
  heuristics (`bytedmd_classic`, `bytedmd_live`) are what distinguish
  transpose variants — their LRU stacks grow differently under row-major
  vs block-order access. This is the clearest signal that the Manhattan
  model is a **placement** cost, not a **scheduling** cost.
- Hot-slot allocation matters a lot for `matvec`: putting accumulator `y`
  and input `x` at addresses 1..2n cuts manual cost roughly in half
  compared to the default order.
- **Manual can exceed `bytedmd_classic`** for `mergesort` (8,416 vs 4,344)
  and `lcs_dp` (85,929 vs 47,066). When temporaries are many and live
  briefly (mergesort's merge temps) or the working set is one large bulk
  region at high addresses (LCS's (m+1)(n+1) DP table), fixed-placement
  pays the full `⌈√addr⌉` on every access while LRU heuristics amortize
  via recency. Fixed Manhattan is not always an upper envelope.
- **Manual can beat `bytedmd_live`** for `fft_iterative` (788 vs 1,646),
  `fft_conv` (4,253 vs 5,629), and `fused_strassen` (140,526 vs 173,919).
  A tight in-place layout that parks everything in the hot region
  short-circuits what any recency heuristic can model on the abstract
  trace.
