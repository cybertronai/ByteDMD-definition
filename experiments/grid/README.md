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

**Rows — algorithms**:

| family    | variants                                           |
|-----------|----------------------------------------------------|
| matmul    | naive, tiled, rmm (cache-oblivious), fused strassen|
| attention | naive, flash (Bk-block online softmax)             |
| transpose | naive (column-major read), blocked, recursive (CO) |
| matvec    | row-major, column-major                            |

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

    ./run_grid.py

Outputs: `grid.csv`, `grid.md`, stdout table.

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
- Skipped for now (each warrants its own design pass): FFT (complex
  arithmetic needs care in the tracer), mergesort (comparisons need a
  trace-safe stand-in), and the cache-oblivious counterparts of Jacobi
  stencil / LCS DP (trapezoidal decomposition, not one-nighters).
