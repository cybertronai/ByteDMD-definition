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

**Columns — algorithms**:

| family    | variants                                                   |
|-----------|------------------------------------------------------------|
| matmul    | naive, tiled, rmm (cache-oblivious), strassen              |
| attention | naive, flash (Bk-block online softmax)                     |
| transpose | naive (column-major read), blocked, recursive (CO)         |
| matvec    | row-major, column-major                                    |

**Rows — heuristics** (cheapest to most faithful):

| row               | meaning                                                          |
|-------------------|------------------------------------------------------------------|
| `n_loads`         | raw load count — energy if every access cost 1                   |
| `mwis_lower_bound`| interval-LP lower bound on any allocator's cost                   |
| `bytedmd_classic` | Mattson LRU stack-depth estimate (no liveness)                    |
| `bytedmd_live`    | LRU with liveness compaction                                      |
| `min_heap`        | greedy live-bytes allocator — realistic automatic placement       |
| `belady`          | offline-optimal allocator — best automatic placement              |
| `manual`          | hand-placed schedule — **gold standard** per-algorithm reference  |

## Manual placement

Each algorithm in `manual.py` is re-implemented with an explicit
bump-pointer allocator: hot scalars and scratchpads occupy the lowest
addresses, bulk data lives farther out, and recursion uses `push/pop` so
intermediates unwind cleanly. Every memory touch is charged
`ceil(sqrt(addr))`; stores are free (matching `bytedmd_ir.cost`).

## Run

    ./run_grid.py

Outputs: `grid.csv`, `grid.md`, stdout table.

## Notes

- `min_heap == belady` everywhere here: each intermediate in these traces
  is read exactly once, so the greedy online allocator coincides with the
  offline oracle.
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
