# grid — heuristics × algorithms

A table of cache-friendliness heuristics evaluated across several algorithms.

## Model

The 2D cache model: memory forms a Manhattan-distance disc. Distance 1 holds
1 cell, distance 2 holds 3 cells, distance 3 holds 5 cells, …; the disc of
radius r holds r² cells. A load of a cell at address `a` (1-indexed) costs
`ceil(sqrt(a))` — the Manhattan distance of the closest unoccupied slot if
the first `a-1` slots are taken.

Every heuristic below takes an abstract trace (L2 LOAD/STORE/OP events) and
returns a scalar cost. Allocator-based heuristics (`min_heap`, `belady`)
assign each variable to a physical slot and sum `ceil(sqrt(addr))` per load
— they model **manually placed** memory and are the reference the cheaper
heuristics try to approximate.

## Axes

Columns (algorithms): `naive_matmul`, `tiled_matmul`, `rmm`, `strassen`,
`naive_attention`, `flash_attention`.

Rows (heuristics):

- `n_loads` — raw load count (FLOP-like baseline)
- `footprint` — distinct variables ever stored (total bytes ever touched)
- `peak_live` — max concurrently-live variables (peak working set)
- `mwis_lower_bound` — LP/interval lower bound on any allocator's cost
- `bytedmd_classic` — Mattson LRU stack distance, no compaction
- `bytedmd_live` — LRU stack distance with liveness compaction
- `min_heap` — greedy live-bytes allocator cost (realistic manual placement)
- `belady` — offline-optimal allocator cost (gold-standard manual placement)

## Run

    ./run_grid.py

Outputs: `grid.csv`, `grid.png` (heatmap), stdout table.

## Notes

- Every algorithm here produces a DAG where each intermediate is read exactly
  once, so the greedy `min_heap` allocator coincides with `belady` — the two
  rows will match in every column.
- `peak_live` catches the flash-attention locality win most dramatically
  (~5× smaller working set than naive attention at N=32) because flash
  processes K/V in blocks and never materializes the full N×N score matrix.
- At the sizes used here, Strassen is *worse* than naive matmul under every
  heuristic — the overhead of 7 recursive temporaries dominates until the
  recursion base gets large.

