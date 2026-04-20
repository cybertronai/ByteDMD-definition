# Tiled Matrix Multiplication: Visualizing Cache Locality

This experiment visualizes why tiled (blocked) matrix multiplication is more energy-efficient than the naive algorithm by plotting their memory access patterns.

## The Key Insight

By plotting **Time** (operation sequence) on the X-axis and **1D Physical Memory Address** on the Y-axis, cache locality becomes visually obvious:

- **Cache-friendly algorithm**: small vertical footprint at any given time — the CPU repeatedly reads a small neighborhood of addresses that fits in L1/L2 cache.
- **Cache-hostile algorithm**: large vertical sweeps — the CPU touches distant addresses, causing constant cache misses.

## Results

![Naive vs Tiled Access Patterns](matmul_access_pattern.svg)

### Naive (top panel)

Matrix B forms massive diagonal sweeps across its entire memory space (0 to N^2). The time gap between reusing the first row of B is O(N^2) operations — by then the cache has completely overwritten it. Almost every read of B is a slow RAM fetch.

### Tiled (bottom panel)

Instead of sweeping the entire matrix, accesses form tight, localized blocks ("clouds"). For hundreds of operations, the CPU is locked into a tiny vertical band of memory. Because the T x T block easily fits in cache, the CPU executes the math at near-lightspeed before moving to the next block.

## Running

```bash
uv run --script visualize_tiling.py
```

Generates `matmul_access_pattern.svg` (N=64, T=16, ~262k access points).
