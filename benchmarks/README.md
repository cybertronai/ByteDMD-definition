# ByteDMD Benchmarks

A folder of self-contained benchmark scripts that print ByteDMD cost tables for different algorithm families. Each script imports the regular tracer from `bytedmd.py`, runs a fixed set of algorithms on canonical inputs, and prints a formatted table to stdout. All numbers are reproducible — every script self-verifies via `assert` statements.

## How to run

From the repo root:

```bash
python3 benchmarks/benchmark_linalg.py
python3 benchmarks/benchmark_microgpt.py
python3 benchmarks/benchmark_attention.py
```

## Benchmarks

### `benchmark_linalg.py` — 4x4 linear algebra

Compares nine matrix-vector and matrix-multiply variants on 4x4 inputs. Demonstrates how loop order, tiling, and recursive algorithms (Strassen, Winograd) trade off data movement against arithmetic count. The headline finding is that the TSP-ordered matmul (895) beats every other variant including Winograd (2178) and Strassen (2435), which lose under ByteDMD because their additional intermediate allocations dominate the savings from fewer multiplies.

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------:|
| matvec (i-j) | y = A @ x | 194 |
| vecmat (j-i) | y = x^T @ A | 191 |
| matmul (i-j-k) | C = A @ B | 948 |
| matmul (i-k-j) | C = A @ B | 1016 |
| matmul (snake-j) | C = A @ B | 906 |
| matmul (2x2 tiled) | C = A @ B | 947 |
| matmul (TSP order) | C = A @ B | 895 |
| Strassen (leaf=1) | C = A @ B | 2435 |
| Winograd | C = A @ B | 2178 |

### `benchmark_microgpt.py` — single-token forward pass

Runs one forward pass through a tiny configuration of [Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) (`vocab=4, embd=4, heads=2, head_dim=2, 1 layer, block_size=4`). Exercises the full GPT pipeline (token + position embedding, RMSNorm, multi-head attention, residual connections, MLP, language-model head) at small enough scale to trace in reasonable time.

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------:|
| microGPT (1 layer, embd=4) | single token forward | 7047 |

### `benchmark_attention.py` — naive vs flash attention

Sweeps over `(seq_len, head_dim, flash_block_size)` from N=4 up to N=128, measuring both ByteDMD cost and analytical FLOP count for each configuration. The key finding: under FLOPs the two methods are essentially identical (flash attention does *slightly more* arithmetic due to online softmax rescaling), while under ByteDMD flash attention's advantage grows from 0.94x at N=4 to **3.25x at N=128** as the buried-V-under-N²-attention-matrix penalty grows in the naive version. The full analysis is in [`attention_report.md`](attention_report.md); cached results are in [`attention_results.json`](attention_results.json).

| N | Naive ByteDMD | Flash ByteDMD (best Bk) | Ratio | FLOP Ratio |
|--:|--------------:|------------------------:|------:|-----------:|
| 4 | 1,406 | 1,498 | 0.94x | 0.70x |
| 8 | 7,939 | 6,841 | 1.16x | 0.83x |
| 16 | 46,584 | 32,701 | 1.42x | 0.91x |
| 32 | 293,648 | 163,643 | 1.79x | 0.86x |
| 64 | 1,953,613 | 822,108 | 2.38x | 0.93x |
| 128 | 13,705,802 | 4,221,808 | **3.25x** | 0.96x |

## Files

```
benchmarks/
├── README.md                 (this file)
├── benchmark_linalg.py       linalg cost table
├── benchmark_microgpt.py     microGPT forward pass cost
├── benchmark_attention.py    naive vs flash attention sweep
├── attention_results.json    cached attention sweep output
└── attention_report.md       full attention analysis
```
