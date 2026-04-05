# Flash Attention vs Naive Attention: ByteDMD Analysis

## Summary

Under FLOPs, naive attention and flash attention perform the same core computation (Q @ K^T, softmax, @ V) with identical FLOP counts. Flash attention actually does *slightly more* FLOPs due to online softmax rescaling. This is why FLOP-based analysis sees no benefit from flash attention.

ByteDMD tells a different story. As sequence length grows, **flash attention becomes increasingly cheaper** under ByteDMD because it avoids materializing the full N x N attention matrix, keeping intermediates local on the LRU stack.

## Key Finding

The ByteDMD ratio (naive/flash) **increases with sequence length**, showing flash attention's advantage grows with scale:

| N | d | Block Size | Naive ByteDMD | Flash ByteDMD | ByteDMD ratio | FLOP ratio |
|---|---|-----------|--------------|--------------|---------------|------------|
| 4 | 2 | 2 | 1,740 | 1,970 | 0.88x | 0.70x |
| 4 | 4 | 2 | 3,068 | 3,762 | 0.82x | 0.72x |
| 8 | 2 | 2 | 9,617 | 10,048 | 0.96x | 0.61x |
| 8 | 2 | 4 | 9,617 | 8,652 | **1.11x** | 0.83x |
| 8 | 4 | 2 | 16,572 | 19,884 | 0.83x | 0.64x |
| 8 | 4 | 4 | 16,572 | 17,398 | 0.95x | 0.84x |
| 16 | 2 | 2 | 55,605 | 49,522 | **1.12x** | 0.58x |
| 16 | 2 | 4 | 55,605 | 43,181 | **1.29x** | 0.76x |
| 16 | 4 | 2 | 91,639 | 99,552 | 0.92x | 0.61x |
| 16 | 4 | 4 | 91,639 | 88,181 | **1.04x** | 0.78x |

*ByteDMD ratio > 1.0 means flash attention is cheaper (bold). FLOP ratio < 1.0 means flash uses more FLOPs.*

## Analysis

### Why flash attention wins under ByteDMD at larger N

Naive attention materializes the full N x N score matrix S. Each element of S gets pushed onto the LRU stack. By the time we compute O = S @ V, the elements of V have been buried deep under N^2 intermediate values, making each V read expensive (high LRU depth -> high sqrt(depth) cost).

Flash attention never builds the full S matrix. It processes K/V in blocks of size B, computing partial scores and accumulating into the output. The working set at any point is only O(B * d) intermediates, so V elements stay close on the LRU stack.

### Block size matters

Larger block sizes generally improve flash attention's ByteDMD cost (compare Bk=2 vs Bk=4 at N=16, d=2: ratio goes from 1.12x to 1.29x). However, if the block size equals the sequence length, flash attention degenerates to naive attention.

The optimal block size balances two effects:
- **Too small**: more overhead from online softmax rescaling (alpha, beta computations at each merge)
- **Too large**: the intermediate block attention matrix grows, pushing V elements deeper on the stack

### At small sizes, flash loses

At N=4, flash attention is consistently worse because:
1. The overhead of online softmax rescaling (extra multiplications for alpha, beta) is proportionally large
2. The N x N matrix is only 16 elements -- not large enough to significantly bury V on the stack

This matches real-world experience: flash attention only helps at longer sequence lengths.

### FLOPs cannot distinguish the algorithms

The FLOP ratio is always < 1.0, meaning flash attention does *more* FLOPs than naive attention (due to rescaling). FLOP analysis would incorrectly predict flash attention is always slower. ByteDMD captures the data movement advantage that makes flash attention faster in practice.

## Methodology

- All experiments use `bytedmd()` from the ByteDMD library
- Attention is implemented in pure Python with element-wise operations
- Softmax is approximated with polynomial exp and normalization (same data movement pattern as real softmax)
- `max` is implemented branch-free to satisfy ByteDMD's escape-hatch checker
- Inputs are all-ones matrices (values don't affect ByteDMD cost, only access patterns matter)
- FLOPs are counted analytically

## Reproducing

```bash
python3 benchmarks/benchmark_attention.py
```
