# A cost model of complexity for the 21st century: ByteDMD

Data movement matters more than FLOPs. Recently accessed bytes can be cached, penalize non-local reads using the following cost model:

$$C=\sum_{b \in bytes} \sqrt{D(b)}$$

where $D(b)$ is the depth of byte $b$ in the LRU stack. Square-root is motivated by VLSI routing cost in 2D.

## Usage

```python
from bytedmd import bytedmd

def dot(a, b):
    return sum(i1*i2 for (i1,i2) in zip(a,b))

a = [0, 1]
b = [2, 3]

# dot product
assert dot(a,b) == 3

# ByteDMD cost of dot product
assert bytedmd(dot, (a, b)) == 13
```

## Motivation


Modern architectures spend more energy moving data than doing arithmetic, making FLOP counts an outdated cost metric. Bill Dally ([ACM Opinion](https://cacm.acm.org/opinion/on-the-model-of-computation-point/)) proposed penalizing data movement based on 2D spatial distance to the processor. To avoid manual spatial mapping, Ding and Smith ([Beyond Time Complexity, 2022](https://arxiv.org/abs/2203.02536)) automated this via Data Movement Distance (DMD): a rule treating memory as an LRU stack where reading a byte at depth $d$ costs $\sqrt{d}$, modeling a cache laid out in 2D.

To avoid floating point issues, we round up to the nearest integer.

![ByteDMD](docs/ceil_figure.svg)

This rounding corresponds to routing wire length on a 2D grid with LRU stack arranged in the following order.

![ByteDMD](docs/manhattan_figure.svg)

The original DMD treats values abstractly. ByteDMD counts accesses at byte level. This rewards algorithms that use smaller data types.

## Computation Model

An idealized processor operates directly on an element-level LRU stack. **Computations and writes are free; only memory reads incur a cost.**

- **Stack State:** Ordered from least recently used (bottom) to most recently used (top). Depth is measured in bytes from the top (topmost byte = depth 1). Multi-byte scalars are treated as a contiguous blocks of bytes.
- **Initialization:** On function entry, arguments are pushed to the top in call order.
- **Read Cost:** Reading a byte at depth $d$ costs $\lceil\sqrt{d}\rceil$.
- **Eviction (liveness analysis):** The tracer performs liveness analysis: after executing the function once to record the full operation sequence, it replays with perfect knowledge of each value's last read. After a value's final read, it is replaced by a **tombstone** at the top of the stack. Tombstones preserve the physical cache footprint: like a real cache line that hasn't been overwritten yet, they still occupy depth. This prevents the "cache teleportation" artifact where streaming through dead data would otherwise shrink the stack and let hot values float up for free. The next STORE recycles exactly one tombstone (the bottom-most, i.e. the oldest dead cache line), keeping the stack anchored at its high-water mark. Eviction and recycling are both free.

### Instruction Semantics

See [Instruction Set](docs/instruction_set.md) for the complete list of supported instructions.

For an instruction with inputs $x_1, \dots, x_m$ and outputs $y_1, \dots, y_n$ with $m\ge 1, n\ge 0$

1. **Price reads:** Evaluate $\sum C(x_j)$ against the stack state *before* the instruction begins. Repeated inputs are charged per occurrence (e.g., `a + a` charges for reading `a` twice).
2. **Update LRU:** Move inputs to the top of the stack sequentially in read order. *(Note: Because of this sequential update, `b + c` vs. `c + b` yields the same cost but different final stack states).*
3. **Push outputs:** Allocate new output blocks and push them to the top at zero cost.

## Example Walkthrough

Consider the following function with four scalar arguments:

```python
def my_add(a, b, c, d):
    return b + c
```

**1. Initial Stack** 
Arguments are pushed in call order `[a, b, c, d]`, yielding element depths from the top:
- `d`: depth 1
- `c`: depth 2
- `b`: depth 3
- `a`: depth 4

**2. Read Cost**  
Inputs are priced simultaneously against the initial stack state:

$$C(b) + C(c) = \lceil\sqrt{3}\rceil + \lceil\sqrt{2}\rceil = 2 + 2 = 4$$

**3. Update Stack**  
Inputs move to the top sequentially in read order (`b`, then `c`), followed by the new `result` being pushed:
```text
[a, d, b, c, result]
```


## Inspecting the IR

The tracer also emits a small **intermediate representation** that makes the
LRU stack lifecycle explicit. Four event types: `STORE k` (recycle one
tombstone if available, then allocate vk on top), `READ k@d` (read vk at
depth d and LRU-bump), `OP name(vk@d, …)` (summary of the preceding reads
— this is what incurs cost), and `DROP k` (tombstone vk after its last
read — determined by liveness analysis). Op results are materialized by the
`STORE` that immediately follows the `OP`.

```python
from bytedmd import inspect_ir, format_ir, bytedmd

def matvec2(A, x):
    y0 = A[0][0]*x[0] + A[0][1]*x[1]
    y1 = A[1][0]*x[0] + A[1][1]*x[1]
    return [y0, y1]

print(format_ir(inspect_ir(matvec2, ([[1,2],[3,4]], [5,6]))))
```

```text
STORE v1                                # A[0][0]
STORE v2                                # A[0][1]
STORE v3                                # A[1][0]
STORE v4                                # A[1][1]
STORE v5                                # x[0]
STORE v6                                # x[1]
  READ v1@6  cost=3                     # v1 bumped to top
DROP  v1                                # last read of v1 → tombstone at top
  READ v5@3  cost=2                     # v5 bumped (tombstone above it)
OP    mul(v1@6, v5@3)  cost=5           # A[0][0]*x[0]
STORE v7                                # recycles the tombstone
  READ v2@6  cost=3                     # v2 bumped to top
DROP  v2                                # last read of v2 → tombstone
  READ v6@4  cost=2                     # v6 bumped
OP    mul(v2@6, v6@4)  cost=5           # A[0][1]*x[1]
STORE v8                                # recycles tombstone
  READ v7@3  cost=2                     # v7 sank below v6, v8
DROP  v7                                # last read → tombstone
  READ v8@2  cost=2
DROP  v8                                # last read → tombstone
OP    add(v7@3, v8@2)  cost=4           # y0
STORE v9                                # recycles one tombstone
  READ v3@6  cost=3                     # depth 6 = 5 live + 1 tombstone
DROP  v3                                # last read → tombstone
  READ v5@5  cost=3
DROP  v5                                # last read → tombstone
OP    mul(v3@6, v5@5)  cost=6
STORE v10                               # recycles one tombstone
  READ v4@6  cost=3
DROP  v4                                # last read → tombstone
  READ v6@6  cost=3
DROP  v6                                # last read → tombstone
OP    mul(v4@6, v6@6)  cost=6
STORE v11                               # recycles one tombstone
  READ v10@4  cost=2
DROP  v10
  READ v11@2  cost=2
DROP  v11
OP    add(v10@4, v11@2)  cost=4         # y1
STORE v12
# total cost = 30
```

The liveness analysis knows exactly when each value's last read occurs.
For example, `v1` (= `A[0][0]`) is only read once, so it is tombstoned
immediately after that read: `DROP v1` fires right after `READ v1@6`. The
tombstone floats to the top of the stack, inflating the depth of the next
read (`v5@3` instead of `v5@2`). The following `STORE v7` recycles this
tombstone, keeping the stack anchored at its high-water mark. This models
how a real cache still holds dead lines until they are overwritten by new
allocations.

## ByteDMD benchmarks

See "benchmarks/" folder

### Matrix-vector (4x4 matrix, 4-vector)

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| matvec (i-j) | y = A @ x | 187 |
| vecmat (j-i) | y = x^T @ A | 181 |

### Matrix multiply (4x4)

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| matmul (i-j-k) | C = A @ B | 830 |
| matmul (i-k-j) | C = A @ B | 865 |
| matmul (snake-j) | C = A @ B | 797 |
| matmul (2x2 tiled) | C = A @ B | 835 |
| matmul (TSP) | C = A @ B | 779 |
| Strassen (leaf=1) | C = A @ B | 1957 |
| Winograd | C = A @ B | 1960 |

### microGPT single-token forward pass

Architecture: `vocab=4, embd=4, heads=2, head_dim=2, 1 layer, block_size=4`.
Based on [Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| microGPT (1 layer, embd=4) | single token forward | 5811 |

# Reports

In-depth reports applying ByteDMD to specific algorithms and design questions:

- [Strassen vs naive matmul](docs/report-strassen-benchmarks/report.md) — at what matrix size does Strassen's recursive algorithm beat naive matmul under ByteDMD? Includes a crossover-point experiment.
- [Modern flash attention vs naive attention](docs/report-modern-flash-attention/report.md) — full sweep across sequence length, head dim, and block size showing flash attention's advantage growing as O(sqrt(N/Bk)) under ByteDMD while FLOPs see no benefit. Uses an optimised tracer (`bytedmd_fast.py`).
- [Antigravity flash attention experiments](docs/report-antigravity-flash-attention/report.md) — alternative flash attention implementations and their ByteDMD costs.
- [Attention benchmark notes](benchmarks/attention_report.md) — the small-scale flash vs naive results that motivated the modern-attention deep dive.

# Python Gotcha's
The tracer implements ByteDMD by wrapping Python objects. This means that the "Instruction Set" of this metric corresponds to Python built-ins, documented under [docs/instruction_set.md](docs/instruction_set.md).

Python behavior means this implementation occasionally doesn't match README semantics and it is possible to escape the wrapping mechanism (local arrays, exception side-channels, identity ops, type introspection, f-strings, math.trunc/ceil/floor on tracked values, etc.). Known failure cases are documented in `test_gotchas.py` — avoid those patterns when writing code you want measured.


[Original Google Doc](https://docs.google.com/document/d/1sj5NqOg6Yqh10bXzGVEF5uIzSjFWAnqqTE75AMng2-s/edit?tab=t.0#heading=h.ujy6ygk7sjmb)

