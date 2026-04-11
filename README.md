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
assert bytedmd(dot, (a, b)) == 13  # demand-paged: cold misses + hot hits
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
- **Initialization (demand paging):** The LRU stack starts **empty**. Arguments are assigned tracking IDs but are not pushed onto the stack. A value enters the stack only upon its first read, priced as a "cold miss" at the DRAM frontier (a monotonically increasing counter tracking total unique elements ever fetched). This eliminates bias from Python's argument ordering.
- **Read Cost:** Reading a byte at depth $d$ costs $\lceil\sqrt{d}\rceil$. For a cold miss, $d$ = current stack size + 1 (just outside the known universe).
- **Simultaneous pricing:** All inputs to an instruction are priced against the stack state *before* any LRU bumping. This guarantees commutativity: `Cost(a+b) == Cost(b+a)`.
- **Natural LRU aging:** Dead variables are not tombstoned or evicted. They simply sink toward the bottom of the stack as newer values are accessed, modeling a fully-associative cache with natural aging.

### Instruction Semantics

See [Instruction Set](docs/instruction_set.md) for the complete list of supported instructions.

For an instruction with inputs $x_1, \dots, x_m$ and outputs $y_1, \dots, y_n$ with $m\ge 1, n\ge 0$

1. **Price reads:** Evaluate $\sum C(x_j)$ simultaneously against the stack state *before* the instruction begins. All inputs see the same pre-instruction snapshot. Repeated inputs are charged per occurrence at the same depth (e.g., `a + a` charges `⌈√d⌉` twice where `d` is `a`'s pre-instruction depth).
2. **Update LRU:** Batch-move unique inputs to the top of the stack in read order. `b + c` and `c + b` yield the same cost (commutativity) but may differ in final stack order.
3. **Push outputs:** Allocate new output blocks and push them to the top at zero cost.

## Example Walkthrough

Consider the following function with four scalar arguments:

```python
def my_add(a, b, c, d):
    return b + c
```

**1. Initial Stack** 
The stack starts empty (demand paging). Arguments `a, b, c, d` have tracking IDs but are not on the stack yet.

**2. Read Cost**  
`b + c` reads `b` and `c` simultaneously. Both are cold misses:
- `b`: DRAM frontier → 1, depth = 1 (frontier + stack size 0). Enters stack.
- `c`: DRAM frontier → 2, depth = 2 (frontier + stack size 1). Enters stack.

$$C(b) + C(c) = \lceil\sqrt{1}\rceil + \lceil\sqrt{2}\rceil = 1 + 2 = 3$$

**3. Update Stack**  
After the instruction, `b` and `c` are LRU-bumped and `result` is pushed:
```text
[b, c, result]
```


## Inspecting the IR

The tracer also emits a small **intermediate representation** that makes the
LRU stack lifecycle explicit. Three event types: `STORE k` (allocate vk on
top), `READ k@d` (read vk at depth d — cold miss if first access — and
LRU-bump), `OP name(vk@d, …)` (summary of the preceding reads — this is
what incurs cost). Op results are materialized by the `STORE` that
immediately follows the `OP`.

```python
from bytedmd import inspect_ir, format_ir, bytedmd

def matvec2(A, x):
    y0 = A[0][0]*x[0] + A[0][1]*x[1]
    y1 = A[1][0]*x[0] + A[1][1]*x[1]
    return [y0, y1]

print(format_ir(inspect_ir(matvec2, ([[1,2],[3,4]], [5,6]))))
```

```text
  READ v1@1  cost=1                     # cold miss: A[0][0] (stack was empty)
  READ v5@2  cost=2                     # cold miss: x[0] (priced simultaneously)
OP    mul(v1@1, v5@2)  cost=3           # A[0][0]*x[0]
STORE v7
  READ v2@4  cost=2                     # cold miss: A[0][1], depth = stack_size+1
  READ v6@5  cost=3                     # cold miss: x[1]
OP    mul(v2@4, v6@5)  cost=5           # A[0][1]*x[1]
STORE v8
  READ v7@4  cost=2                     # hot hit: v7 sank as v2, v6 entered
  READ v8@1  cost=1                     # hot hit: v8 still at top
OP    add(v7@4, v8@1)  cost=3           # y0
STORE v9
  READ v3@8  cost=3                     # cold miss: A[1][0], depth = 7+1
  READ v5@6  cost=3                     # hot hit: x[0] still on stack
OP    mul(v3@8, v5@6)  cost=6
STORE v10
  READ v4@10  cost=4                    # cold miss: A[1][1], depth = 9+1
  READ v6@7  cost=3                     # hot hit: x[1]
OP    mul(v4@10, v6@7)  cost=7
STORE v11
  READ v10@4  cost=2
  READ v11@1  cost=1
OP    add(v10@4, v11@1)  cost=3         # y1
STORE v12
# total cost = 27
```

Note the demand-paged initialization: no `STORE` events at the top — values
enter the stack only on their first read as cold misses. A cold miss is
priced at `len(stack) + 1` — just outside the current known universe.
Subsequent reads of the same value are hot hits priced at their LRU stack
depth. Dead variables (like `v1` after its single read) are never evicted —
they simply sink to the bottom as newer values push above them (the
"Infinite Graveyard" model).

## ByteDMD benchmarks

See "benchmarks/" folder

### Matrix-vector (4x4 matrix, 4-vector)

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| matvec (i-j) | y = A @ x | 177 |
| vecmat (j-i) | y = x^T @ A | 177 |

### Matrix multiply (4x4)

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| matmul (i-j-k) | C = A @ B | 921 |
| matmul (i-k-j) | C = A @ B | 987 |
| matmul (snake-j) | C = A @ B | 879 |
| matmul (2x2 tiled) | C = A @ B | 914 |
| matmul (TSP) | C = A @ B | 779 |
| Strassen (leaf=1) | C = A @ B | 1957 |
| Winograd | C = A @ B | 1960 |

### microGPT single-token forward pass

Architecture: `vocab=4, embd=4, heads=2, head_dim=2, 1 layer, block_size=4`.
Based on [Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| microGPT (1 layer, embd=4) | single token forward | 6383 |

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

