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

# ByteDMD cost of dot product (two-stack model: arg elements at arg
# depths 1..4, intermediates on the geometric stack)
assert bytedmd(dot, (a, b)) == 11
```

## Motivation


Modern architectures spend more energy moving data than doing arithmetic, making FLOP counts an outdated cost metric. Bill Dally ([ACM Opinion](https://cacm.acm.org/opinion/on-the-model-of-computation-point/)) proposed penalizing data movement based on 2D spatial distance to the processor. To avoid manual spatial mapping, Ding and Smith ([Beyond Time Complexity, 2022](https://arxiv.org/abs/2203.02536)) automated this via Data Movement Distance (DMD): a rule treating memory as an LRU stack where reading a byte at depth $d$ costs $\sqrt{d}$, modeling a cache laid out in 2D.

To avoid floating point issues, we round up to the nearest integer.

![ByteDMD](docs/ceil_figure.svg)

This rounding corresponds to routing wire length on a 2D grid with LRU stack arranged in the following order.

![ByteDMD](docs/manhattan_figure.svg)

## Computation Model

An idealized processor operates directly on an element-level LRU stack. **Computations and writes are free; only memory reads incur a cost.**

- **Two stacks:** Reads are priced against one of two stacks, each with the same $\lceil\sqrt{d}\rceil$ depth-cost shape:
  - *Argument stack* (read-only): holds the input arguments, packed left-to-right. The first argument sits at depth 1, the second at depth 2, and so on — all input elements are live and addressable from the start.
  - *Geometric stack* (read-write): holds intermediates produced during execution, ordered from least recently used (bottom) to most recently used (top). Depth is measured in bytes from the top (topmost byte = depth 1). Multi-byte scalars are treated as contiguous blocks of bytes.
- **Argument promotion:** The **first** read of an argument is priced against its depth on the argument stack; that read then promotes the argument onto the top of the geometric stack, as if it had just been produced. Every **subsequent** read of that argument is priced against the geometric stack like any other intermediate.
- **Read Cost:** Reading a byte at depth $d$ on either stack costs $\lceil\sqrt{d}\rceil$.
- **Simultaneous pricing:** All inputs to an instruction are priced against the stack state *before* any LRU bumping or argument promotion. This guarantees commutativity: `Cost(a+b) == Cost(b+a)`.
- **Only live contribute to depth of the geometric stack:** Any value that's dead (no longer used) is immediately removed from the geometric stack and remaining elements slide up to close the gap. This models an optimal compiler that keeps the stack clamped to the active working set. The argument stack is fixed and does not compact — arguments that have already been promoted no longer contribute depth on it.
- **Output epilogue:** At the end of execution, every element of the return value is read once from the geometric stack, modelling the final pass that writes the result to the caller's buffer. The write is free; the read is priced.

### Instruction Semantics

See [Instruction Set](docs/instruction_set.md) for the complete list of supported instructions.

For an instruction with inputs $x_1, \dots, x_m$ and outputs $y_1, \dots, y_n$ with $m\ge 1, n\ge 0$

1. **Price reads:** Evaluate $\sum C(x_j)$ simultaneously against the stack state *before* the instruction begins. All inputs see the same pre-instruction snapshot. Repeated inputs are charged per occurrence at the same depth (e.g., `a + a` charges `⌈√d⌉` twice where `d` is `a`'s pre-instruction depth).
2. **Update LRU:** Batch-move unique inputs to the top of the stack in read order. `b + c` and `c + b` yield the same cost (commutativity) but may differ in final stack order.
3. **Push outputs:** Allocate new output blocks and push them to the top at zero cost.

## Example Walkthrough

Consider the following function with three scalar arguments:

```python
def my_add(a, b, c):
    return (a + b) + c
```

**1. Initial state (left = top, right = bottom)**
Arguments are packed left to right on the argument stack; the geometric stack starts empty:
```text
arg stack:  [a, b, c]    ← a at depth 1, b at depth 2, c at depth 3
geom stack: []
```

**2. First operation: `a + b`**
`a` and `b` are both first reads — priced against the argument stack. Both operands see the pre-instruction snapshot simultaneously:

$$C(a) + C(b) = \lceil\sqrt{1}\rceil + \lceil\sqrt{2}\rceil = 1 + 2 = 3$$

After the op, `a` and `b` are promoted onto the top of the geometric stack, the result `t = a + b` is pushed, and liveness evicts `a` and `b` (their last use just happened):
```text
arg stack:  [a, b, c]    ← a and b remain in their slots but no longer matter
geom stack: [t]          ← t at depth 1
```

**3. Second operation: `t + c`**
`t` is read from the geometric stack at depth 1. `c` is a first read — priced against the argument stack at depth 3:

$$C(t) + C(c) = \lceil\sqrt{1}\rceil + \lceil\sqrt{3}\rceil = 1 + 2 = 3$$

**Total cost:** $3 + 3 = 6$. Trace: `[1, 2, 1, 2]`.


## Inspecting the IR

The tracer also emits a small **intermediate representation** that makes
the two-stack lifecycle explicit. Four event types: `ARG vk @ arg=d`
(place input element `vk` on the argument stack at static depth `d`),
`STORE vk` (push intermediate `vk` on top of the geometric stack),
`READ vk@arg=d` or `READ vk@geom=d` (read priced against whichever stack
currently holds `vk` — first-read-of-an-arg is always `@arg=`, every
subsequent read of the same `vk` is `@geom=` after promotion), and
`OP name(vk@d, …)` (summary of the preceding reads — this is what
incurs cost). Op results are materialized by the `STORE` that
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
ARG   v1 @ arg=1                        # A[0][0] at arg depth 1 (top)
ARG   v2 @ arg=2                        # A[0][1]
ARG   v3 @ arg=3                        # A[1][0]
ARG   v4 @ arg=4                        # A[1][1]
ARG   v5 @ arg=5                        # x[0]
ARG   v6 @ arg=6                        # x[1] — deepest arg slot
  READ v1@arg=1  cost=1                 # A[0][0] first read (on arg stack)
  READ v5@arg=5  cost=3                 # x[0] first read
OP    mul(v1@1, v5@5)  cost=4           # A[0][0]*x[0]; v1, v5 promoted
STORE v7
  READ v2@arg=2  cost=2                 # A[0][1] first read
  READ v6@arg=6  cost=3                 # x[1] first read
OP    mul(v2@2, v6@6)  cost=5           # A[0][1]*x[1]; v2, v6 promoted
STORE v8
  READ v7@geom=3  cost=2                # v7 sunk on geom as v6 promoted
  READ v8@geom=1  cost=1                # v8 still at top
OP    add(v7@3, v8@1)  cost=3           # y0
STORE v9
  READ v3@arg=3  cost=2                 # A[1][0] first read
  READ v5@geom=3  cost=2                # x[0] — hot hit, already promoted!
OP    mul(v3@3, v5@3)  cost=4
STORE v10
  READ v4@arg=4  cost=2                 # A[1][1] first read
  READ v6@geom=3  cost=2                # x[1] — hot hit
OP    mul(v4@4, v6@3)  cost=4
STORE v11
  READ v10@geom=2  cost=2
  READ v11@geom=1  cost=1
OP    add(v10@2, v11@1)  cost=3         # y1
STORE v12
# total cost = 23
```

Two-stack structure visible: the six `ARG` lines place `A` elements at
arg depths 1..4 (first argument, top of arg stack) and `x` at depths
5..6. First reads against the arg stack use the static arg depth; once
an element is read, it is *promoted* onto the top of the geometric
stack, so the second read of `x[0]` and `x[1]` in the second y-sweep
pays a cheap geom-stack depth (`v5@geom=3`, `v6@geom=3`) instead of the
deeper arg position. Liveness still aggressively evicts dead variables
on the geometric stack (the arg stack is fixed).

## ByteDMD benchmarks

See "benchmarks/" folder

### Matrix-vector (4x4 matrix, 4-vector)

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| matvec (i-j) | y = A @ x | 157 |
| vecmat (j-i) | y = x^T @ A | 150 |

### Matrix multiply (4x4)

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| naive matmul (i-j-k) | C = A @ B | 720 |

### microGPT single-token forward pass

Architecture: `vocab=4, embd=4, heads=2, head_dim=2, 1 layer, block_size=4`.
Based on [Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| microGPT (1 layer, embd=4) | single token forward | 3214 |

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

