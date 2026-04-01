# A metric of complexity for the 21st century: ByteDMD

## Motivation

In modern computational architectures, execution time and energy consumption are overwhelmingly dominated by data movement, making FLOP-count an obsolete heuristic. Our objective is to define a computation model and cost function that realistically measures data locality.

Bill Dally ([ACM Opinion](https://cacm.acm.org/opinion/on-the-model-of-computation-point/)) proposed a spatial model where bytes live on a 2D grid, penalizing movement based on the Manhattan distance to the processor. Manually specifying spatial $x, y$ coordinates is a big step away from classical algorithm designer, so we compromise by using an automatic placement strategy.

An elegant solution comes from the "geometric stack" introduced by Ding and Smith ([Beyond Time Complexity, 2022](https://arxiv.org/abs/2203.02536)). Modern processors rely on automated cache replacement policies. Sleator and Tarjan's classic competitive analysis (1985) proved that an automatic LRU (Least Recently Used) policy is strictly competitive with optimal offline caching.

Instead of defining a rigid cache hierarchy (L1/L2/L3) which creates hardware-specific metrics, the geometric stack models an *infinitely layered* LRU stack. The cost to access data at depth $d$ is $\sqrt{d}$. This acts as a continuous 1D approximation of Dally's Manhattan distance combined with an LRU cache.

**The Byte-Level Extension:**
The original DMD metric models abstract variables. However, modern algorithms frequently optimize runtime by downcasting intermediate variables to smaller data types (e.g., `int8` vs `float32`). **ByteDMD** extends the geometric stack by treating variables as contiguous blocks of bytes and tracking distances strictly at the *byte level*. This mathematically rewards the use of smaller data types.

## Computation Model

We model an idealized processor with infinite registers and a byte-aware LRU stack. Writes are free, but reads incur a cost based on the depth of the bytes being read.

Execution proceeds as follows:
1. **Initialization:** The scheduler sequentially loads function arguments (and flattened array elements) onto the LRU stack at 0 cost. The most recently added byte is at the top of the stack (distance 1).
2. **Instruction Execution:** For each instruction, the processor:
   - **Calculates Cost:** Deduplicates unique input operands. The depth $d$ of *every individual byte* is calculated based on the *current* stack state. Reading a byte incurs a cost of $\sqrt{d}$ (or the squared variant $d = (\sqrt{d})^2$ via `measureDMDSquared`).
   - **Updates LRU:** The accessed variables are moved to the top of the stack in the order they were read. Multi-byte variables are moved as a single contiguous block.
   - **Pushes Result:** A new variable representing the instruction's output (e.g., `_r0`) is allocated and pushed to the top of the stack at 0 cost.

## Example Walkthrough

### 1. Basic Execution (`int8` Variables)
Consider calling the following function with four 1-byte (`int8`) values:

```python
def add(a, b, c, d):
    return b + c
```

**Step 1: Load Arguments**

Stack (right is top/MRU): [a, b, c, d]

Distances: d=1, c=2, b=3, a=4

Cost: 0

**Step 2: Evaluate b + c** (Distances are evaluated before the stack updates)

Read b: distance 3 $\rightarrow$ cost $\sqrt{3}$

Read c: distance 2 $\rightarrow$ cost $\sqrt{2}$

Total Read Cost: $\sqrt{3} + \sqrt{2}$

**Step 3: Update Stack & Write Result**

Move read arguments to the top of the stack in read order: [a, d, b, c]

Push the new computed value (allocated as _r0): [a, d, b, c, _r0]

(This cost can be measured programmatically in Python via `cost, result = measureDMD(add, a, b, c, d)`).

### 2. Multi-Byte Variables (The "Byte" in ByteDMD)

To illustrate why byte-level tracking is powerful, consider executing `b + c` where `b` and `c` are 2-byte `int16` integers, while `a` and `d` are 1-byte `int8` integers.

Initial Stack Distances (Top-Down):
- d (1 byte): distance 1
- c (2 bytes): distances 2, 3
- b (2 bytes): distances 4, 5
- a (1 byte): distance 6

Executing `b + c` reads b then c.

Trace of byte distances read: [5, 4, 3, 2].

The total operation cost is $\sqrt{5} + \sqrt{4} + \sqrt{3} + \sqrt{2}$.

### 3. Read Order Matters

Because the LRU stack updates sequentially based on the order variables are read, instruction ordering natively impacts the final stack state and the cost of subsequent operations:

- `return b + c` $\rightarrow$ final stack: [a, d, b, c, _r0]
- `return c + b` $\rightarrow$ final stack: [a, d, c, b, _r0]

## Future Work

### Bits vs. Bytes

ByteDMD currently operates on byte boundaries because bit-level operations are not readily exposed in standard high-level runtimes. Future iterations could adjust the metric to evaluate at the bit level for greater precision when analyzing extreme quantization models.

### Extending to Parallel Execution

The current model assumes 0 cost for initial argument loading. In multi-processor systems, we may introduce a location-aware scheduler that incurs a load penalty dependent on the physical or networked distribution of the original data source.
