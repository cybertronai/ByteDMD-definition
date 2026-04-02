# A metric of complexity for the 21st century: ByteDMD

[Original Google Doc](https://docs.google.com/document/d/1sj5NqOg6Yqh10bXzGVEF5uIzSjFWAnqqTE75AMng2-s/edit?tab=t.0#heading=h.ujy6ygk7sjmb)

## Motivation

In modern architectures, the energy cost of an algorithm is dominated by data movement, which is not captured by FLOP counts. Our objective is to define a more representative scalar measure.

Bill Dally ([ACM Opinion](https://cacm.acm.org/opinion/on-the-model-of-computation-point/)) proposed a spatial model where bytes live on a 2D grid, penalizing movement based on the Manhattan distance to the processor. Manually specifying spatial $x, y$ coordinates makes it inapplicable to classical algorithms; we address this by introducing an automatic placement strategy.

An elegant solution comes from the "geometric stack" introduced by Ding and Smith ([Beyond Time Complexity, 2022](https://arxiv.org/abs/2203.02536)). Modern processors rely on automated cache replacement policies. Sleator and Tarjan's classic competitive analysis (1985) proved that an automatic LRU (Least Recently Used) policy is strictly competitive with optimal offline caching.

Instead of defining a cache hierarchy (L1/L2/L3) which creates a family of metrics, the geometric stack models an *infinitely layered* LRU stack which provides us with a single natural metric. The cost to access data at depth $d$ is $\sqrt{d}$. This acts as a continuous approximation of Dally's Manhattan distance combined with an LRU cache arranged in a two-dimensional layout.

**The Byte-Level Extension:**
The original DMD metric models abstract variables. Modern algorithms often optimize runtime by downcasting intermediate variables to smaller data types (e.g., `int8` vs `float32`). **ByteDMD** treats scalar entries as contiguous blocks of bytes and tracks distances at the *byte level* to reward this optimization.

![ByteDMD](dmd.png)

## Computation Model

We model an idealized processor with infinite registers, a byte-level LRU stack, and a list of valid instructions. Writes are free, execution is free, while reads incur a cost based on the depth of the target byte in the LRU stack. Each instruction accepts 1 or more inputs and produces 0 or more outputs.

Execution proceeds as follows:
1. **Initialization:** The scheduler loads function arguments onto the LRU stack at 0 cost. The most recently added byte is at the top of the stack (distance 1).
2. **Instruction Execution:** For each instruction, the processor:
   - **Reads inputs into the registers:**  Reading a byte incurs a cost of $\sqrt{d}$ where $d$ is the position in the LRU stack.
   - **Updates LRU:** Inputs are moved to the top of the stack in the order they were read. Multi-byte inputs are moved as a single contiguous block.
   - **Pushes Result:** A new value or values representing the instruction's output are allocated and pushed to the top of the stack at 0 cost.

## Example Walkthrough

### 1. Basic Execution (`int8` Variables)
Consider calling the following function with four 1-byte (`int8`) values:

```python
def myAdd(a, b, c, d):
    return b + c
```

**1. Initialization**

Stack (right is top/MRU): [a, b, c, d]

Distances: d=1, c=2, b=3, a=4

Cost: 0

**2. Instruction Execution** (`b + c`)

- **Reads inputs into the registers:** (Distances are evaluated before the stack updates)
  - Read b: distance 3 $\rightarrow$ cost $\sqrt{3}$
  - Read c: distance 2 $\rightarrow$ cost $\sqrt{2}$
  - Total Read Cost: $\sqrt{3} + \sqrt{2}$

- **Updates LRU:**
  - Move inputs to the top of the stack in order of reading: [a, d, b, c]

- **Pushes Result:**
  - Push the new computed value to the top of the stack (allocated as _r0): [a, d, b, c, _r0]

(This cost can be measured programmatically in Python via `cost, result = measureDMD(myAdd, a, b, c, d)`).

### 2. Multi-Byte Variables

Consider executing `myAdd(a,b,c,d)` where `b` and `c` are 2-byte `int16` integers, while `a` and `d` are 1-byte `int8` integers.

Initial Stack Distances (Top-Down):
- d (1 byte): distance 1
- c (2 bytes): distances 2, 3
- b (2 bytes): distances 4, 5
- a (1 byte): distance 6

Executing `b + c` reads b then c.

Trace of byte distances read: [5, 4, 3, 2].

The total operation cost is $\sqrt{5} + \sqrt{4} + \sqrt{3} + \sqrt{2}$.

### 3. Read Order Matters

Because the LRU stack updates sequentially based on the order variables are read, argument ordering impacts the final stack state:

- `return b + c` $\rightarrow$ final stack: [a, d, b, c, _r0]
- `return c + b` $\rightarrow$ final stack: [a, d, c, b, _r0]

## Notes
If an instruction repeats the argument, the cost is charged multiple times. For example, `A[i][j] * A[i][j]` will load `A[i][j]` twice into the stack, incurring the data movement cost twice.

## Future Work

### Bits vs. Bytes

ByteDMD currently operates on byte boundaries because bit-level operations are not readily exposed in standard high-level runtimes. One could consider a BitDMD metric which works on bit-level boundaries. 

### Extending to Parallel Execution

The current model assumes 0 cost for loading function arguments onto the stack. In multi-processor systems, we may use a different number, which replaces 0 cost with a distance based measure incorporating the physical location of the processor relative to the original data source.
