# A cost model of complexity for the 21st century: ByteDMD

**TLDR;**
- The memory wall makes reducing data movement more important than reducing arithmetic.
- ByteDMD is a scalar cost function that measures data movement instead of FLOPs.
- Memory is modeled as an LRU stack where accessing an $L$-byte cache scales $\propto \sqrt{L}$ (a 2D VLSI heuristic).

## Motivation

Modern architectures spend more energy moving data than doing arithmetic, making FLOP count an outdated cost metric. ByteDMD is intended as a more representative scalar replacement.

Bill Dally ([ACM Opinion](https://cacm.acm.org/opinion/on-the-model-of-computation-point/)) proposed penalizing data movement based on 2D spatial distance to the processor. Because manually assigning spatial coordinates is awkward, Ding and Smith ([Beyond Time Complexity, 2022](https://arxiv.org/abs/2203.02536)) proposed Data Movement Distance (DMD): an automatic rule treating memory as an LRU stack where reading depth $d$ costs $\sqrt{d}$ (conceptually modeling a cache laid out in 2D concentric circles).

The original DMD treats values abstractly. ByteDMD refines the model to the byte level: a $k$-byte value occupies $k$ consecutive stack positions. This rewards algorithms that prefer smaller data types like `int8` over `float32`.

![ByteDMD](dmd.png)

## Computation model

We model an idealized processor operating directly on a byte-level LRU stack. Computations and writes are free. The only charged operation is reading existing bytes from the stack.

### Stack state

The stack is ordered from least recently used (bottom) to most recently used (top). Depth is counted in bytes from the top, so the topmost byte has depth 1. Each live value $x$ occupies a contiguous block of $|x|$ byte positions.

### Initialization

On function entry, arguments are pushed to the top of the stack in call order. 

### Cost of reading a value

To avoid floating-point arithmetic, the read cost is the ceiling of the square root, denoted $\lceil\sqrt{d}\rceil$.

If a value $x$ occupies byte depths $D(x)$, reading it costs:

$$
C(x) = \sum_{d \in D(x)} \lceil\sqrt{d}\rceil.
$$

For example, a 1-byte value at depth 3 costs $\lceil\sqrt{3}\rceil = 2$. A 2-byte value spanning depths 4 and 5 costs $\lceil\sqrt{4}\rceil + \lceil\sqrt{5}\rceil = 2 + 3 = 5$.

### Instruction semantics

Consider an instruction with input list $x_1, \ldots, x_m$ and outputs $y_1, \ldots, y_n$:

1. **Price the reads:** Charge each input against the stack state at the **start** of the instruction. Repeated inputs are charged per occurrence.
$$
\mathrm{cost} = \sum_{j=1}^{m} C(x_j).
$$

2. **Update recency:** Move the inputs to the top of the stack sequentially in read order.

3. **Push outputs:** Allocate the outputs as fresh blocks and push them to the top of the stack at zero cost.

## Example walkthrough

### 1. Multi-byte values & execution

Consider 1-byte arguments `a` and `d`, and 2-byte arguments `b` and `c` executing `return b + c`. 

**Initial stack**

Arguments are pushed in call order `[a, b, c, d]`. The 6-byte initial stack depths are:
- `d`: `{1}`
- `c`: `{2, 3}`
- `b`: `{4, 5}`
- `a`: `{6}`

**Compute read cost**

$$
\mathrm{cost} = C(b) + C(c) = (\lceil\sqrt{4}\rceil + \lceil\sqrt{5}\rceil) + (\lceil\sqrt{2}\rceil + \lceil\sqrt{3}\rceil) = (2 + 3) + (2 + 2) = 9.
$$

**Update stack**

Inputs are moved to the top sequentially in read order (`[a, d, b, c]`), then the `result` is pushed:

```text
[a, d, b, c, result]
```

### 2. Read order matters

Because recency updates sequentially from the input list, input order changes the resulting stack state for future instructions:

- `return b + c` yields `[a, d, b, c, result]`
- `return c + b` yields `[a, d, c, b, result]`

*(Note: The read cost of the current instruction is unchanged since all inputs are priced against the initial stack state).*

### 3. Repeated inputs

Because of the rules above, `a+a` is charged twice for memory read of `a`.

## Future work

### Bits vs. bytes

ByteDMD works at byte granularity because byte-level objects are directly exposed in most high-level runtimes. A finer BitDMD variant could charge at the bit level.

### Parallel execution

The current model gives function-entry argument placement zero cost. In a multi-processor setting, that step could instead be charged according to the distance from the original data location to the processor executing the computation.

[Original Google Doc](https://docs.google.com/document/d/1sj5NqOg6Yqh10bXzGVEF5uIzSjFWAnqqTE75AMng2-s/edit?tab=t.0#heading=h.ujy6ygk7sjmb)
