# A cost model of complexity for the 21st century: ByteDMD

TLDR;
- Memory wall means that reducing data movement is more important than reducing arithmetic
- Implement a cost function which captures the cost of data movement 
- Use 2D VLSI heuristic where cost of accessing a cache of size $L$ scales in proportion to $\sqrt{L}$.

## Motivation

Modern architectures spend more energy moving data than doing arithmetic, hence FLOP count is no longer representative of real-life cost. ByteDMD is intended as a more representative scalar replacement of FLOP count. 

Bill Dally ([ACM Opinion](https://cacm.acm.org/opinion/on-the-model-of-computation-point/)) proposed a spatial model in which bytes live on a 2D grid and movement is penalized by Manhattan distance to the processor. Manually assigning spatial coordinates makes that model awkward for classical algorithms, so we replace manual placement with an automatic rule.

A natural starting point is the "geometric stack" introduced by Ding and Smith ([Beyond Time Complexity, 2022](https://arxiv.org/abs/2203.02536)). In that model, memory is represented as an infinitely layered LRU stack, and accessing depth $d$ costs $\sqrt{d}$. This can be viewed as Dally's Manhattan distance approach applied to cache arranged in concentric circles in 2D. We refer to this cost as the Data Movement Distance (DMD).

The original DMD treats values abstractly. ByteDMD refines the model to the byte level: a $k$-byte value occupies $k$ consecutive positions in the stack. This rewards algorithms that prefer smaller data-types like `int8` instead of `float32`.

![ByteDMD](dmd.png)

## Computation model

We model an idealized single processor with a byte-level LRU stack and a fixed set of instructions. Instructions and writes are free. The only charged operation is reading existing bytes from the stack.

### Stack state

The stack is ordered from least recently used at the bottom to most recently used at the top. Depths are counted in bytes from the top, so the topmost byte has depth 1.

Each value $x$ of size $|x|$ bytes occupies a contiguous block of $|x|$ byte positions.

### Initialization

On function entry, the arguments are placed on the stack in call order, so the last argument is closest to the top. For example,

```text
[a, b, c, d]
```

The top is the most recently used end, giving depths `d=1`, `c=2`, `b=3`, `a=4` when each argument is 1 byte.

### Cost of reading a value

To avoid floating-point arithmetic, we compute the cost using the ceiling of the square root, denoted $\lceil\sqrt{d}\rceil$. This is implemented as `usqrt(x) = isqrt(x - 1) + 1` where `isqrt` is the standard integer (floor) square root.

If a value $x$ occupies byte depths $D(x)$, then reading $x$ costs

$$
C(x) = \sum_{d \in D(x)} \lceil\sqrt{d}\rceil.
$$

A 1-byte value at depth 3 costs $\lceil\sqrt{3}\rceil = 2$. A 2-byte value spanning depths 4 and 5 costs $\lceil\sqrt{4}\rceil + \lceil\sqrt{5}\rceil = 2 + 3 = 5$.

### Instruction semantics

Consider an instruction with input list $x_1, \ldots, x_m$ and outputs $y_1, \ldots, y_n$.

1. **Price the reads.** Each input occurrence $x_j$ is charged against the stack state at the **start** of the instruction:

$$
\mathrm{cost} = \sum_{j=1}^{m} C(x_j).
$$

   If the same input appears twice, it is charged twice.

2. **Update recency.** After all read costs are computed, move each input to the top of the stack sequentially in read order. If an input appears twice, it is simply moved to the top twice, leaving it at the top.

3. **Push outputs.** Allocate the outputs as fresh blocks and push them onto the top of the stack at zero cost.

## Example walkthrough

### 1. Basic execution (`int8` variables)

Consider

```python
def my_add(a, b, c, d):
    return b + c
```

with 1-byte arguments.

**Initial stack**

```text
[a, b, c, d]
```

Top is MRU, so `d=1`, `c=2`, `b=3`, `a=4`.

**Read cost of `b + c`**

- read `b`: cost $\lceil\sqrt{3}\rceil = 2$
- read `c`: cost $\lceil\sqrt{2}\rceil = 2$

Total cost: $2 + 2 = 4$.

**Update stack**

Remove the referenced inputs and append them in read order:

```text
[a, d, b, c]
```

Then push the result `_r0`:

```text
[a, d, b, c, _r0]
```

### 2. Multi-byte values

Now let `a` and `d` be 1-byte values, and let `b` and `c` be 2-byte values.

The initial stack contains 6 bytes total:

- `d` occupies depth `{1}`
- `c` occupies depths `{2, 3}`
- `b` occupies depths `{4, 5}`
- `a` occupies depth `{6}`

Executing `b + c` charges

$$
C(b) + C(c) = (\lceil\sqrt{4}\rceil + \lceil\sqrt{5}\rceil) + (\lceil\sqrt{2}\rceil + \lceil\sqrt{3}\rceil) = (2 + 3) + (2 + 2) = 9.
$$

The post-instruction block order is still

```text
[a, d, b, c, _r0]
```

where `b`, `c`, and `_r0` are now multi-byte blocks.

### 3. Read order matters

Because recency is updated from the operand list, operand order changes the final stack state:

- `return b + c` gives

  ```text
  [a, d, b, c, _r0]
  ```

- `return c + b` gives

  ```text
  [a, d, c, b, _r0]
  ```

For a fixed set of distinct operands, the read cost of the current instruction is unchanged; what changes is the stack state seen by later instructions.

### 4. Repeated operands

If an instruction reads the same value twice, it is charged twice because the input list contains two read events. For example, in

```python
A[i][j] * A[i][j]
```

the value `A[i][j]` contributes its read cost twice. Both charges are computed from the stack state at the start of the instruction.

## Future work

### Bits vs. bytes

ByteDMD works at byte granularity because byte-level objects are directly exposed in most high-level runtimes. A finer BitDMD variant could charge at the bit level.

### Parallel execution

The current model gives function-entry argument placement zero cost. In a multi-processor setting, that step could instead be charged according to the distance from the original data location to the processor executing the computation.


[Original Google Doc](https://docs.google.com/document/d/1sj5NqOg6Yqh10bXzGVEF5uIzSjFWAnqqTE75AMng2-s/edit?tab=t.0#heading=h.ujy6ygk7sjmb)