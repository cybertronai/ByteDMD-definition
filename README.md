# A cost model of complexity for the 21st century: ByteDMD

**TL;DR**
- The memory wall makes reducing data movement more important than reducing arithmetic.
- Instead of FLOP count, focus on the cost of data movement.
- Incorporate 2D wire-length heuristic to model the cost of accessing faraway bytes.

![ByteDMD](docs/dmd_animated.gif)

## Motivation

Modern architectures spend more energy moving data than doing arithmetic, making FLOP counts an outdated cost metric. Bill Dally ([ACM Opinion](https://cacm.acm.org/opinion/on-the-model-of-computation-point/)) proposed penalizing data movement based on 2D spatial distance to the processor. To avoid manual spatial mapping, Ding and Smith ([Beyond Time Complexity, 2022](https://arxiv.org/abs/2203.02536)) automated this via Data Movement Distance (DMD): a rule treating memory as an LRU stack where reading depth $d$ costs $\sqrt{d}$, modeling a cache laid out in 2D.

The original DMD treats values abstractly. ByteDMD refines this to the byte level: a $k$-byte value occupies $k$ consecutive stack positions, intrinsically favoring smaller data types.

## Computation Model

An idealized processor operates directly on a byte-level LRU stack. **Computations and writes are free; only memory reads incur a cost.**

- **Stack State:** Ordered from least recently used (bottom) to most recently used (top). Depth is measured in bytes from the top (topmost byte = depth 1). A $k$-byte value occupies $k$ contiguous depths.
- **Initialization:** On function entry, arguments are pushed to the top in call order.
- **Read Cost:** Reading a value $x$ spanning byte depths $D(x)$ costs:
  $$ C(x) = \sum_{d \in D(x)} \lceil\sqrt{d}\rceil $$
  *(e.g., a 2-byte value spanning depths 4 and 5 costs $\lceil\sqrt{4}\rceil + \lceil\sqrt{5}\rceil = 2 + 3 = 5$)*

### Instruction Semantics

For an instruction with inputs $x_1, \dots, x_m$ and outputs $y_1, \dots, y_n$:

1. **Price reads:** Evaluate $\sum C(x_j)$ against the stack state *before* the instruction begins. Repeated inputs are charged per occurrence (e.g., `a + a` charges for reading `a` twice).
2. **Update LRU:** Move inputs to the top of the stack sequentially in read order. *(Note: Because of this sequential update, `b + c` vs. `c + b` yields the same cost but different final stack states).*
3. **Push outputs:** Allocate new output blocks and push them to the top at zero cost.

## Example Walkthrough

Consider 1-byte arguments `a`, `d`, and 2-byte arguments `b`, `c`. Execute `return b + c`.

**1. Initial Stack** 
Arguments are pushed in call order `[a, b, c, d]`, yielding these stack depths from the top:
- `d`: `{1}`
- `c`: `{2, 3}`
- `b`: `{4, 5}`
- `a`: `{6}`

**2. Read Cost**  
Inputs are priced simultaneously against the initial stack state:
$$ \mathrm{cost} = C(b) + C(c) = (\lceil\sqrt{4}\rceil + \lceil\sqrt{5}\rceil) + (\lceil\sqrt{2}\rceil + \lceil\sqrt{3}\rceil) = (2 + 3) + (2 + 2) = 9 $$

**3. Update Stack**  
Inputs move to the top sequentially in read order (`b`, then `c`), followed by the new `result` block being pushed:
```text
[a, d, b, c, result]
```


## Future Work
### Bits vs. Bytes

ByteDMD operates at byte granularity because bytes are exposed directly in high-level runtimes. A finer BitDMD variant could charge at the bit level.

### Parallel Execution

Currently, initializing a function's arguments on the stack is considered a free operation. To model parallelism we could introduce multiple processors with their own LRU stacks. Initializing arguments onto the stack could incur a cost proportional to the distance between the data's original location and the location of the LRU stack.

[Original Google Doc](https://docs.google.com/document/d/1sj5NqOg6Yqh10bXzGVEF5uIzSjFWAnqqTE75AMng2-s/edit?tab=t.0#heading=h.ujy6ygk7sjmb)

