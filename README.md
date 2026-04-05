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
assert bytedmd(dot, (a, b)) == 14
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


## ByteDMD Costs for higher-level ops

### Matrix-vector (4x4 matrix, 4-vector)

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| matvec (i-j) | y = A @ x | 194 |
| vecmat (j-i) | y = x^T @ A | 191 |

### Matrix multiply (4x4)

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| matmul (i-j-k) | C = A @ B | 948 |
| matmul (i-k-j) | C = A @ B | 1016 |
| matmul (snake-j) | C = A @ B | 906 |
| matmul (2x2 tiled) | C = A @ B | 947 |
| Strassen (leaf=1) | C = A @ B | 2435 |
| Winograd | C = A @ B | 2178 |

### microGPT single-token forward pass

Architecture: `vocab=4, embd=4, heads=2, head_dim=2, 1 layer, block_size=4`.
Based on [Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| microGPT (1 layer, embd=4) | single token forward | 7047 |

# Python Gotcha's
This version implements ByteDMD by wrapping Python objects. This means that "Instruction Set" of this metric corresponds to Python built-ins, documented under [docs/instruction_set.md](docs/instruction_set.md).

Python behavior means this implementation occasionally doesn't match README semantics and it's possible to escaping the wrapping mechanism occasionally, known failures cases are documented as test_gotchas.py .


[Original Google Doc](https://docs.google.com/document/d/1sj5NqOg6Yqh10bXzGVEF5uIzSjFWAnqqTE75AMng2-s/edit?tab=t.0#heading=h.ujy6ygk7sjmb)

