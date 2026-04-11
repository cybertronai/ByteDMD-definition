# ByteDMD Instruction List

**Source:** https://chatgpt.com/share/69d1b354-6454-83e8-90f3-cbb5e8cf5a5a
**Created with:** ChatGPT (Sutro group)

---

## Prompt 1

> For the ByteDMD model here, give me the list of "instructions" implemented by this model. For instance, "add" is an instruction as well as divmod. Give me a list of instructions, and for each give the number of inputs and the number of outputs.

I'm taking the instruction list from `bytedmd.py` itself rather than the README, and I'll treat reverse dunder forms like `radd` as the same instruction with reversed operand order.

I've pinned down the actual op families now: binary ops, unary ops, and comparisons. Comparisons produce 0 tracked outputs here, while `divmod` is the one instruction with 2 outputs.

From `bytedmd.py`, there are **24 canonical instruction families**. If you count reverse binary dispatch separately (`radd`, `rsub`, …), there are **38 implemented methods total**. Comparisons return a plain Python `bool`, so they produce 0 tracked outputs in this model.

### Binary Operations (2 inputs → 1 output)

| instruction | inputs | outputs |
|---|---|---|
| add | 2 | 1 |
| sub | 2 | 1 |
| mul | 2 | 1 |
| truediv | 2 | 1 |
| floordiv | 2 | 1 |
| mod | 2 | 1 |
| pow | 2 | 1 |
| lshift | 2 | 1 |
| rshift | 2 | 1 |
| and | 2 | 1 |
| xor | 2 | 1 |
| or | 2 | 1 |
| matmul | 2 | 1 |

### Special Binary (2 inputs → 2 outputs)

| instruction | inputs | outputs |
|---|---|---|
| divmod | 2 | 2 |

### Unary Operations (1 input → 1 output)

| instruction | inputs | outputs |
|---|---|---|
| neg | 1 | 1 |
| pos | 1 | 1 |
| abs | 1 | 1 |
| invert | 1 | 1 |

### Comparisons (2 inputs → 0 tracked outputs*)

| instruction | inputs | outputs |
|---|---|---|
| eq | 2 | 0* |
| ne | 2 | 0* |
| lt | 2 | 0* |
| le | 2 | 0* |
| gt | 2 | 0* |
| ge | 2 | 0* |

\* Logical result is 1 bool, but it is not wrapped/pushed as a tracked output.

Reverse binary forms also exist with the same arity/output counts: `radd`, `rsub`, `rmul`, `rtruediv`, `rfloordiv`, `rmod`, `rdivmod`, `rpow`, `rlshift`, `rrshift`, `rand`, `rxor`, `ror`, `rmatmul`.

---

## Prompt 2

> Augment this table with 1. Built-in Iterable Reductions — These functions consume an iterable and collapse it down to a single value. When tracking data movement, these require reading up to N elements from memory. IE, sum, any. And built-in mathematical functions like abs, round, for a more complete instruction set.

Appending two sections: $N$ = iterable length. I'm counting logical outputs here; bools still escape the current Python wrapper. `bytedmd.py` only wires binary ops, reverse binary ops, unary neg/pos/abs/invert, and comparisons; the README already uses `sum(...)` in the dot-product example; and the tests note that implicit truthiness bypasses tracing.

### 1. Built-in Iterable Reductions

| instruction | inputs | outputs | current status |
|---|---|---|---|
| sum | $N$ or $N+1$ with `start` | 1 | composite over `add` / `radd` |
| any | up to $N$ | 1 bool | needs `__bool__` |
| all | up to $N$ | 1 bool | needs `__bool__` |
| min | $N$ | 1 | composite over comparisons |
| max | $N$ | 1 | composite over comparisons |

### 2. Built-in Math Functions

| instruction | inputs | outputs | current status |
|---|---|---|---|
| abs | 1 | 1 | already native |
| round | 1 or 2 (`x[, ndigits]`) | 1 | needs `__round__` |

One small correction to the old table: Python builtin `pow` is really 2 or 3 → 1 (`pow(x, y[, mod])`), although the current wrapper only implements the 2-input `__pow__` form.
