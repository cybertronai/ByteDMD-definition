# ByteDMD Instruction List

**Source:** https://chatgpt.com/share/69d1b354-6454-83e8-90f3-cbb5e8cf5a5a
**Created with:** ChatGPT (Sutro group)
**Topic:** Complete list of "instructions" implemented by the ByteDMD model

---

## Prompt
For the ByteDMD model, give the list of "instructions" implemented by this model. For each, give the number of inputs and outputs.

## 24 Canonical Instruction Families (38 methods total with reverse forms)

### Binary Operations (2 inputs, 1 output)
| Instruction | Inputs | Outputs |
|-------------|--------|---------|
| add         | 2      | 1       |
| sub         | 2      | 1       |
| mul         | 2      | 1       |
| truediv     | 2      | 1       |
| floordiv    | 2      | 1       |
| mod         | 2      | 1       |
| pow         | 2      | 1       |
| lshift      | 2      | 1       |
| rshift      | 2      | 1       |
| and         | 2      | 1       |
| xor         | 2      | 1       |
| or          | 2      | 1       |
| matmul      | 2      | 1       |

### Special Binary (2 inputs, 2 outputs)
| Instruction | Inputs | Outputs |
|-------------|--------|---------|
| divmod      | 2      | 2       |

### Unary Operations (1 input, 1 output)
| Instruction | Inputs | Outputs |
|-------------|--------|---------|
| neg         | 1      | 1       |
| pos         | 1      | 1       |
| abs         | 1      | 1       |
| invert      | 1      | 1       |

### Comparisons (2 inputs, 0 tracked outputs — returns plain Python bool)
| Instruction | Inputs | Outputs |
|-------------|--------|---------|
| eq          | 2      | 0*      |
| ne          | 2      | 0*      |
| lt          | 2      | 0*      |
| le          | 2      | 0*      |
| gt          | 2      | 0*      |
| ge          | 2      | 0*      |

Reverse binary forms also exist with the same arity: radd, rsub, rmul, rtruediv, rfloordiv, rmod, rdivmod, rpow, rlshift, rrshift, rand, rxor, ror, rmatmul.

---

## Follow-up: Built-in Iterable Reductions and Math Functions

### Built-in Iterable Reductions (N = iterable length)
| Instruction | Inputs     | Outputs | Current Status                    |
|-------------|------------|---------|-----------------------------------|
| sum         | N or N+1   | 1       | composite over add/radd           |
| any         | up to N    | 1 bool  | needs __bool__                    |
| all         | up to N    | 1 bool  | needs __bool__                    |
| min         | N          | 1       | composite over comparisons        |
| max         | N          | 1       | composite over comparisons        |

### Built-in Math Functions
| Instruction | Inputs        | Outputs | Current Status     |
|-------------|---------------|---------|-------------------|
| abs         | 1             | 1       | already native     |
| round       | 1 or 2        | 1       | needs __round__    |

Note: Python builtin pow is really 2 or 3 -> 1 (pow(x, y[, mod])), although the current wrapper only implements the 2-input __pow__ form.
