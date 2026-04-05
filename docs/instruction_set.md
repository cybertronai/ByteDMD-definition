# ByteDMD Instruction Set

This documents the complete set of instructions recognized by the ByteDMD cost model. Each instruction reads its inputs from the LRU stack (incurring cost), then pushes outputs onto the stack (free).

## Binary Arithmetic

Each has a reverse form (e.g., `radd`) with the same arity for when the tracked value is the right-hand operand.

| Instruction | Inputs | Outputs | Python syntax |
|-------------|--------|---------|---------------|
| add | 2 | 1 | `a + b` |
| sub | 2 | 1 | `a - b` |
| mul | 2 | 1 | `a * b` |
| truediv | 2 | 1 | `a / b` |
| floordiv | 2 | 1 | `a // b` |
| mod | 2 | 1 | `a % b` |
| divmod | 2 | 2 | `divmod(a, b)` |
| pow | 2 or 3 | 1 | `a ** b`, `pow(a, b[, m])` |
| matmul | 2 | 1 | `a @ b` |

## Bitwise & Shift

Each has a reverse form. 

| Instruction | Inputs | Outputs | Python syntax |
|-------------|--------|---------|---------------|
| lshift | 2 | 1 | `a << b` |
| rshift | 2 | 1 | `a >> b` |
| and | 2 | 1 | `a & b` |
| xor | 2 | 1 | `a ^ b` |
| or | 2 | 1 | `a \| b` |

## Unary

| Instruction | Inputs | Outputs | Python syntax |
|-------------|--------|---------|---------------|
| neg | 1 | 1 | `-a` |
| pos | 1 | 1 | `+a` |
| abs | 1 | 1 | `abs(a)` |
| invert | 1 | 1 | `~a` |

## Rounding

| Instruction | Inputs | Outputs | Python syntax |
|-------------|--------|---------|---------------|
| trunc | 1 | 1 | `math.trunc(a)` |
| floor | 1 | 1 | `math.floor(a)` |
| ceil | 1 | 1 | `math.ceil(a)` |
| round | 1 or 2 | 1 | `round(a[, ndigits])` |

## Comparisons

Comparisons read both operands and push a tracked result onto the stack.

| Instruction | Inputs | Outputs | Python syntax |
|-------------|--------|---------|---------------|
| eq | 2 | 1 | `a == b` |
| ne | 2 | 1 | `a != b` |
| lt | 2 | 1 | `a < b` |
| le | 2 | 1 | `a <= b` |
| gt | 2 | 1 | `a > b` |
| ge | 2 | 1 | `a >= b` |

## Type Conversions (escape tracking)

These read the operand (incurring cost) but return a native Python value that leaves the LRU stack. They are used for control flow and indexing.

| Instruction | Inputs | Outputs | Python trigger |
|-------------|--------|---------|----------------|
| bool | 1 | 0 (escapes) | `if a:`, `not a`, `a and b` |
| int | 1 | 0 (escapes) | `int(a)` |
| float | 1 | 0 (escapes) | `float(a)`, `math.sqrt(a)` |
| complex | 1 | 0 (escapes) | `complex(a)` |
| index | 1 | 0 (escapes) | `range(a)`, `xs[a]` |
| hash | 1 | 0 (escapes) | `hash(a)`, dict keys |

## Container Operations

`_TrackedList` and `_TrackedTuple` intercept element access, recording reads when elements are retrieved.

| Operation | Reads | Python trigger |
|-----------|-------|----------------|
| getitem | 1 per element accessed | `a[i]`, `a[i:j]` |
| iter | 1 per element yielded | `for x in a:`, `zip(a, b)` |
| setitem | 0 (writes are free) | `a[i] = v` |

## Built-in Reductions (composite)

These are not single instructions but are composed of the primitives above. Their cost depends on input size N.

| Function | Composed of | Notes |
|----------|-------------|-------|
| `sum(iterable)` | N-1 `add`/`radd` ops | initial `0` is a constant |
| `min(iterable)` | N-1 comparisons | |
| `max(iterable)` | N-1 comparisons | |
| `any(iterable)` | up to N `bool` conversions | short-circuits |
| `all(iterable)` | up to N `bool` conversions | short-circuits |

## Constants

Literal constants (e.g., `10`, `0`, `2.5`) are allocated on the LRU stack on first use and cached for subsequent accesses. This means expressions like `a + 10` read both `a` and the constant `10`.
