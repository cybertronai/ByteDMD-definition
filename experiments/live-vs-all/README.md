# ByteDMD-classic vs ByteDMD-live

This experiment contrasts the two ByteDMD measures with several concrete
register-allocation strategies, and checks the predicted
`O(N^3.5)` vs `O(N^3 log N)` asymptotics on matmul.

## The two ByteDMD measures (no allocator)

Both are computed **directly on the L2 trace** by running an LRU stack and
charging `ceil(sqrt(depth))` per LOAD.

- **ByteDMD-classic** — LRU stack with **no** liveness compaction. Dead
  variables never leave the stack, polluting deeper rings. Cost of a LOAD
  of X = `ceil(sqrt(D))` where `D` is the number of distinct variables
  referenced since X's previous LOAD, dead or alive.
- **ByteDMD-live** — LRU stack **with** liveness compaction. A variable is
  dropped from the stack on its last LOAD. Cost of a LOAD of X =
  `ceil(sqrt(L))` where `L` is the number of **live bytes** referenced
  between X's previous LOAD and the current one.

## The three IR levels

The framework lives in **`bytedmd_ir.py`** — the module name reflects its
multi-level intermediate representation, which separates an algorithm into
three lowering levels.

### Level 1 — Python source

Plain Python. Example (`bytedmd_ir.matmul_rmm`):

```python
def matmul_rmm(A, B):
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    ...
```

### Level 2 — Abstract IR

`bytedmd_ir.trace(func, args)` returns a flat sequence of events:

```
L2Store(var=1)              # input a
L2Store(var=2)              # input b
L2Load(var=1)               # read a
L2Load(var=2)               # read b
L2Op(name='add', in_vars=(1, 2), out_var=3)
L2Store(var=3)              # store a+b
```

The two ByteDMD measures are priced directly on this L2 event stream.

### Level 3 — Concrete IR (optional, for allocators)

An allocator lowers L2 to L3 by assigning a physical addr to every
variable. Cost on L3 is `sum ceil(sqrt(addr))` per LOAD.

## Allocators used for reference

| Column              | Internal key  | Behaviour                                       |
|---------------------|---------------|-------------------------------------------------|
| No reuse            | `no_reuse`    | Every variable gets a fresh addr; no recycling  |
| LIFO slots          | `lru_static`  | Liveness-driven; freed addrs returned LIFO      |
| Belady (offline)    | `belady`      | Offline oracle; picks lowest free addr with future-load info |
| Min-heap reuse      | `min_heap`    | Liveness-driven; smallest-free-first            |

All four allocators are stationary (no LRU bumping), so they cost *more*
than `bytedmd_classic` on matmul.

## Results

### Cache-oblivious RMM (8-way)

| N  | ByteDMD-classic | ByteDMD-live | No reuse      | LIFO slots  | Belady (offline) | Min-heap reuse | classic / live |
|---:|----------------:|-------------:|--------------:|------------:|-----------------:|---------------:|---------------:|
|  4 |           1,043 |          689 |         1,469 |       1,010 |              985 |            985 |          1.51× |
|  8 |          13,047 |        7,773 |        29,964 |      16,415 |           16,315 |         16,315 |          1.68× |
| 16 |         154,251 |       80,716 |       623,025 |     263,663 |          266,593 |        266,593 |          1.91× |
| 32 |       1,779,356 |      794,969 |    13,222,807 |   4,219,845 |        4,320,478 |      4,320,478 |          2.24× |
| 64 |      20,291,116 |    7,554,413 |   285,417,081 |  67,561,749 |       69,716,078 |     69,716,078 |          2.69× |

### Tiled matmul (one level, tile = ⌈√N⌉)

| N  | ByteDMD-classic | ByteDMD-live | No reuse      | LIFO slots  | Belady (offline) | Min-heap reuse | classic / live |
|---:|----------------:|-------------:|--------------:|------------:|-----------------:|---------------:|---------------:|
|  4 |           1,000 |          644 |         1,472 |       1,005 |              961 |            961 |          1.55× |
|  8 |          12,368 |        7,210 |        30,052 |      15,460 |           15,128 |         15,128 |          1.72× |
| 16 |         143,280 |       74,560 |       624,600 |     238,438 |          233,811 |        233,811 |          1.92× |
| 32 |       1,740,310 |      790,183 |    13,243,856 |   3,669,464 |        3,683,154 |      3,683,154 |          2.20× |
| 64 |      19,737,581 |    7,917,595 |   285,686,728 |  57,841,639 |       57,162,017 |     57,162,017 |          2.49× |

**Empirical asymptotics**:

- `bytedmd_classic ≈ 9.6 · N^{3.5}` at N = 64 (constant stable across N).
- `bytedmd_live ≈ 4.8 · N^3 · log₂ N`.
- `classic / live` ratio grows with N at roughly the predicted
  `√N / log N` rate.

See **[REPORT.md](REPORT.md)** for the full writeup with allocator
descriptions, asymptotic derivation, and discussion.

## Reproducibility

```bash
uv run pytest test_bytedmd_ir.py
uv run --script envelope.py 4,8,16,32,64
```
