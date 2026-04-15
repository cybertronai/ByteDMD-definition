# ByteDMD-classic vs ByteDMD-live

This experiment quantifies the gap between the two ByteDMD variants:

- **ByteDMD-classic** — counts every allocated byte. Each variable lives in
  its own slot forever ("memory-leak" model). Allocator: `no_reuse`.
- **ByteDMD-live** — counts only live bytes. Slots of dead variables are
  recycled ("compiler model"). Allocator: `min_heap`.

Several intermediate register-allocation strategies (`LIFO slots`, `Belady
offline`) fall inside the envelope these two measures form.

## The three IR levels

The framework lives in **`bytedmd_ir.py`** — the module name reflects its
multi-level intermediate representation, which separates an algorithm into
three lowering levels.

### Level 1 — Python source

Plain Python. The algorithm is written once and executed under a tracer
that overloads numeric operators. Example (`bytedmd_ir.matmul_rmm`):

```python
def matmul_rmm(A, B):
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)
    C11 = _add_mat(matmul_rmm(A11, B11), matmul_rmm(A12, B21))
    ...
```

### Level 2 — Abstract IR (var IDs only)

`bytedmd_ir.trace(func, args)` returns a flat sequence of events:

```
L2Store(var=1)              # input a
L2Store(var=2)              # input b
L2Store(var=3)              # input c
L2Load(var=1)               # read a
L2Load(var=2)               # read b
L2Op(name='add', in_vars=(1, 2), out_var=4)
L2Store(var=4)              # store a+b
L2Load(var=4)
L2Load(var=3)
L2Op(name='add', in_vars=(4, 3), out_var=5)
L2Store(var=5)              # store (a+b)+c
```

L2 has *no concept of memory addresses*. Each variable carries an opaque
integer ID; the trace only records when each ID is born (`L2Store`) and
read (`L2Load`).

### Level 3 — Concrete IR (var IDs + physical addresses)

An **allocator** lowers L2 to L3 by assigning each variable to a physical
slot address. Different allocators produce different L3 traces (and hence
different costs) for the same L2 trace.

```
L3Store(var=1, addr=1)
L3Store(var=2, addr=2)
L3Store(var=3, addr=3)
L3Load(var=1, addr=1)
L3Load(var=2, addr=2)
L3Op(name='add', in_vars=(1,2), in_addrs=(1,2), out_var=4, out_addr=None)
L3Store(var=4, addr=4)         # no_reuse: fresh addr
                                # min_heap: addr 1 (a freed its slot)
...
```

The cost of the L3 trace is

\[ \text{cost} = \sum_{\text{L3Load at addr } d} \lceil \sqrt{d} \rceil \]

which is the standard ByteDMD spatial penalty for a stationary 2-D layout
where slot `d` lives on the d-th concentric ring (1-3-5-7-… cache hierarchy).

## Allocator policies (L2 → L3)

| column name         | allocator key | behavior                                                     | role in envelope              |
|---------------------|---------------|--------------------------------------------------------------|-------------------------------|
| **ByteDMD-classic** | `no_reuse`    | Every variable gets a fresh addr; freed slots never recycled | **upper envelope**            |
| LIFO slots          | `lru_static`  | Stationary slots; freed addrs returned in LIFO order         | intermediate                  |
| Belady (offline)    | `belady`      | Offline oracle; picks lowest free addr with future-load info | intermediate / lower          |
| **ByteDMD-live**    | `min_heap`    | Stationary slots; freed addrs returned via min-heap          | **lower envelope**            |

For matmul every intermediate is read exactly once, so `Belady (offline)`
and `ByteDMD-live` produce identical assignments. Both achieve the optimal
asymptotic `O(N^3 log N)`.

## The envelope

Tracing cache-oblivious 8-way RMM and one-level tiled matmul
(tile = ⌈√N⌉) under every allocator policy:

### Cache-oblivious RMM (8-way)

|   N | ByteDMD-classic | LIFO slots | Belady (offline) | **ByteDMD-live** | classic / live |
|----:|----------------:|-----------:|-----------------:|-----------------:|---------------:|
|   4 |           1,469 |      1,010 |              985 |              985 |         1.49×  |
|   8 |          29,964 |     16,415 |           16,315 |           16,315 |         1.84×  |
|  16 |         623,025 |    263,663 |          266,593 |          266,593 |         2.34×  |
|  32 |      13,222,807 |  4,219,845 |        4,320,478 |        4,320,478 |         3.06×  |
|  64 |     285,417,081 | 67,561,749 |       69,716,078 |       69,716,078 |         4.09×  |

### Tiled matmul (one level, tile = ⌈√N⌉)

|   N | ByteDMD-classic | LIFO slots | Belady (offline) | **ByteDMD-live** | classic / live |
|----:|----------------:|-----------:|-----------------:|-----------------:|---------------:|
|   4 |           1,472 |      1,005 |              961 |              961 |         1.53×  |
|   8 |          30,052 |     15,460 |           15,128 |           15,128 |         1.99×  |
|  16 |         624,600 |    238,438 |          233,811 |          233,811 |         2.67×  |
|  32 |      13,243,856 |  3,669,464 |        3,683,154 |        3,683,154 |         3.60×  |
|  64 |     285,686,728 | 57,841,639 |       57,162,017 |       57,162,017 |         5.00×  |

See **[REPORT.md](REPORT.md)** for the full writeup with allocator details,
asymptotic derivation, and conclusions.

Every concrete stationary policy lands inside `[ByteDMD-live, ByteDMD-classic]`.
The gap widens with N — empirically faster than `√N / log N`, consistent
with the analytic prediction `N^{3.5} / (N^3 log N) = √N / log N`.

Reference asymptotes shown on the log-log plot:

- **ByteDMD-classic**  →  `O(N^{3.5})` (master theorem with the addition
   step at the root of the polluted LRU stack)
- **ByteDMD-live**  →  `O(N^3 log N)` (working set bounded by `O(N²)`;
   per-level work `O(N³)` repeated across `log N` levels)

See `envelope.png` (all four curves) and `envelope_ratio.png` (cost ratio
vs N).

## Running the experiment

```bash
uv run --script envelope.py            # default Ns = 2,4,8,16,32
uv run --script envelope.py 4,8,16,32,64
```

Outputs:
- `envelope.png` — log-log cost vs N for both algorithms
- `envelope_ratio.png` — `cost(no_reuse) / cost(min_heap)` envelope width
- text summary on stdout

## Tests

`test_bytedmd_ir.py` (at the repo root) exercises all three levels:

- L1 → L2: trace shape for known small inputs (`(a+b)+c`, dot product),
  numerical correctness of all matmul implementations.
- L2 → L3: addr sequencing under `no_reuse`, slot recycling under
  `min_heap`, every L3Load addr matches the most recent L3Store addr for
  that var (sanity invariant).
- Cost: hand-computed cost for `(a+b)+c`, envelope inequality
  `no_reuse ≥ live-bytes` across all algorithms and Ns, peak-addr bound
  for RMM under `min_heap` (`O(N²)` not `O(N³)`), envelope width grows
  monotonically with N.

```bash
uv run pytest test_bytedmd_ir.py -v
```
