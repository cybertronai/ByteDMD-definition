# Final Report: Three Matmul Memory Models

This experiment makes the matrix-multiplication pipeline explicit at three levels:

1. **Python algorithm level**: regular Python implementations of one-level tiled matmul, 8-way recursive matmul, and Strassen live in [hierarchy.py](./hierarchy.py).
2. **Abstract trace level**: running those algorithms over traced scalar values emits a flat logical stream like `load A[0,0]`, `store t17_mul`, `load t17_mul`.
3. **Concrete address level**: the same logical stream is compiled to a concrete-address `tombstone` stack, producing accesses like `load A[0,0] addr=256`, `store t17_mul addr=498`.

The implementation is in [hierarchy.py](./hierarchy.py), the runner is [run_experiment.py](./run_experiment.py), the raw data is [results.json](./results.json), and the unit tests are [test_matmul_hierarchy.py](/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/ByteDMDcodex/test_matmul_hierarchy.py).

## The three metrics

The report in [15apr26-dmdlive-analysis.md](/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/ByteDMD/gemini/15apr26-dmdlive-analysis.md) argues that the useful comparison is not between many allocators, but between three memory-management regimes:

| Metric | Level | Mental model | What is charged |
|--------|-------|--------------|-----------------|
| `ByteDMD-classic` | abstract | infinite graveyard | logical reuse depth with dead values never reclaimed |
| `ByteDMD-live` | abstract | teleporting cache | logical reuse depth after dead values vanish and the stack closes up for free |
| `Tombstone` | concrete | high-water-mark physical cache | current concrete address in an LRU-with-holes stack; load cost is charged from that address |

This is the framing used throughout the experiment now.

## Cost model

The discrete ByteDMD price of a load at depth or address `d` is

```text
cost(d) = ceil(sqrt(d))
```

That exactly matches the continuous cache picture with:

- shell sizes `1, 3, 5, 7, ...`
- cumulative capacities `1, 4, 9, 16, ...`
- access costs `1, 2, 3, 4, ...`

The geometric interpretation is:

- a cache of radius `c` contains `c^2` scalar slots
- the `d`-th slot therefore costs the smallest `c` such that `d <= c^2`
- so the access cost is `ceil(sqrt(d))`

For `ByteDMD-classic` and `ByteDMD-live`, `d` is a logical reuse depth.
For `Tombstone`, `d` is the concrete address of the loaded value in the tombstone stack at that moment in the trace.

## The three methods in plain language

### `ByteDMD-classic`

This is the original total-bytes model. Every temporary that is ever created stays in the logical LRU history forever, even after it is dead. The effect is an **infinite graveyard**: dead temporaries keep pushing older useful data farther outward.

### `ByteDMD-live`

This is the live-bytes model. As soon as a value reaches its last use, it disappears from the logical stack. That makes the stack instantly collapse inward. It is a useful lower bound, but it implicitly gives the cache free compaction, which is not physically realistic.

### `Tombstone`

This is the concrete-address model used for the physical middle ground.

- When a temporary dies, it leaves a hole at its current address.
- Unrelated older values do **not** slide inward for free.
- A store or reload moves only the touched value toward the center by filling the closest available inner hole; if no such hole exists, the frontier grows.

In the current implementation, `tombstone` is an **LRU stack with holes**:

- dead values leave tombstones behind
- a new store fills the nearest inner tombstone or extends the frontier
- a load charges the value's current address, then refreshes only that value
- no global compaction occurs

That makes `Tombstone` a direct model of the high-water-mark picture from the Gemini note: fragmentation is real, but the active frontier stays bounded by reused holes instead of growing like a memory leak.

## Sample trace

For the tiled kernel, the first few logical events are:

```text
 load A[0,0]         role=input
 load B[0,0]         role=input
store t1_mul         role=temp
 load A[0,1]         role=input
 load B[1,0]         role=input
store t2_mul         role=temp
 load t1_mul         role=temp
 load t2_mul         role=temp
store t3_add         role=temp
```

Compiling the same prefix with `tombstone` produces:

```text
 load A[0,0]         addr= 256 role=input
 load B[0,0]         addr= 513 role=input
store t1_mul         addr= 513 role=temp
 load A[0,1]         addr= 256 role=input
 load B[1,0]         addr= 498 role=input
store t2_mul         addr= 498 role=temp
 load t1_mul         addr= 514 role=temp
 load t2_mul         addr= 498 role=temp
store t3_add         addr= 498 role=temp
```

The logical schedule is unchanged, but the concrete trace now exposes which values really stay near the center, which ones drift outward, and how holes get recycled without globally collapsing the stack.

## N=16 results

The runner fixes `N=16`, uses a one-level tile size of `4`, and computes:

- `ByteDMD-classic` from logical reuse depths
- `ByteDMD-live` from logical reuse depths with liveness compaction
- `Tombstone` from concrete load addresses under hole reuse

### ByteDMD totals

| Algorithm | ByteDMD-classic | ByteDMD-live | Tombstone |
|-----------|-----------------:|-------------:|----------:|
| Tiled | 143,796 | 75,596 | 123,225 |
| Recursive | 154,474 | 81,197 | 132,102 |
| Strassen | 352,655 | 172,892 | 255,867 |

These numbers have the intended interpretation:

- `ByteDMD-classic` is the pessimistic upper model.
- `ByteDMD-live` is the optimistic lower model.
- `Tombstone` sits between them as the concrete no-free-compaction cost.

That matches the physical story from the analysis note:

- leaking dead temporaries is too pessimistic
- free inward sliding is too optimistic
- hole reuse plus stationary outer data is the realistic middle ground

## Plot

![Reuse-distance envelope](./reuse_envelope_n16.png)

The curves show `loads above cache size` against cache capacity in scalar slots. For all three algorithms, the concrete `Tombstone` curve stays inside the abstract envelope formed by `ByteDMD-live` and `ByteDMD-classic`.

## Tests

The unit tests cover:

- correctness of all three matmul algorithms on small matrices
- emission of all three levels
- the expected ordering `ByteDMD-live <= Tombstone <= ByteDMD-classic` on a small recursive case
- monotonicity of the cache-size sweep

Run them with:

```bash
uv run pytest -q test_matmul_hierarchy.py
```

Regenerate the figure and data with:

```bash
uv run experiments/matmul_hierarchy/run_experiment.py
```
