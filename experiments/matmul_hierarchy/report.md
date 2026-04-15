# Final Report: Three-Level Matmul Trace Hierarchy

This experiment adds an explicit three-level pipeline for matrix multiplication traces:

1. **Python algorithm level**: regular Python implementations of one-level tiled matmul, 8-way recursive matmul, and Strassen live in [hierarchy.py](./hierarchy.py).
2. **Abstract load/store level**: running those algorithms over traced scalar values emits a flat logical access stream like `load A[0,0]`, `store t17_mul`, `load t17_mul`, without concrete addresses.
3. **Compiled concrete-address level**: the same logical stream is compiled to slot traces under concrete reuse policies such as `belady`, `lifo`, and offline `edf`, producing accesses like `load A[0,0] addr=1`, `store t17_mul addr=769`.

The implementation is in [hierarchy.py](./hierarchy.py), the runner is [run_experiment.py](./run_experiment.py), the raw data is [results.json](./results.json), and the unit tests are [test_matmul_hierarchy.py](/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/ByteDMDcodex/test_matmul_hierarchy.py).

## Why this exists

The repo already had abstract ByteDMD tracers, but not a single experiment that makes the hierarchy explicit end to end:

- source algorithm
- logical loads/stores
- compiled address trace

That missing layer matters if we want to compare:

- **ByteDMD-classic**, the original total-bytes logical stack model
- **ByteDMD-live**, the live-bytes logical stack model
- concrete address reuse strategies that a compiler could plausibly generate

on the **same** execution.

## Cost model

The discrete ByteDMD cost of a load at reuse depth `d` is:

```text
cost(d) = ceil(sqrt(d))
```

That means the cost tiers are:

| Cost | Depth range | Cumulative capacity |
|------|-------------|--------------------:|
| 1 | `1` | 1 |
| 2 | `2..4` | 4 |
| 3 | `5..9` | 9 |
| 4 | `10..16` | 16 |

So the natural interpretation is **square cumulative capacities** `1, 4, 9, 16, ...`.
If you look at the *incremental shell sizes* between tiers, those are the odd numbers
`1, 3, 5, 7, ...`.

## Method glossary

| Method | Level | Future knowledge | Stable addresses | Meaning |
|--------|-------|------------------|------------------|---------|
| `ByteDMD-classic` | abstract | no | no | Logical LRU over all values ever created; dead temps still contribute to depth |
| `ByteDMD-live` | abstract | no | no | Logical LRU after removing dead values immediately |
| `never-reuse` | concrete | no | yes | Compiled addresses never recycled; concrete analog of `ByteDMD-classic` |
| `lifo` | concrete | no | yes | Reuse the most recently freed concrete slot |
| `edf` | concrete | yes, at interval level | yes | Offline interval packing of temp lifetimes into fixed slots |
| `belady` | oracle | yes, exact next use | no | Two-pass oracle ranking of the live set by next use at every step |

The crucial distinction is that `belady` is **not** a stable-address compiler. It is an
oracle live-set ranking. The emitted `addr` is really a slot rank in the current optimal
ordering, not a persistent physical location.

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

Compiling the same prefix with `lifo` produces:

```text
 load A[0,0]         addr=   1 role=input
 load B[0,0]         addr= 257 role=input
store t1_mul         addr= 769 role=temp
 load A[0,1]         addr=   2 role=input
 load B[1,0]         addr= 273 role=input
store t2_mul         addr= 770 role=temp
 load t1_mul         addr= 769 role=temp
 load t2_mul         addr= 770 role=temp
store t3_add         addr= 770 role=temp
```

The logical and concrete traces have the same control/dataflow, but concrete slot reuse changes the reuse-distance curve.

## Why Belady can be lower than ByteDMD-live

Yes, it is normal **for the current oracle Belady column** to be lower than `ByteDMD-live`.

That does **not** mean a realistic compiled program with fixed addresses is doing less data
movement than the live-byte logical model. It means the current `belady` method is solving
a different, more optimistic problem:

1. `ByteDMD-live` keeps the access order fixed and uses the induced logical LRU order on the live set.
2. `belady` gets a full second pass and knows the exact next use of every live byte.
3. At each step it is allowed to re-rank the entire live set by next use and then charge the load against that oracle rank.
4. No cost is charged for changing that ranking.

So `belady` is acting like an **offline lower bound on the live set**, not like a compiled
stable-address trace. That is why it can drop below `ByteDMD-live`, and why the `belady`
curve can become almost all depth-1 loads for tiled and recursive matmul.

If the goal is a realistic compiler model, then the directly comparable columns are:

- `ByteDMD-classic`
- `ByteDMD-live`
- `never-reuse`
- `lifo`
- `edf`

If the goal is an **oracle lower bound with full future knowledge**, then `belady` is the
right column to keep, but it should be interpreted as an optimistic bound rather than as a
physical allocator.

## N=16 results

The runner fixes `N=16`, uses a one-level tile size of `4`, computes logical reuse depths under:

- `ByteDMD-classic`
- `ByteDMD-live`

and concrete reuse depths under:

- `never-reuse`
- `belady`
- `lifo`
- `edf`

then converts each depth trace into:

- a memory-size sweep `misses(M) = #{depth > M}`
- a discrete ByteDMD total `sum ceil(sqrt(depth))`

### ByteDMD totals

| Algorithm | ByteDMD-classic | ByteDMD-live | never-reuse | belady | lifo | edf |
|-----------|-----------------:|-------------:|------------:|-------:|-----:|----:|
| Tiled | 143,796 | 79,342 | 143,796 | 15,872 | 89,635 | 87,979 |
| Recursive | 154,474 | 84,132 | 154,474 | 15,872 | 98,312 | 99,446 |
| Strassen | 352,655 | 175,672 | 352,655 | 30,542 | 203,976 | 220,394 |

`never-reuse` is included explicitly because it is the compiled concrete policy that corresponds to `ByteDMD-classic` in this pipeline.

`belady` is now a true **offline two-pass oracle** over the logical trace: in pass 1 it records every future load, and in pass 2 it keeps the current live set ranked by the next time each live byte will be used. The emitted `addr` for a Belady access is that byte's current oracle slot rank. This makes `belady` an optimistic lower-bound allocator, not a stable-address compiler like `never-reuse` or `lifo`.

The qualitative interpretation stays the same:

- **ByteDMD-live** is a clear lower envelope and **ByteDMD-classic** is a clear upper envelope.
- `belady` is much lower than the stable-address policies here because it is an oracle next-use ranking over the live set, not a persistent-address allocator.
- `lifo` and `edf` remain the concrete stable-address policies in this table, so their ordering is empirical rather than guaranteed by one simple dominance relation.

For the current implementation, the clean interpretation is:

- `ByteDMD-live` = best online logical model without oracle reordering
- `belady` = offline oracle lower bound on that live set
- `lifo` / `edf` = realistic compiled-slot heuristics

## Plot

![Reuse-distance envelope](./reuse_envelope_n16.png)

The curves show `loads above cache size` against cache capacity in scalar slots. For every algorithm, the concrete curves stay inside the abstract envelope:

- `ByteDMD-live` is the optimistic lower bound
- `ByteDMD-classic` is the pessimistic upper bound
- `lifo` and `edf` stay between them, while `belady` is an oracle live-set ranking that can dip below `ByteDMD-live`

## Tests

The unit tests cover:

- correctness of all three algorithms on small matrices
- emission of all three levels
- equality between `ByteDMD-classic` and compiled `never-reuse`
- envelope sanity for small recursive examples

Run them with:

```bash
uv run pytest -q test_matmul_hierarchy.py
```

Regenerate the figure and data with:

```bash
uv run experiments/matmul_hierarchy/run_experiment.py
```
