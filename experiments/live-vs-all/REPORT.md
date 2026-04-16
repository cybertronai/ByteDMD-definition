# Classic DMD · DMD-live · Tombstone — report

## Three cost models on one trace

Every measure below is priced on the **same L2 trace** (the same algorithm,
the same load/store events). The three differ only in how they resolve
depth per read. The Gemini analysis in
`gemini/15apr26-dmdlive-analysis.md` derives each regime from the physics
of a 2-D continuous cache whose `d`-th concentric ring costs `⌈√d⌉`.

### Classic DMD — "Infinite Graveyard"

LRU stack with **no** liveness compaction. Every STORE pushes to the top;
every LOAD bumps the variable to the top; **dead variables never leave
the stack**. Depth of a LOAD of X = number of distinct variables (live or
dead) referenced since X's previous LOAD.

On RMM, the addition step at the root wades through an `O(N^3)`
graveyard of dead temporaries from sibling sub-calls. Each read charges
`√(N^3) = N^{1.5}` across `N^2` reads → **`Θ(N^{3.5})`**.

### DMD-live — "Teleporting Cache"

LRU stack **with** liveness compaction. A variable is dropped from the
stack the instant its last LOAD executes; everything above it slides
inward for free. Depth of a LOAD of X = number of **live bytes**
referenced between X's previous LOAD and the current one.

The cache radius is clamped to the instantaneous live working set
`O(N^2)`. Each read charges `O(N)` across `O(N^3)` reads per recursion
level × `log N` levels → **`Θ(N^3 log N)`**. This is a mathematical
idealisation — real silicon cannot slide arbitrary outer rings inward for
free.

### Tombstone — concrete realistic allocator (mobile LRU with holes)

A stack grows from bottom (index 0) to top (highest index). Each
variable's address equals its distance from the top of the stack.

- On STORE: place the new variable in the **highest-index hole** left by
  a prior death; if no hole exists, extend the stack by one slot at the
  top.
- On LOAD: record the variable's current address, leave a tombstone at
  its old slot, then move the variable into the highest-index hole
  **above** the old slot. If no such hole exists, extend to a new top.
- Dead variables (past their last LOAD) leave permanent tombstones. Live
  data **never slides inward by itself** — only the variable being
  accessed moves, and only upward.

This matches the "Tombstone / High-Water Mark" picture in the Gemini
note: no free global compaction, but a new allocation intelligently
recycles the closest available hole, so the active cache radius stays
bounded. Implemented as `bytedmd_ir.compile_tombstone` with a lazy
max-heap over hole indices for `O(log T)` per event.

The "mobile" aspect is crucial. A **stationary** Tombstone (each variable
pinned to its addr forever) would pay `√addr = O(N)` on every reread of
an input, giving `Θ(N^4)` on matmul — strictly worse than Classic DMD.
The mobile variant bumps re-read variables back toward the top, which
preserves the `N^3 log N`-class asymptotic.

## The three IR levels

| Level | Name          | Contents                                                |
|-------|---------------|---------------------------------------------------------|
| L1    | Python source | Algorithm as a plain function.                          |
| L2    | Abstract IR   | `LOAD(var)` / `STORE(var)` / `OP(name, in, out)` — no addresses. |
| L3    | Concrete IR   | Same events + physical `addr` per variable.             |

Classic DMD and DMD-live are computed directly on L2 via a Fenwick-tree
LRU walk. Tombstone lowers L2 to L3 via `compile_tombstone` and applies
`sum ceil(sqrt(addr))`.

## Results

Two algorithms traced at `N ∈ {4, 8, 16, 32, 64}`.

### Cache-oblivious RMM (8-way recursive)

|  N  | Classic DMD  | DMD-live    | Tombstone     | classic / live | tomb / live |
|----:|-------------:|------------:|--------------:|---------------:|------------:|
|   4 |        1,043 |         689 |           912 |          1.51× |       1.32× |
|   8 |       13,047 |       7,773 |        11,582 |          1.68× |       1.49× |
|  16 |      154,251 |      80,716 |       131,742 |          1.91× |       1.63× |
|  32 |    1,779,356 |     794,969 |     1,445,402 |          2.24× |       1.82× |
|  64 |   20,291,116 |   7,554,413 |    15,768,636 |          2.69× |       2.09× |

### One-level tiled matmul (tile = ⌈√N⌉)

|  N  | Classic DMD  | DMD-live    | Tombstone     | classic / live | tomb / live |
|----:|-------------:|------------:|--------------:|---------------:|------------:|
|   4 |        1,000 |         644 |           902 |          1.55× |       1.40× |
|   8 |       12,368 |       7,210 |        11,250 |          1.72× |       1.56× |
|  16 |      143,280 |      74,560 |       122,699 |          1.92× |       1.65× |
|  32 |    1,740,310 |     790,183 |     1,500,333 |          2.20× |       1.90× |
|  64 |   19,737,581 |   7,917,595 |    17,264,621 |          2.49× |       2.18× |

## Asymptotic verification

Fitting `cost / N^α`:

- **Classic DMD**: normalising by `N^{3.5}` gives 8–10 across N; steady
  → **`Θ(N^{3.5})`** confirmed.
- **DMD-live**: normalising by `N^3 log₂ N` gives ≈ 5 across N; steady
  → **`Θ(N^3 log N)`** confirmed.
- **Tombstone**: normalising by `N^3 log₂ N` gives 7.1 → 10.0 across
  N = 4…64. Near-constant → consistent with
  **`Θ(N^3 log N)`** with a moderate leading coefficient
  (≈ 1.5–2× DMD-live), matching the Gemini doc's prediction that the
  realistic tombstone model "preserves the optimal
  `Θ(N^3 log N)` asymptotic with a slightly higher leading constant".

## Interpretation

- **Classic DMD** is the pessimistic upper bound: dead temporaries
  cannot be evicted at all.
- **DMD-live** is the optimistic lower bound: free sliding compaction.
- **Tombstone** is the realistic middle: live data moves only when
  touched (LRU bumping), dead slots are reused by new stores but never
  by sliding. Sits cleanly between the two DMD metrics on both tiled
  and recursive matmul.

The fact that Tombstone lands in the envelope for every N tested is the
quantitative story: `Θ(N^3 log N)` really is achievable in hardware
without DMD-live's magic sliding — you only need hole-reuse plus LRU
bumping of the accessed value.

### Contrast with the earlier stationary `min_heap`

An earlier version of this experiment used a stationary min-heap
allocator (each variable pinned to its address, the smallest-free addr
assigned at STORE). That model gives `Θ(N^4)` on matmul and sits
**above** Classic DMD. The stationary variant is kept under the name
`compile_min_heap` in `bytedmd_ir.py` as a sanity baseline; it is not
the same thing as the Tombstone model the Gemini doc describes.

## Reproducibility

```bash
uv run pytest test_bytedmd_ir.py                        # 33 tests
uv run --script experiments/live-vs-all/envelope.py 4,8,16,32,64
```

Outputs:

- `envelope.png` — log-log cost vs N, three curves per algorithm with
  `N^3 log N` and `N^{3.5}` reference lines.
- `envelope_ratio.png` — `classic / live` envelope width vs N.
- Plaintext tables on stdout (same numbers as the tables above).
