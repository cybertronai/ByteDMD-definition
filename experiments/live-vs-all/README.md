# Classic DMD · DMD-live · Tombstone

Three-column experiment: two ByteDMD measures priced directly on the L2
trace (Classic DMD, DMD-live) plus one concrete LRU-with-holes allocator
(Tombstone). Reference: `gemini/15apr26-dmdlive-analysis.md`.

## The three measures

| Column       | What it is                                                                                                         | RMM asymptotic  |
|--------------|--------------------------------------------------------------------------------------------------------------------|-----------------|
| Classic DMD  | LRU stack, **no** liveness — dead vars pollute deeper rings ("graveyard")                                          | `Θ(N^{3.5})`    |
| DMD-live     | LRU stack **with** liveness — dead vars vaporize and everything above slides inward for free ("teleporting cache") | `Θ(N^3 log N)`  |
| Tombstone    | Mobile LRU stack with holes — dead vars leave permanent tombstones, LOADs bump the accessed value to the highest hole above it (or a new top), new STOREs take the highest hole (or extend) | `Θ(N^3 log N)` empirical |

Classic DMD and DMD-live are computed directly on the L2 event stream
(Fenwick-tree-indexed LRU, `O(log T)` per op). Tombstone lowers L2 to
L3 via `bytedmd_ir.compile_tombstone` and applies
`sum ceil(sqrt(addr))` on L3 LOADs.

## The three IR levels

- **L1** — Python source.
- **L2** — abstract IR: `LOAD(var)`, `STORE(var)`, `OP(name, in, out)`.
- **L3** — concrete IR: same events with `addr` per variable.

## Results

### Cache-oblivious RMM (8-way)

|  N  | Classic DMD  | DMD-live    | Tombstone     |
|----:|-------------:|------------:|--------------:|
|   4 |        1,043 |         689 |           912 |
|   8 |       13,047 |       7,773 |        11,582 |
|  16 |      154,251 |      80,716 |       131,742 |
|  32 |    1,779,356 |     794,969 |     1,445,402 |
|  64 |   20,291,116 |   7,554,413 |    15,768,636 |

### Tiled matmul (one level, tile = ⌈√N⌉)

|  N  | Classic DMD  | DMD-live    | Tombstone     |
|----:|-------------:|------------:|--------------:|
|   4 |        1,000 |         644 |           902 |
|   8 |       12,368 |       7,210 |        11,250 |
|  16 |      143,280 |      74,560 |       122,699 |
|  32 |    1,740,310 |     790,183 |     1,500,333 |
|  64 |   19,737,581 |   7,917,595 |    17,264,621 |

On both algorithms, `DMD-live ≤ Tombstone ≤ Classic DMD` at every tested
N. Tombstone sits ≈ 1.5–2.2× above DMD-live and 10–20% below Classic DMD
across the whole range — matching the Gemini doc's prediction that
Tombstone preserves `Θ(N^3 log N)` with a moderately higher constant.

**Asymptotic fits**:
`Classic ≈ 9.6 · N^{3.5}`, `Live ≈ 4.8 · N^3 log₂ N`, `Tombstone ≈
7–10 · N^3 log₂ N`.

See **[REPORT.md](REPORT.md)** for the full writeup with the physical
picture, derivation, and a note on why the earlier stationary `min_heap`
allocator landed above Classic DMD (and why the mobile variant fixes
that).

## Reproducibility

```bash
uv run pytest test_bytedmd_ir.py
uv run --script envelope.py 4,8,16,32,64
```
