# matmul-exhaustive — n^3 matmul strategies under manual allocation

## Summary

Restricted to the (*, +) semiring (no Strassen / Winograd), ranks
strategies for matrix multiplication under manual allocation + flat
Manhattan DMD cost. Two sweeps:

- **2x2 exhaustive** — every strategy (83 total). See below.
- **4x4 catalog** — 11 representative strategies covering loop orders,
  row/column hoisting, scalar accumulation, product batching, and 2x2
  blocking. See the [4x4 section](#4x4-catalog-11-strategies) further down.

## 2x2 sweep

For 2x2 matrix multiplication restricted to the (*, +) semiring (8
scalar products + 4 scalar sums), this experiment enumerates every
strategy that can use a scratchpad under manual allocation, and ranks
them by DMD cost.

**Cost model.** Flat Manhattan distance: every read at address `a`
costs `ceil(sqrt(a))`; writes are free. Two bump-pointer allocators
run in parallel — one for arg inputs (A, B), one for scratch
(products + output C). Addresses are fixed at allocation time; nothing
is relocated. This is the same `manual` model used in
`experiments/grid/`.

**83 strategies sweep.** For each of the four output cells C[i][j] we
pick one of three assembly modes, and we include two additional
"concurrent-batched" variants that keep multiple pairs' products alive
simultaneously:

| mode       | how C[i][j] is assembled                                     | scratch reads this pair |
|------------|---------------------------------------------------------------|-----------------------:|
| `direct`   | MUL1 writes **directly to C[i][j]**; MUL2 writes to P0; ADD reads C + P0 → C | 1 P0 + 1 C = 2 |
| `indirect` | MUL1 writes to P0; ASSIGN copies P0 → C[i][j]; MUL2 overwrites P0; ADD reads C + P0 → C | 2 P0 + 1 C = 3 |
| `batched`  | MUL1 writes to P0; MUL2 writes to P1; ADD reads P0 + P1 → C   | 1 P0 + 1 P1 = 2 |

P0 and P1 are single scratch slots reused across every pair. Each C
cell is also read once in the epilogue (the caller reading the
output). 3^4 = 81 mode combinations plus 2 concurrent-batched variants
(`k_live=2`, `k_live=4`) gives 83 strategies.

**Arg layout.** A[0][0], A[0][1], A[1][0], A[1][1] at arg addrs 1..4;
B[0][0], B[0][1], B[1][0], B[1][1] at arg addrs 5..8. Every arg cell
is read exactly twice under any n^3 schedule, so arg order is
cost-neutral — total arg-read cost is always
`2 * (1+2+2+2+3+3+3+3) = 38`.

**Scratch layout.** Under optimal placement — sorting cells by read
count descending and assigning addresses 1, 2, 3, ... — which is
achievable for every strategy here because all allocated cells are
mutually live throughout the computation (no push/pop reuse is
required to hit the optimum).

## Ranked table (grouped by cost)

| rank | cost | count | example | cells + read counts |
|-----:|-----:|------:|---------|---------------------|
|   1  |  60  |  16   | `naive:bbbb` | P0×4, P1×4, C×1 ×4 |
|   2  |  61  |  32   | `naive:bbbi` | P0×5, P1×3, three C×1 and one C×2 |
|   3  |  62  |  24   | `naive:bbii` | P0×6, P1×2, two C×1 and two C×2 |
|   4  |  63  |   8   | `naive:biii` | P0×7, P1×1, one C×1 and three C×2 |
|   5  |  64  |   2   | `naive:iiii`, `batched_parallel:k_live=2` | see full table |
|   6  |  72  |   1   | `batched_parallel:k_live=4` | 8× P×1 and 4× C×1 |

Full sorted list: [`ranked_full.md`](ranked_full.md). Raw results:
[`results.json`](results.json).

## Findings

1. **16 strategies tie at the optimum, cost = 60.** They are exactly
   the 2^4 combinations where every pair is either `direct` or
   `batched` (no `indirect`). The degenerate `naive:dddd` (all direct,
   5 scratch cells) and `naive:bbbb` (all batched, 6 scratch cells)
   both land on 60.

2. **Every `indirect` pair costs exactly +1.** The read-count
   contribution an `indirect` pair adds to P0 is 2 (vs. 1 for direct
   or batched), and under optimal placement P0 sits at address 1, so
   each extra indirect pair costs 1 × (⌈√1⌉) extra. This produces the
   clean +1-per-indirect progression 60 → 61 → 62 → 63 → 64 as the
   indirect count goes 0 → 1 → 2 → 3 → 4.

3. **Keeping more products alive concurrently never helps.**
   `batched_parallel:k_live=2` costs 64 (tie with all-indirect) and
   `k_live=4` costs 72. The extra P cells dilute addresses to higher
   rings without reducing any individual read count.

4. **Arg layout is irrelevant under n^3.** Every arg cell is read
   exactly twice in any legal schedule, so permuting A and B on the
   arg stack rearranges cost contributions among identical-count
   cells and the total stays at 38.

5. **"Direct-first" is never worse than "indirect."** Since both
   modes have the same lifetime footprint and identical C-cell reads,
   but direct saves the ASSIGN's P0 read, direct strictly dominates
   indirect per pair. Any optimal strategy avoids indirect.

## 4x4 catalog (11 strategies)

At 4x4 the strategy space is too large to enumerate exhaustively
(64 scalar products, 48 sums, unbounded scratchpad layouts), so this
section is a *curated* list — eleven representative strategies that
cover the interesting axes: loop order, A-row hoisting, B-column
hoisting, scalar accumulator, batched storage, and 2x2 blocking. Each
is simulated end-to-end with an explicit bump-pointer allocator.

**Arg layout.** A @ 1..16, B @ 17..32, both row-major. The "pure arg
baseline" — every A and B cell read 4 times at its raw address — is
`4 * (1 + 3*2 + 5*3 + 7*4 + 9*5 + 7*6) = 4 * 137 = 548` units of
cost. Strategies that hoist move some of those reads into (cheaper)
scratch addresses at the cost of extra scratch reads.

### Ranked results

| rank | strategy | cost | peak_scratch | n_reads |
|-----:|----------|-----:|-------------:|--------:|
|  1 | `naive_b_col_cached`    |  739 | 21 | 256 |
|  2 | `outer_b_row_a_scalar`  |  741 | 22 | 272 |
|  3 | `naive_a_row_scalar_acc`|  801 | 22 | 272 |
|  4 | `naive_ijk_direct`      |  812 | 17 | 240 |
|  5 | `naive_jik_direct`      |  812 | 17 | 240 |
|  6 | `naive_kij_rank1`       |  812 | 17 | 240 |
|  7 | `block_2x2_direct`      |  847 | 21 | 240 |
|  8 | `batched_per_pair`      |  849 | 20 | 240 |
|  9 | `naive_a_row_cached`    |  850 | 21 | 256 |
| 10 | `naive_ijk_always_acc`  |  882 | 17 | 272 |
| 11 | `batched_all_64`        | 1352 | 80 | 240 |

Full raw output: [`results_4x4.json`](results_4x4.json),
[`ranked_4x4.md`](ranked_4x4.md).

### Findings

1. **Hoist B, not A.** B sits at arg addresses 17..32 (read cost 5–6)
   while A sits at 1..16 (read cost ≤ 4), so hoisting B's 16 cells
   saves ~261 units of arg cost and pays ~144 in scratch — net
   win. Hoisting A *costs* a few units because its arg reads were
   already cheap.

2. **Loop order is invisible.** `naive_ijk_direct`, `naive_jik_direct`,
   and `naive_kij_rank1` all cost exactly 812. Flat-addr cost depends
   only on the *read profile* per cell, and the three schedules have
   identical DAGs.

3. **The MUL1-to-C shortcut matters.** `naive_ijk_direct` (812) vs
   `naive_ijk_always_acc` (882): avoiding one tmp-assign per
   `(i, j)` pair saves 70 (the `+N*N*1` C-read × 16 pairs costs the
   difference). At 2x2 this same shortcut is worth exactly 4.

4. **Batching products is strictly a loss.** `batched_per_pair`
   (849) holds 4 live products per pair instead of 1 tmp; the extra
   cells push C to higher addresses. `batched_all_64` (1352) is the
   extreme — 64 P cells displacing C deep into the disc.

5. **2x2 blocking doesn't help at 4x4.** `block_2x2_direct` (847) is
   worse than plain `naive_ijk_direct` (812). The 4-cell tmp_block
   that carries the second 2x2 product doesn't pay for itself because
   each tmp_block cell is only read once per outer block (4 ADDs per
   block), yet lives at low addresses and displaces C.

6. **Best combined strategy: hoist B + scalar A.** `outer_b_row_a_scalar`
   (741) is only 2 units behind `naive_b_col_cached` (739). The extra
   `a_scalar` slot adds 64 reads at address 2 (cost 128) but
   eliminates a chunk of the costly A-arg reads; net difference is
   tiny. Together they show that two small caches aimed at the
   expensive side of the arg stack are worth close to what a single
   well-placed B-column cache is.

## Reproduce

```
python3 run_experiment.py   # 2x2 sweep — writes ranked_*.md, results.json
python3 run_4x4.py          # 4x4 catalog — writes ranked_4x4.md, results_4x4.json
```

## Files

- `tracer.py` — flat-addr Manhattan cost model + `evaluate_layout` + `Allocator`.
- `algorithms.py` — 2x2 mode profiles + 83-strategy enumerator.
- `algorithms_4x4.py` — 4x4 strategy simulators (11 entries).
- `run_experiment.py` / `run_4x4.py` — sweep runners + table writers.
- `ranked_full.md` / `ranked_grouped.md` — 2x2 sorted tables.
- `ranked_4x4.md` — 4x4 sorted table.
- `results.json` / `results_4x4.json` — raw output.
