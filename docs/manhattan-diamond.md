# The Manhattan-Diamond Model

*A single-core formalization of Bill Dally's PECM fused with the
Geometric Stack cost model used in the ByteDMD grid experiments.*

## 1. Motivation

Bill Dally ([*On the Model of Computation*, CACM
2022](https://cacm.acm.org/opinion/on-the-model-of-computation-point/))
argues that the RAM/PRAM tradition of charging every memory access `O(1)`
is now physically wrong: a 32-bit addition on modern silicon costs
~20 fJ, while moving two 32-bit words 1 mm costs ~1.9 pJ — a 64,000× gap
to off-chip DRAM, and still a 100× gap across a single chip. He
proposes the **Parallel Explicit Communication Model (PECM)**: keep
arithmetic unit-cost, but price every memory reference by the physical
distance the data has to travel to reach the ALU.

The Geometric Stack model (Ding and Smith, 2022; extended in this
repository as *ByteDMD*) answers the question "*how far?*" under a
specific, reasonable assumption: an optimal compiler packs the active
working set into a 2-D region around the ALU, and physical distance is
Manhattan distance on that 2-D grid. Under that assumption, the
read cost of the `d`-th most-recently-used cell is `⌈√d⌉`, and the
total data-movement cost of a program is `Σ ⌈√d⌉` across every memory
read.

The **Manhattan-Diamond model** is the synthesis: a concrete
single-core, single-function communication model that makes PECM
evaluable on real Python traces, with two physically-separated memory
arenas whose geometry is fixed by the Manhattan metric.

## 2. The Core

- One compute unit (ALU) sits at the origin `(0, 0)` of a 2-D integer
  grid.
- Arithmetic operations are **free** — they happen entirely inside
  the core.
- Writes to memory are **free** — they terminate at a cell but do not
  charge wire energy (they are modelled as driving the output line
  once, amortised against the compute cost).
- **Reads are the only priced events**: reading a cell at Manhattan
  distance `r` costs `r` units. All cost is integer.

## 3. The Two Memory Arenas

The core sits between two independent arenas packed outward along
opposite half-lanes:

```
                 arg stack            (read-only inputs)
                 ────────────────────
                 …  a₅   a₄   a₃   a₂   a₁           ← top of arg = depth 1
                                                        closest to ALU

                           [ ALU core ]              ← compute origin

                 s₁   s₂   s₃   s₄   s₅   …           ← top of scratch = depth 1
                 ────────────────────
                 scratch stack        (read-write temporaries + outputs)
```

Both arenas have their **own** Manhattan-disc origin at logical
address 1 and grow outward from there. They are independent: cells at
arg-depth 3 and scratch-depth 3 both cost `⌈√3⌉ = 2`, but they are
physically different addresses connected to the core by different
wire runs.

This matches Dally's PECM intuition that arguments arrive from a
different direction than intermediates do (e.g., from a distributed
SRAM router on one side, from a local register file on the other),
so they should be priced against independent geometries.

### 3.1 Why "Diamond"?

Under the Manhattan metric, the set of cells at distance `≤ r` from
the origin forms a diamond (an L₁ ball): the rotated square
`{(x, y) : |x| + |y| ≤ r}`. It contains `2r² + 2r + 1` integer cells.
Packing `k` cells in the cheapest region needs radius
`r ≈ √(k/2) = O(√k)`. So when an arena holds the `k` most-recently-
used variables, the farthest one lives at Manhattan distance
proportional to `√k`. The cost model rounds this to `⌈√k⌉` so every
address has an integer Manhattan distance.

The two arenas each form their own diamond rooted at the ALU, and
those two diamonds meet at the origin — the overall picture is two
half-diamonds fused at the core, hence *Manhattan-Diamond*.

## 4. Pricing Semantics

Let `addr` be the 1-indexed position of a cell within its arena. The
read cost is

```
cost(addr) = ⌈√addr⌉ = isqrt(addr - 1) + 1
```

This is the closed-form radius of a sufficiently-large Manhattan
disc; the exact geometric identity `cells(r) = 2r² + 2r + 1` gives
`r = ⌈√addr⌉` for `addr ≤ 2r² + 2r + 1`.

### 4.1 Events

Every trace is a sequence of primitive events emitted as the function
executes:

| Event                | Effect                                                           | Cost            |
|----------------------|------------------------------------------------------------------|-----------------|
| `alloc_arg(size)`    | Reserve `size` cells at the top of the argument arena.           | 0               |
| `alloc(size)`        | Reserve `size` cells at the top of the scratch arena.            | 0               |
| `touch_arg(addr)`    | Read a cell at position `addr` in the argument arena.            | `⌈√addr⌉`       |
| `touch(addr)`        | Read a cell at position `addr` in the scratch arena.             | `⌈√addr⌉`       |
| `write(addr)`        | Store a value at position `addr` (scratch arena only).           | 0               |
| `read_output()`      | Epilogue — the caller reads each output cell once from scratch.  | `Σ ⌈√addr⌉`     |

The total ByteDMD cost of a function is the sum of all event costs.
Writes and allocations are free; reads are priced by Manhattan
distance in their respective arena; the epilogue models the caller
reading the result back into its own scope.

### 4.2 Addressing conventions

- **Argument arena.** Inputs to the function are packed left-to-right
  at function entry. The first argument's first element sits at
  arg-addr 1, the second at 2, and so on. Positions are **fixed** for
  the life of the call — they do not drift, compact, or slide.
- **Scratch arena.** A simple bump pointer. The first cell allocated
  lives at scratch-addr 1 (closest to the core). Subsequent
  allocations stack outward. In the strictest "geometric stack"
  variant, the scratch arena is managed as an LRU stack with
  liveness compaction: dead cells are removed and all cells above
  them slide down one slot. In the "manual" variant used in the
  grid, the bump pointer is never reclaimed — the programmer is
  responsible for choosing which cells deserve low addresses.

Both conventions price reads identically; they differ only in how
scratch addresses are assigned. Compare `experiments/grid/manual.py`'s
`Allocator` (manual bump pointer) with `bytedmd_ir.bytedmd_live`
(LRU with liveness compaction).

## 5. The Single-Function, Single-Core Contract

The model evaluates *one* function at a time on *one* core. The
contract is:

1. **Entry.** Caller hands a list of input values. The tracer packs
   them into the argument arena in source order (first arg at
   addr 1, subsequent at 2, 3, …). Multi-cell arguments (lists,
   arrays) occupy contiguous slots.
2. **Execution.** The function performs a sequence of scalar
   arithmetic operations. Each operation reads `1..k` operands, does
   a unit-cost computation on them, and stores the result into a
   fresh cell on the scratch arena. All reads are priced by the
   formula above.
3. **Epilogue.** Whatever the function returns is read out of the
   scratch arena by the caller. Each output cell is touched once
   more, priced by its scratch-arena position.
4. **Scratch is infinite.** There is no capacity limit: the scratch
   arena extends outward indefinitely, but each cell pays its
   Manhattan distance. A programmer who wastes scratch addresses on
   cold cells pushes hot ones farther from the core.

## 6. First-Read Promotion (Two-Stack Dynamics)

An argument that is read *twice* must not pay twice for its original
arg-arena position — physically, once data reaches the core it has
already been moved; a hypothetical local register file captures it
for reuse.

The model's **argument promotion** rule:

- The first `touch_arg(addr)` of a particular argument cell prices the
  read at its arg-arena depth `addr`.
- Immediately after that read, the cell is **promoted** onto the top
  of the scratch arena (at scratch-depth 1, the cheapest slot) as if
  it had just been produced.
- Every subsequent read of that logical value is priced against the
  scratch arena — it participates in LRU bumping, liveness
  compaction, and all other scratch dynamics like any intermediate.

This captures Dally's observation that the physical energy of moving
an operand once is paid at the original distance; moving it again
(from the register file) is cheap. It also makes reused arguments
first-class citizens of the scratch arena without requiring the
programmer to insert an explicit `assign(arg, scratch)` step.

The IR tags reads accordingly — `READ v@arg=5` for an un-promoted
first read, `READ v@geom=3` for a subsequent read after promotion.

## 7. From Cost Model to Dally Joules

Under 45 nm process assumptions from Dally's CACM article:

- Cost of an add: ~20 fJ.
- Cost to move 64 bits by 1 mm: ~1.9 pJ.

One ByteDMD cost unit corresponds to one read that traverses a wire
of Manhattan length `⌈√d⌉`. Calibrating against "an add is worth 10 µm
of movement":

```
1 ByteDMD cost unit  ≈  ⌈√d⌉ Manhattan hops  ≈  ⌈√d⌉ × 10 µm
                      ≈  ⌈√d⌉ × 20 fJ of dynamic on-chip energy
```

So a function with total cost `C` dissipates roughly `20·C` fJ of
data-movement energy under the Dally calibration. This is the
quantity that the grid experiment minimises when it searches for the
best manual placement.

## 8. Relation to Classical Models

| Model                               | Memory cost                  | Movement physics | Stack semantics       |
|-------------------------------------|------------------------------|------------------|-----------------------|
| RAM / PRAM                          | `O(1)` per access            | Ignored          | —                     |
| Hong-Kung red/blue pebble game      | Counts I/O events only       | Discrete tiers   | Fast / slow partition |
| Aggarwal-Chandra-Snir HMM `f(x)=√x` | `Σ √d`, no liveness          | 2-D planar       | Single LRU stack      |
| Ding-Smith DMD (2022)               | `Σ √d` on raw LRU stack      | 2-D planar       | Single LRU stack      |
| **ByteDMD / Manhattan-Diamond**     | `Σ ⌈√d⌉` with live compaction + 2-stack promotion | 2-D Manhattan, two arenas fused at the core | Two stacks: fixed arg + LRU scratch |

The Manhattan-Diamond model is HMM with `f(x) = √x` plus two
physically-separated arenas plus Bélády-optimal liveness on scratch
(Faldu et al. 2017's *live-distance* metric) plus explicit arg
promotion. It is the minimum formalism needed to make PECM
actually *runnable* on a Python trace.

## 9. Bounds

Two theoretical results bound the cost `C_manual(f)` of any manual
placement for a given function `f`:

- **Lower bound** ([gemini/tarjan-bytedmd-lower-bound.md](../gemini/tarjan-bytedmd-lower-bound.md)).
  For any correct schedule,
  `C_manual(f) ≥ (2 / (3√3)) · C_live(f) ≈ 0.3849 · C_live(f)`
  where `C_live(f)` is the cost under a liveness-compacted LRU
  scratch arena. Proof via Sleator-Tarjan 1985 lifted to the
  continuous `√d` cost.
- **Upper bound** ([gemini/bytedmd-upper-bound.md](../gemini/bytedmd-upper-bound.md)).
  `C_manual(f) ≤ 4.0 · C_live(f)` for an optimally-managed DMA
  scratchpad. Proof via competitive-caching analysis.

Together: `0.3849 · C_live(f) ≤ C_manual(f) ≤ 4.0 · C_live(f)`. Every
row of the grid experiment empirically honours this sandwich — the
tightest is `stencil_time_diamond` at `0.591 · C_live`, still 53 %
above the Tarjan floor.

## 10. Implementation

The reference implementation is the `Allocator` class in
[`experiments/grid/manual.py`](../experiments/grid/manual.py):

```python
class Allocator:
    def __init__(self, logging: bool = False) -> None:
        self.cost = 0
        self.ptr = 1         # scratch bump pointer
        self.arg_ptr = 1     # arg bump pointer
        ...

    def alloc(self, size: int) -> int:
        addr, self.ptr = self.ptr, self.ptr + size
        return addr

    def alloc_arg(self, size: int) -> int:
        addr, self.arg_ptr = self.arg_ptr, self.arg_ptr + size
        return addr

    def touch(self, addr: int) -> None:
        self.cost += math.isqrt(max(0, addr - 1)) + 1

    def touch_arg(self, addr: int) -> None:
        self.cost += math.isqrt(max(0, addr - 1)) + 1

    def write(self, addr: int) -> None:
        pass   # free; only logged for visualization
```

Two independent bump pointers, identical `⌈√d⌉` pricing on each, free
writes. The whole model is 30 lines of Python. Every manual schedule
in the grid calls these primitives directly; the 45-algorithm
benchmark table in `experiments/grid/README.md` is the empirical test
of the Manhattan-Diamond model against classical heuristics.

## 11. Summary

The Manhattan-Diamond model is the smallest possible concretisation of
Dally's PECM that is:

- **Executable** — any Python function can be traced against it.
- **Two-arena** — inputs and intermediates live in physically distinct
  geometries, as they do on real hardware.
- **Calibrated** — a cost unit corresponds to a specific number of
  femtojoules under published 45 nm numbers.
- **Bracketed** — manual placements are provably within `(0.385, 4.0)×`
  of the automatic LRU cost.

Implemented in ~30 lines. Benchmarked on 45 algorithms. Bounded above
and below by 40-year-old competitive-caching theorems. That's the
whole model.
