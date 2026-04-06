# ByteDMD Tracing Methods: Consolidation Report

<begin> notes by yaroslavvb
06apr26 I generated multiple implementations by using Gemini DeepThink with prompt (Find potential holes in this implementation and suggest ways to fix it), And then instructing others to implement the fixes. The result was multiple implementations analyzed below. This consultation report consolidates it back into two implementations. 
(</end>> notes by yaroslav)

## Current Inventory

The repo currently ships **seven** tracer implementations (~3,800 lines total). They split along three orthogonal axes: tracking mechanism, IR construction, and strictness.

| File | Lines | Mechanism | IR | Strictness |
|------|------:|-----------|-----|------------|
| `bytedmd.py` | 279 | Proxy + dunders | None | Soft (`assert_noescape` is catchable) |
| `bytedmd_fx.py` | 456 | Proxy + dunders | torch.fx | None |
| `bytedmd_fx_nopytorch.py` | 541 | Proxy + dunders | Custom Graph/Node | None |
| `bytedmd_fx2.py` | 541 | Proxy + dunders | Custom Graph/Node | None |
| `bytedmd_fx_strict.py` | 607 | Proxy + dunders | torch.fx | AST validator |
| `bytedmd_fx_strict_nopytorch.py` | 586 | Proxy + dunders | Custom Graph/Node | AST validator |
| `bytedmd_settrace.py` | 813 | `sys.settrace` opcodes | None | Bytecode-level (no escape) |

The "FX" variants are the same proxy tracer wrapped to additionally emit a graph IR (`torch.fx.Graph` or a hand-rolled equivalent). The "strict" variants add an AST validator that statically rejects functions containing dangerous syntax. The settrace tracer is a fundamentally different mechanism that intercepts every CPython bytecode instruction.

## The Six Escape Hatches

These are the adversarial loopholes documented across `test_gotchas.py`, `test_escape_hatches.py`, and `test_escape_hatches_settrace.py`:

| # | Hatch | Example | Why proxy misses it |
|---|-------|---------|---------------------|
| 1 | **Local/global state** | `for x in [10]*1000: s += x` | Proxy only wraps function args; locally constructed lists never invoke dunders |
| 2 | **Exception side-channels** | `try: 1/(a-i) except ZeroDivisionError` | Eager evaluation lets the user trigger native exceptions to probe values without `__bool__` |
| 3 | **Catching tracer assertions** | `try: if a > 0: ... except AssertionError` | Strict checker raises AssertionError, but try/except swallows it |
| 4 | **Identity/introspection** | `if a is b`, `type(a) is int`, `id(a)` | C-level operators bypass dunder dispatch |
| 5 | **Stringification** | `f"{a}"`, `str(a)`, `repr(a)` | Must return native strings, branchable on |
| 6 | **Math coercions** | `math.trunc(a)`, `math.ceil(a)`, `round(a)` | Use parallel `__trunc__`/`__ceil__`/`__round__` dunders that proxies often forget |

## How Each Implementation Handles the Hatches

### `bytedmd.py` (proxy + dunders, soft)

- **Catches**: arithmetic, comparisons, indexing via `__getitem__`, branching via `__bool__`, `range`/indexing via `__index__`
- **Misses**: all six escape hatches above
- **Failure mode**: silently undercounts. The user gets a low cost number with no warning.
- **Author overhead**: ~zero. Write Python normally; the tracer handles the common cases.

### `bytedmd_fx*.py` (proxy + dunders + IR)

- **Catches**: same as `bytedmd.py`
- **Misses**: same as `bytedmd.py`
- **Adds**: an inspectable graph IR for debugging. The trace numbers are identical to `bytedmd.py`.
- **Failure mode**: same silent undercounting, but you can `print(format_ir(ctx))` to see what was actually counted.
- **Author overhead**: zero, plus an optional graph to read.

### `bytedmd_fx_strict*.py` (proxy + dunders + AST validator)

- **Catches**: refuses to trace functions containing try/except, `is`/`is not`, `type/isinstance/id/str/repr/round`, `math.trunc/ceil/floor`, f-strings, list literals with non-None constants, `__class__`.
- **Misses**: clever escape hatches that don't use these exact syntactic forms (e.g., custom helper functions imported from another module, since the AST validator only sees the source of the entry function).
- **Failure mode**: loud — refuses with `AssertionError` and a message naming the violation.
- **Author overhead**: substantial. Cannot use try/except, f-strings, `isinstance`, `type()`, `round()`, `math.trunc/ceil/floor`, or `[constant] * N`. Algorithm authors must rewrite common idioms.

### `bytedmd_settrace.py` (bytecode-level)

- **Catches**: every read at the bytecode level. All six hatches are caught structurally because `LOAD_FAST`, `BINARY_OP`, `IS_OP`, `CALL_FUNCTION`, `FORMAT_VALUE`, `FOR_ITER` are all intercepted.
- **Misses**: nothing in pure Python. C extensions that don't go through Python frames are still invisible (e.g., `numpy.add(arr, 1)` runs entirely in C), but neither tracer can see those.
- **Failure mode**: cost grows monotonically with work; no silent under-counting. Trace numbers are different from the proxy tracer because the bytecode model treats eval-stack temporaries as free (no per-`BINARY_OP` slot), so absolute values diverge by a roughly constant factor.
- **Author overhead**: zero. Write Python normally. But ~10–50x slower because every opcode triggers a Python callback.

## Trade-off Summary

| Property | proxy (`bytedmd.py`) | proxy + AST strict | settrace |
|----------|---------------------|-------------------|----------|
| Speed | Fast (~1x baseline) | Fast (validation is one-time) | Slow (~10–50x) |
| Adversarial robustness | Low | Medium (catches syntactic hatches) | High (catches all bytecode-level reads) |
| Author overhead | None | High (forbids common idioms) | None |
| Trace stability under refactor | Stable (proxy follows dunders) | Stable | Stable |
| Graph IR for debugging | No (yes in `_fx` variants) | Yes (`_fx_strict` variants) | No (could be added) |
| Lines of code | 279 | 586–607 | 813 |
| Dependencies | none | none / torch | none |
| Catches escape hatch 1 (local arrays) | ✗ | ✗ refuses | ✓ |
| Catches escape hatch 2 (exception probe) | ✗ | ✗ refuses | ✓ |
| Catches escape hatch 3 (catch AssertionError) | n/a | ✗ refuses | ✓ |
| Catches escape hatch 4 (identity/type) | ✗ | ✗ refuses | ✓ |
| Catches escape hatch 5 (stringification) | ✗ | ✗ refuses | ✓ |
| Catches escape hatch 6 (math coercion) | ✗ | ✗ refuses | ✓ |

The "✗ refuses" entries for the strict-AST tracer are important: it doesn't *trace* those programs — it *rejects* them. This is qualitatively different from settrace, which traces every program correctly.

## Cost Equivalence Between Methods

The proxy and settrace tracers do **not** produce identical trace numbers on the same input, even on completely benign code. The reason is that the proxy allocates a fresh LRU slot per intermediate `BINARY_OP` result, while settrace treats all eval-stack temporaries as free. For example, `b + c` with `b` at depth 3 and `c` at depth 2:

- **Proxy**: trace `[3, 2]`, cost 4
- **Settrace**: trace `[3, 3]` (after b moves to MRU before c is read), cost 4

The costs **happen to match** for this case because the depths land on the same `ceil(sqrt)` step. They don't always match. A refactoring that hoists a sub-expression into a named local will change the proxy trace (the temp is now a named slot) but not the settrace trace (it was already a named slot in spirit).

For the proposed two-method system to "agree most of the time," we have two options:

**Option A**: Document the divergence and provide a comparison helper. Users see a single cost number from the regular tracer; if they want to verify it's not undercounting, they run the strict tracer and compare. They should be within ~2x of each other on benign code.

**Option B**: Make the regular tracer charge identically to the strict tracer. Drop the proxy's per-temp slot allocation. This breaks the existing tests in `test_bytedmd.py` (trace `[3, 2]` becomes `[3, 3]`) but makes the two cost values identical on benign code, modulo opcodes the regular tracer doesn't intercept (e.g., `is`).

**Recommendation**: **Option A**. The proxy's per-temp slot model is mature, well-tested, and maps cleanly to existing benchmarks. Bringing it into bytecode parity would break 19 existing tests and benchmark numbers (linalg, microGPT, attention), and the cost values are already informative even if they don't exactly match.

## Recommendation: Two-Method Consolidation

### Method 1: `bytedmd.bytedmd()` — the regular tracer

**Use it 95% of the time.** Fast, simple, easy to understand. Suitable for everyday algorithm exploration, benchmarks against well-behaved code, and integration with the existing test/benchmark suite.

**Implementation**:
- **Keep `bytedmd.py` as-is.** It's the canonical 279-line proxy tracer with dunder interception. This is the "regular" path.
- **Delete `bytedmd_fx.py`, `bytedmd_fx2.py`, `bytedmd_fx_nopytorch.py`.** These are duplicates that add only a graph IR for debugging. If anyone needs IR inspection, fold the simplest one (`bytedmd_fx_nopytorch.py`) into a subcommand or import-on-demand helper. The proxy semantics they wrap are identical to `bytedmd.py`.
- **Delete `bytedmd_fx_strict.py`** (the torch.fx-dependent strict variant). It's redundant with `bytedmd_fx_strict_nopytorch.py` and adds a torch dependency.
- **Remove `assert_noescape` from `bytedmd.py`.** It's a half-measure that gives users false confidence: it's catchable via try/except and only blocks a small subset of escape hatches. Replace it with a docstring pointing to the strict tracer.

After consolidation: one 200–250 line file, no dependencies, no surprises.

### Method 2: `bytedmd_strict.bytedmd()` — the strict tracer

**Use it for benchmarks against adversarial code, security audits, or to verify the regular tracer is not undercounting.** Slow but provably correct: every read in pure Python is captured.

**Implementation**:
- **Rename `bytedmd_settrace.py` to `bytedmd_strict.py`.** The settrace mechanism is the right substrate; the name should reflect the user-facing guarantee, not the implementation detail.
- **Delete `bytedmd_fx_strict_nopytorch.py`.** The AST-validation approach is fundamentally limited: it only inspects the entry function's source, doesn't follow imports, and forbids common Python idioms. The settrace tracer subsumes it without imposing any author overhead.
- **Add a `verify(func, args)` helper** that runs both tracers and warns if their costs diverge by more than some threshold (e.g., 3x). This is the practical tool authors will use to sanity-check their regular-tracer benchmarks.

After consolidation: one ~800 line file, no dependencies, no escape hatches.

### File layout after consolidation

```
bytedmd.py           # regular tracer, proxy-based, 250 lines
bytedmd_strict.py    # strict tracer, settrace-based, 800 lines (renamed)
test_bytedmd.py
test_bytedmd_strict.py    # renamed from test_bytedmd_settrace.py
test_escape_hatches.py    # repurposed for the strict tracer
test_gotchas.py           # documents the regular tracer's known gotchas
```

Files to **delete** (5 files, ~2,400 lines):
- `bytedmd_fx.py`
- `bytedmd_fx2.py`
- `bytedmd_fx_nopytorch.py`
- `bytedmd_fx_strict.py`
- `bytedmd_fx_strict_nopytorch.py`

Plus their test files: `test_bytedmd_fx.py`, `test_bytedmd_fx2.py`, `test_bytedmd_fx_strict.py`, `test_bytedmd_fx_strict_nopytorch.py`.

### Public API after consolidation

```python
# Regular: fast, easy, may undercount adversarial code
from bytedmd import bytedmd, traced_eval, trace_to_bytedmd

# Strict: slow, robust, no escape hatches
from bytedmd_strict import bytedmd as bytedmd_strict, traced_eval as traced_eval_strict

# Sanity check: warns if the two tracers disagree significantly
from bytedmd_strict import verify
verify(my_function, args)
# > Regular: 7047
# > Strict:  7142  (1.01x — within tolerance)
```

## Migration Steps

1. **Create `bytedmd_strict.py`** by renaming `bytedmd_settrace.py` and adding the `verify()` helper.
2. **Update `test_bytedmd_settrace.py`** to import from `bytedmd_strict` and rename to `test_bytedmd_strict.py`.
3. **Strip `assert_noescape` from `bytedmd.py`** and update the docstring to point to `bytedmd_strict` for adversarial scenarios.
4. **Delete the FX variants and their tests.**
5. **Run the full benchmark suite** (`benchmarks/`) to confirm `bytedmd.py` numbers are unchanged.
6. **Add a `verify()` call to one or two existing benchmarks** as a smoke test that the two tracers stay within tolerance on real code.
7. **Update README** to describe the two-method model.

## Why Two Tracers, Not One

The fundamental tension: bytecode tracing is **correct but slow** (~10–50x), proxy tracing is **fast but incomplete**. There is no Python implementation that is both fast and complete (this would require either a C-level tracer or AOT analysis, both of which are beyond the scope of a pure-Python research tool).

By providing both, users get the best of both worlds:
- Iterate quickly with the regular tracer during algorithm development
- Verify with the strict tracer before publishing benchmark numbers
- Use the strict tracer exclusively when running adversarial or unfamiliar code

The single-tracer alternatives (proxy-only or settrace-only) each fail for half the use cases. The AST-validator alternative imposes author overhead that scales with the strictness, eventually forcing users to rewrite Python in a restricted dialect. Two complementary tracers — one fast, one robust — is the simplest design that covers the full workflow.
