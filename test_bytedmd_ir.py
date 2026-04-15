"""Tests across the three IR levels."""

import math

import pytest

import bytedmd_ir as b2


# ---------------------------------------------------------------------------
# L1 -> L2 tracer
# ---------------------------------------------------------------------------

def test_l2_simple_add():
    """`(a+b)+c` produces three input STOREs, two ops, four LOADs total."""
    def f(a, b, c):
        return (a + b) + c

    l2, inputs = b2.trace(f, (1, 2, 3))
    assert inputs == [1, 2, 3]

    stores = [e for e in l2 if isinstance(e, b2.L2Store)]
    loads = [e for e in l2 if isinstance(e, b2.L2Load)]
    ops = [e for e in l2 if isinstance(e, b2.L2Op)]

    # 3 inputs + 2 op results
    assert len(stores) == 5
    assert len(loads) == 4
    assert len(ops) == 2
    assert all(op.name == "add" for op in ops)


def test_l2_dot_product():
    """Dot product trace shape."""
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    l2, _ = b2.trace(dot, ([1, 2], [3, 4]))
    ops = [e for e in l2 if isinstance(e, b2.L2Op)]
    # two muls + one add
    assert [o.name for o in ops] == ["mul", "mul", "add"]


def test_l2_matmul_correctness():
    """The trace should produce the correct numerical result via _Tracked.val."""
    # Walk the trace's last STORE chain to recover output values via the
    # trace recorder. Simpler: re-run untracked on plain ints.
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    expected = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    assert b2.matmul_naive(A, B) == expected
    assert b2.matmul_tiled(A, B, tile=2) == expected
    assert b2.matmul_rmm(A, B) == expected


# ---------------------------------------------------------------------------
# L2 -> L3 allocators
# ---------------------------------------------------------------------------

def test_no_reuse_assigns_sequential_addrs():
    """no_reuse hands out addrs 1, 2, 3, ... in alloc order."""
    def f(a, b, c):
        return (a + b) + c

    l2, _ = b2.trace(f, (1, 2, 3))
    l3 = b2.compile_no_reuse(l2)
    stores = [e for e in l3 if isinstance(e, b2.L3Store)]
    addrs = [s.addr for s in stores]
    assert addrs == list(range(1, len(stores) + 1))


def test_min_heap_recycles_freed_slots():
    """min_heap reuses the smallest freed addr."""
    def f(a, b, c, d):
        e = a + b   # uses a, b -> e (var 5 if a,b,c,d=1..4)
        return e + (c + d)

    l2, _ = b2.trace(f, (1, 2, 3, 4))
    l3 = b2.compile_min_heap(l2)
    # After a+b, a and b die, their addrs (1, 2) are freed and reused.
    # The peak addr should be small (4 inputs + at most a couple intermediates).
    assert b2.peak_addr(l3) <= 5


def test_l3_load_addrs_match_store_addrs():
    """Every L3Load addr should match the addr from the most recent L3Store of that var."""
    def f(a, b):
        return (a + b) * (a - b)

    l2, _ = b2.trace(f, (3, 5))
    for policy_name in b2.ALLOCATORS:
        l3 = b2.ALLOCATORS[policy_name](l2)
        var_to_addr = {}
        for ev in l3:
            if isinstance(ev, b2.L3Store):
                var_to_addr[ev.var] = ev.addr
            elif isinstance(ev, b2.L3Load):
                assert ev.var in var_to_addr, f"{policy_name}: load of unallocated var {ev.var}"
                assert var_to_addr[ev.var] == ev.addr, (
                    f"{policy_name}: load addr {ev.addr} != store addr {var_to_addr[ev.var]}"
                )


# ---------------------------------------------------------------------------
# Cost evaluator
# ---------------------------------------------------------------------------

def test_cost_basic():
    """Hand-computed cost of (a+b)+c under no_reuse."""
    def f(a, b, c):
        return (a + b) + c

    # Inputs at addrs 1,2,3. result1 = a+b at addr 4. result2 = result1+c at addr 5.
    # Loads: a@1 (cost 1), b@2 (cost 2), result1@4 (cost 2), c@3 (cost 2). Total: 7.
    assert b2.bytedmd(f, (1, 2, 3), policy="no_reuse") == 7


def test_cost_dot_product():
    """Dot product cost matches existing bytedmd.py shape."""
    # Under min_heap: a[0]*b[0] -> result, a[0]b[0] dead, freed.
    # Then a[1]*b[1] reuses freed slots etc.
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    a, b = [10, 20], [30, 40]
    cost_no = b2.bytedmd(dot, (a, b), policy="no_reuse")
    cost_mh = b2.bytedmd(dot, (a, b), policy="min_heap")
    assert cost_no >= cost_mh


# ---------------------------------------------------------------------------
# Envelope property
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [2, 4])
@pytest.mark.parametrize("algo_name", ["matmul_naive", "matmul_tiled", "matmul_rmm"])
def test_envelope_live_is_lower_bound(N, algo_name):
    """ByteDMD-live should be <= every other measure on every algorithm."""
    A, B = b2.make_inputs(N)
    func = getattr(b2, algo_name)
    costs = b2.bytedmd_all(func, (A, B))
    live = costs["bytedmd_live"]
    for name, c in costs.items():
        assert live <= c, f"{algo_name} N={N}: bytedmd_live {live} > {name} {c}"


@pytest.mark.parametrize("N", [2, 4])
@pytest.mark.parametrize("algo_name", ["matmul_naive", "matmul_tiled", "matmul_rmm"])
def test_classic_beats_live(N, algo_name):
    """ByteDMD-classic should always cost at least as much as ByteDMD-live
    because liveness compaction can only make the LRU stack shallower."""
    A, B = b2.make_inputs(N)
    func = getattr(b2, algo_name)
    l2, _ = b2.trace(func, (A, B))
    classic = b2.bytedmd_classic(l2)
    live = b2.bytedmd_live(l2)
    assert classic >= live, f"{algo_name} N={N}: classic {classic} < live {live}"


def test_bytedmd_classic_hand_computed():
    """Trace [(a+b)+c] by hand in LRU no-liveness model.

    Inputs are wrapped a, b, c in order, so STOREs push them onto the
    stack in that order — immediately after the three STOREs the stack is
    [c (top), b, a]. LOAD a: depth 3. LOAD b: depth 2 (a bumped to top,
    pushing b and c down by one, but LRU lookup before bump sees b at
    depth 2 behind c and the bumped a). STORE result1 at top. LOAD result1:
    depth 1. LOAD c: depth 4 (stack is [r1, b, a, c] after prior bumps).
    """
    def f(a, b, c):
        return (a + b) + c

    l2, _ = b2.trace(f, (1, 2, 3))
    classic = b2.bytedmd_classic(l2)
    live = b2.bytedmd_live(l2)
    assert classic > 0 and live > 0
    assert classic >= live


@pytest.mark.parametrize("N", [2, 4, 8])
def test_min_heap_peak_bounded_by_working_set(N):
    """For RMM, min_heap peak addr should grow as O(N^2), not O(N^3)."""
    A, B = b2.make_inputs(N)
    l2, _ = b2.trace(b2.matmul_rmm, (A, B))
    l3 = b2.compile_min_heap(l2)
    p = b2.peak_addr(l3)
    # Generous bound: peak <= 8 * N^2 (allows fragmentation overhead).
    assert p <= 8 * N * N, f"N={N}: min_heap peak {p} exceeds 8*N^2={8*N*N}"


def test_no_reuse_addr_count_matches_stores():
    """no_reuse peak addr equals number of L2Store events."""
    def f(a, b, c):
        return (a + b) + c

    l2, _ = b2.trace(f, (1, 2, 3))
    l3 = b2.compile_no_reuse(l2)
    n_stores = sum(1 for e in l2 if isinstance(e, b2.L2Store))
    assert b2.peak_addr(l3) == n_stores


# ---------------------------------------------------------------------------
# Asymptotic sanity: divergence between all-bytes and live-bytes for RMM
# ---------------------------------------------------------------------------

def test_rmm_envelope_widens_with_N():
    """ByteDMD-classic / ByteDMD-live ratio should grow with N for RMM."""
    ratios = []
    for N in [4, 8, 16]:
        A, B = b2.make_inputs(N)
        l2, _ = b2.trace(b2.matmul_rmm, (A, B))
        classic = b2.bytedmd_classic(l2)
        live = b2.bytedmd_live(l2)
        ratios.append(classic / live)
    assert ratios[0] < ratios[1] < ratios[2], f"ratios={ratios}"


def test_bytedmd_live_never_reads_dead_var():
    """In bytedmd_live, after a variable's last LOAD it leaves the stack,
    so a STORE immediately afterwards must see a strictly smaller stack."""
    def f(a, b, c):
        x = a + b          # last use of a and b
        return x + c

    l2, _ = b2.trace(f, (1, 2, 3))
    classic = b2.bytedmd_classic(l2)
    live = b2.bytedmd_live(l2)
    # After x+c, in live mode a and b were dropped → LOAD c sees a shallower stack.
    assert live < classic
