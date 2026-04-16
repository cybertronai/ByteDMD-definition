import math

from experiments.matmul_hierarchy.hierarchy import (
    abstract_reuse_depths,
    bytedmd_cost,
    compile_concrete_trace,
    concrete_address_depths,
    memory_curve,
    numeric_matmul,
    trace_matmul_program,
)


def _make_matrix(n, offset):
    return [[offset + i * n + j + 1 for j in range(n)] for i in range(n)]


def test_all_algorithms_produce_correct_small_result():
    a = _make_matrix(2, 0)
    b = _make_matrix(2, 10)
    expected = numeric_matmul(a, b)
    for algorithm in ("tiled", "recursive", "strassen"):
        program = trace_matmul_program(algorithm, a, b, tile_size=2)
        assert program.result == expected


def test_three_levels_emit_expected_shapes():
    a = _make_matrix(2, 0)
    b = _make_matrix(2, 10)
    program = trace_matmul_program("recursive", a, b)

    assert program.abstract_accesses
    assert {access.kind for access in program.abstract_accesses} == {"load", "store"}

    concrete = compile_concrete_trace(program, policy="tombstone")
    assert len(concrete) == len(program.abstract_accesses)
    assert all(isinstance(access.address, int) for access in concrete)
    assert all(access.policy == "tombstone" for access in concrete)


def test_tombstone_stays_inside_abstract_envelope():
    a = _make_matrix(4, 0)
    b = _make_matrix(4, 100)
    program = trace_matmul_program("recursive", a, b)

    total_depths = abstract_reuse_depths(program, live_only=False)
    live_depths = abstract_reuse_depths(program, live_only=True)
    tombstone_depths = concrete_address_depths(program, policy="tombstone")

    total_cost = bytedmd_cost(total_depths)
    live_cost = bytedmd_cost(live_depths)
    tombstone_cost = bytedmd_cost(tombstone_depths)

    assert live_cost <= tombstone_cost <= total_cost


def test_memory_curve_is_monotone_nonincreasing():
    depths = [8, 3, 5, 1, 2, 8]
    curve = memory_curve(depths, [1, 2, 4, 8, 16])
    assert curve == sorted(curve, reverse=True)
