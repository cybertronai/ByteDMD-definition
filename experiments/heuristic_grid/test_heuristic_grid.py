from experiments.heuristic_grid.algorithms import (
    build_algorithm_specs,
    blocked_transpose,
    flash_attention,
    iterative_fft,
    jacobi_stencil_naive,
    jacobi_stencil_recursive,
    make_matrix,
    make_vector,
    matvec,
    naive_transpose,
    recursive_fft,
    recursive_transpose,
    regular_attention,
)
from experiments.heuristic_grid.measure import measure_function
from experiments.heuristic_grid.run_experiment import CLASSIC, LIVE, TARGET, collect_results


def test_transpose_variants_match():
    matrix = make_matrix(8)
    expected = naive_transpose(matrix)

    assert blocked_transpose(matrix, block=4) == expected
    assert recursive_transpose(matrix, leaf=4) == expected


def test_fft_variants_match_on_small_input():
    vector = [complex(v, 0.5 * v) for v in make_vector(8)]
    iterative = iterative_fft(vector)
    recursive = recursive_fft(vector)

    assert all(abs(a - b) < 1e-9 for a, b in zip(iterative, recursive))


def test_recursive_jacobi_matches_naive():
    matrix = make_matrix(8)

    assert jacobi_stencil_recursive(matrix, leaf=4) == jacobi_stencil_naive(matrix)


def test_measure_function_orders_simple_matvec():
    args = (make_matrix(8), make_vector(8))
    classic = measure_function(matvec, args, strategy="unmanaged")
    manual = measure_function(matvec, args, strategy="tombstone")
    live = measure_function(matvec, args, strategy="aggressive")

    assert classic["cost_discrete"] >= manual["cost_discrete"] >= live["cost_discrete"]
    assert classic["n_reads"] == manual["n_reads"] == live["n_reads"]


def test_flash_attention_beats_regular_attention_on_selected_workload():
    args = (
        make_matrix(32, 4),
        make_matrix(32, 4, offset=1000),
        make_matrix(32, 4, offset=2000),
    )
    regular = measure_function(regular_attention, args, strategy="tombstone")
    flash = measure_function(lambda Q, K, V: flash_attention(Q, K, V, bq=8, bk=4), args, strategy="tombstone")

    assert flash["cost_discrete"] < regular["cost_discrete"]


def test_collect_results_has_expected_shape_and_budget():
    results = collect_results()
    rows = results["algorithms"]

    assert len(rows) == len(build_algorithm_specs())
    assert results["overall_max_cell_seconds"] < 10.0
    assert all(TARGET in row and CLASSIC in row and LIVE in row for row in rows)
