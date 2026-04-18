from experiments.memory_management.algorithms import naive_matmul
from experiments.grid.algorithms import (
    blocked_cholesky,
    blocked_lu,
    blocked_qr,
    build_algorithm_specs,
    blocked_transpose,
    cholesky,
    fft_conv2d,
    fft_conv1d,
    flash_attention,
    fused_strassen,
    gaussian_elimination,
    gauss_jordan_inverse,
    householder_qr,
    iterative_fft,
    jacobi_stencil_naive,
    jacobi_stencil_recursive,
    lcs_dp,
    lu_no_pivot,
    lu_partial_pivot,
    make_linear_system,
    make_linear_system_matrix,
    make_matrix,
    make_filter,
    make_pivot_matrix,
    make_spd_matrix,
    make_vector,
    make_volume,
    matvec,
    mergesort,
    naive_transpose,
    naive_circular_conv1d,
    recursive_cholesky,
    recursive_fft,
    recursive_lu,
    recursive_transpose,
    regular_attention,
    regular_conv2d,
    spatial_conv2d,
    tsqr,
)
from experiments.grid.manual_2d import ManualTracer, measure_manual_2d
from experiments.grid.measure import SpaceDMD, measure_function, measure_space_dmd
from experiments.grid.run_experiment import CLASSIC, LIVE, SPACE, TARGET, collect_results


def _matmul_numeric(A, B):
    rows = len(A)
    inner = len(A[0])
    cols = len(B[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            out[i][j] = sum(float(A[i][k]) * float(B[k][j]) for k in range(inner))
    return out


def _transpose_numeric(A):
    return [list(col) for col in zip(*A)]


def _assert_matrix_close(left, right, tol=1e-6):
    assert len(left) == len(right)
    assert len(left[0]) == len(right[0])
    for row_left, row_right in zip(left, right):
        for value_left, value_right in zip(row_left, row_right):
            assert abs(float(value_left) - float(value_right)) < tol


def test_fused_strassen_matches_standard_output():
    a = make_matrix(8)
    b = make_matrix(8, offset=1000)

    assert fused_strassen(a, b, leaf=2) == naive_matmul(a, b)


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


def test_fft_convolution_matches_spatial_convolution():
    image = make_matrix(8)
    kernel = make_matrix(3, 3, offset=5000)
    spatial = spatial_conv2d(image, kernel)
    fft_based = fft_conv2d(image, kernel)

    for spatial_row, fft_row in zip(spatial, fft_based):
        assert all(abs(a - b) < 1e-6 for a, b in zip(spatial_row, fft_row))


def test_fft_conv1d_matches_naive_circular_convolution():
    signal = make_vector(8)
    kernel = make_vector(8, offset=5000)
    naive = naive_circular_conv1d(signal, kernel)
    fft_based = fft_conv1d(signal, kernel)

    assert all(abs(a - b) < 1e-6 for a, b in zip(naive, fft_based))


def test_mergesort_matches_python_sorted():
    values = [7, 3, 9, 1, 5, 2, 8, 4]

    assert mergesort(values) == sorted(values)


def test_lcs_dp_matches_known_example():
    seq_a = [1, 3, 4, 1, 2, 3]
    seq_b = [3, 4, 1, 2, 1, 3]

    assert lcs_dp(seq_a, seq_b) == 5


def test_gaussian_elimination_solves_known_system():
    A, b, x_true = make_linear_system(6)
    solved = gaussian_elimination(A, b)

    assert all(abs(actual - expected) < 1e-6 for actual, expected in zip(solved, x_true))


def test_gauss_jordan_inverse_recovers_identity():
    A = make_linear_system_matrix(4)
    inverse = gauss_jordan_inverse(A)

    product = [
        [sum(A[i][k] * inverse[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]

    for i in range(4):
        for j in range(4):
            expected = 1.0 if i == j else 0.0
            assert abs(product[i][j] - expected) < 1e-6


def test_lu_variants_reconstruct_original_matrix():
    A = make_linear_system_matrix(8, offset=20)

    for factor in (
        lu_no_pivot(A),
        blocked_lu(A, block=4),
        recursive_lu(A, leaf=4),
    ):
        L, U = factor
        reconstructed = _matmul_numeric(L, U)
        _assert_matrix_close(reconstructed, A)


def test_partial_pivot_lu_reconstructs_permuted_matrix():
    A = make_pivot_matrix(8, offset=30)
    perm, L, U = lu_partial_pivot(A)
    permuted = [A[index] for index in perm]
    reconstructed = _matmul_numeric(L, U)

    _assert_matrix_close(reconstructed, permuted)


def test_cholesky_variants_reconstruct_spd_matrix():
    A = make_spd_matrix(8, offset=40)

    for factor in (
        cholesky(A),
        blocked_cholesky(A, block=4),
        recursive_cholesky(A, leaf=4),
    ):
        reconstructed = _matmul_numeric(factor, _transpose_numeric(factor))
        _assert_matrix_close(reconstructed, A, tol=1e-5)


def test_qr_variants_preserve_gram_matrix():
    A = make_matrix(16, 6, offset=50)
    gram = _matmul_numeric(_transpose_numeric(A), A)

    for R in (
        householder_qr(A),
        blocked_qr(A, block=3),
        tsqr(A, leaf_rows=8),
    ):
        gram_r = _matmul_numeric(_transpose_numeric(R), R)
        _assert_matrix_close(gram_r, gram, tol=1e-5)


def test_blocked_lu_is_not_trace_equivalent_to_unblocked_lu():
    args = (make_linear_system_matrix(24, offset=300),)
    live_unblocked = measure_function(lu_no_pivot, args, strategy="aggressive")["cost_discrete"]
    live_blocked = measure_function(lambda A: blocked_lu(A, block=4), args, strategy="aggressive")["cost_discrete"]

    assert live_unblocked != live_blocked
    assert measure_manual_2d("lu-no-pivot-24")["cost_discrete"] != measure_manual_2d("blocked-lu-24")["cost_discrete"]


def test_blocked_qr_is_not_trace_equivalent_to_unblocked_qr():
    args = (make_matrix(48, 12, offset=500),)
    live_unblocked = measure_function(householder_qr, args, strategy="aggressive")["cost_discrete"]
    live_blocked = measure_function(lambda A: blocked_qr(A, block=4), args, strategy="aggressive")["cost_discrete"]

    assert live_unblocked != live_blocked
    assert measure_manual_2d("householder-qr-48x12")["cost_discrete"] != measure_manual_2d("blocked-qr-48x12")["cost_discrete"]


def test_regular_conv_shape_on_small_input():
    image = make_volume(4, 4, 2)
    kernel = make_filter(3, 3, 2, 3, offset=9000)
    out = regular_conv2d(image, kernel)

    assert len(out) == 4
    assert len(out[0]) == 4
    assert len(out[0][0]) == 3


def test_measure_function_orders_simple_matvec():
    args = (make_matrix(8), make_vector(8))
    classic = measure_function(matvec, args, strategy="unmanaged")
    manual = measure_function(matvec, args, strategy="tombstone")
    live = measure_function(matvec, args, strategy="aggressive")

    assert classic["cost_discrete"] >= manual["cost_discrete"] >= live["cost_discrete"]
    assert classic["n_reads"] == manual["n_reads"] == live["n_reads"]


def test_argument_stack_promotes_inputs_on_first_read():
    def reuse_first_entry(values):
        return values[0] + values[0]

    measurement = measure_function(reuse_first_entry, (make_vector(2),), strategy="aggressive")
    assert measurement["trace"][:2] == [2, 1]

    spatial = measure_space_dmd(reuse_first_entry, (make_vector(2),))
    assert spatial["trace"][:2] == [2, 1]


def test_return_value_readback_is_priced():
    measurement = measure_function(lambda x: [x + 1, x + 2], (1,), strategy="aggressive")
    spatial = measure_space_dmd(lambda x: [x + 1, x + 2], (1,))

    assert measurement["n_reads"] == 4
    assert spatial["n_reads"] == 4


def test_manual_tracer_uses_separate_data_region_and_reads_outputs():
    tracer = ManualTracer()
    scratch = tracer.alloc_vector(2)
    inputs = tracer.alloc_input_vector(2)
    out = tracer.alloc_output_vector(2)
    tracer.read(scratch.addr(0))
    tracer.read(inputs.addr(0))
    result = tracer.result()

    assert result["trace"][:2] == [1, 3]
    assert result["trace"][-2:] == [5, 6]


def test_space_dmd_matches_small_gist_style_sweep():
    model = SpaceDMD()
    arr = [model.allocate(is_input=True) for _ in range(4)]
    for _ in range(2):
        for key in arr:
            model.read(key)

    measurement = model.compute_costs()
    assert measurement["cost_discrete"] == 11
    assert measurement["n_reads"] == 8


def test_space_dmd_measurement_runs_on_matvec():
    measurement = measure_space_dmd(matvec, (make_matrix(8), make_vector(8)))

    assert measurement["cost_discrete"] > 0
    assert measurement["n_reads"] > 0


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
    keys = {spec.key for spec in build_algorithm_specs()}

    assert len(rows) == len(keys)
    assert "fused-strassen-16" in keys
    assert any(row["key"] == "fused-strassen-16" for row in rows)
    assert results["overall_max_cell_seconds"] < 10.0
    assert results["metric_columns"] == [SPACE, LIVE, TARGET, CLASSIC]
    assert all(SPACE in row and TARGET in row and CLASSIC in row and LIVE in row for row in rows)
    assert [row["key"] for row in rows[:7]] == [
        "naive-matmul-16",
        "tiled-matmul-16",
        "rmm-16",
        "rmm-lex-16",
        "rmm-gray-16",
        "strassen-16",
        "fused-strassen-16",
    ]
    assert [entry["metric"] for entry in results["ranking"]] == [SPACE, LIVE, CLASSIC]


def test_manual_2d_has_coverage_for_every_algorithm():
    for spec in build_algorithm_specs():
        measurement = measure_manual_2d(spec.key)
        assert measurement["cost_discrete"] > 0
        assert measurement["n_reads"] > 0
