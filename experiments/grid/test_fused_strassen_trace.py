from experiments.grid.fused_strassen_trace import run_fused_strassen, run_rmm


def test_fused_strassen_stays_inside_expected_memory_envelope():
    log, regions, _ = run_fused_strassen(16, tile=4)

    assert max(log) == regions["main_C"][1]
    assert max(log) <= 3 * 16 * 16 + 3 * 4 * 4


def test_fused_strassen_cost_matches_note_directionally():
    _, _, rmm_cost = run_rmm(16, tile=4)
    _, _, fused_cost = run_fused_strassen(16, tile=4)

    assert fused_cost < 150_000
    assert rmm_cost < fused_cost
