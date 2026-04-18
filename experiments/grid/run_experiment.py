#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib"]
# ///
"""Run the heuristic-grid experiment and write results plus a markdown report."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.grid.algorithms import build_algorithm_specs
from experiments.grid.manual_2d import measure_manual_2d
from experiments.grid.measure import measure_function, measure_space_dmd
from experiments.grid.trace_diagnostics import collect_trace_diagnostics


SPACE = "SpaceDMD"
TARGET = "Manual-2D"
CLASSIC = "ByteDMD-classic"
LIVE = "ByteDMD-live"

METRIC_COLUMNS = [SPACE, LIVE, TARGET, CLASSIC]
TRACED_COLUMNS = [SPACE, LIVE, TARGET, CLASSIC]
ROW_ORDER = [
    "naive-matmul-16",
    "tiled-matmul-16",
    "rmm-16",
    "rmm-lex-16",
    "rmm-gray-16",
    "strassen-16",
    "fused-strassen-16",
    "naive-attention-32x2",
    "flash-attention-32x2-b8",
    "regular-attention-32x4",
    "flash-attention-32x4",
    "layernorm-unfused-1024",
    "layernorm-fused-1024",
    "matvec-32",
    "vecmat-32",
    "matvec-row-64",
    "matvec-col-64",
    "matrix-powers-naive-32-s4",
    "matrix-powers-ca-32-s4",
    "spmv-csr-banded-64",
    "spmv-csr-random-64",
    "scan-row-64",
    "scan-column-64",
    "transpose-naive-32",
    "transpose-blocked-32",
    "transpose-recursive-32",
    "fft-iterative-1024",
    "fft-recursive-1024",
    "jacobi-naive-32",
    "jacobi-recursive-32",
    "stencil-time-naive-32-t4",
    "stencil-time-diamond-32-t4",
    "conv2d-spatial-16x16-k5",
    "spatial-conv-32x32-k5",
    "regular-conv-16x16-k3-c4",
    "fft-conv-32",
    "conv2d-fft-16x16-k5",
    "mergesort-64",
    "bitonic-sort-64",
    "lcs-dp-32x32",
    "floyd-warshall-naive-32",
    "floyd-warshall-recursive-32",
    "gaussian-elimination-24",
    "gauss-jordan-inverse-16",
    "lu-no-pivot-24",
    "blocked-lu-24",
    "recursive-lu-24",
    "lu-partial-pivot-24",
    "cholesky-24",
    "blocked-cholesky-24",
    "recursive-cholesky-24",
    "cholesky-right-looking-24",
    "householder-qr-48x12",
    "blocked-qr-48x12",
    "tsqr-48x12",
]
ROW_INDEX = {key: index for index, key in enumerate(ROW_ORDER)}


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and order[end][1] == order[index][1]:
            end += 1
        average_rank = (index + end - 1) / 2.0 + 1.0
        for slot in range(index, end):
            ranks[order[slot][0]] = average_rank
        index = end
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denom_x = math.sqrt(sum(value * value for value in dx))
    denom_y = math.sqrt(sum(value * value for value in dy))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / (denom_x * denom_y)


def _fit_scale(values: list[float], target: list[float]) -> float:
    denom = sum(value * value for value in values)
    if denom == 0:
        return 0.0
    return sum(value * goal for value, goal in zip(values, target)) / denom


def compute_ranking(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    target = [float(row[TARGET]) for row in rows]
    ranking = []
    for metric in [SPACE, LIVE, CLASSIC]:
        values = [float(row[metric]) for row in rows]
        scale = _fit_scale(values, target)
        mape = sum(abs(scale * value - goal) / goal for value, goal in zip(values, target)) / len(target)
        rho = _pearson(_average_ranks(values), _average_ranks(target))
        ranking.append(
            {
                "metric": metric,
                "spearman_rho": rho,
                "scale_fit": scale,
                "scaled_mape": mape,
            }
        )
    return ranking


def collect_results() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    specs = build_algorithm_specs()
    overall_max_cell = 0.0

    for spec in specs:
        strategy_data = {}
        total_seconds = 0.0
        max_cell_seconds = 0.0

        started = time.perf_counter()
        space_measurement = measure_space_dmd(spec.func, spec.args_factory())
        wall_seconds = time.perf_counter() - started
        space_measurement["wall_seconds"] = round(wall_seconds, 4)
        total_seconds += wall_seconds
        max_cell_seconds = max(max_cell_seconds, wall_seconds)
        overall_max_cell = max(overall_max_cell, wall_seconds)

        started = time.perf_counter()
        manual_measurement = measure_manual_2d(spec.key)
        wall_seconds = time.perf_counter() - started
        manual_measurement["wall_seconds"] = round(wall_seconds, 4)
        total_seconds += wall_seconds
        max_cell_seconds = max(max_cell_seconds, wall_seconds)
        overall_max_cell = max(overall_max_cell, wall_seconds)

        for strategy in ("unmanaged", "aggressive"):
            started = time.perf_counter()
            measurement = measure_function(spec.func, spec.args_factory(), strategy=strategy)
            wall_seconds = time.perf_counter() - started
            measurement["wall_seconds"] = round(wall_seconds, 4)
            strategy_data[strategy] = measurement
            total_seconds += wall_seconds
            max_cell_seconds = max(max_cell_seconds, wall_seconds)
            overall_max_cell = max(overall_max_cell, wall_seconds)

        row = {
            "key": spec.key,
            "algorithm": spec.label,
            "workload": spec.workload,
            "notes": spec.notes,
            SPACE: int(space_measurement["cost_discrete"]),
            LIVE: int(strategy_data["aggressive"]["cost_discrete"]),
            TARGET: int(manual_measurement["cost_discrete"]),
            CLASSIC: int(strategy_data["unmanaged"]["cost_discrete"]),
            "cell_seconds": {
                SPACE: space_measurement["wall_seconds"],
                LIVE: strategy_data["aggressive"]["wall_seconds"],
                TARGET: manual_measurement["wall_seconds"],
                CLASSIC: strategy_data["unmanaged"]["wall_seconds"],
            },
            "max_cell_seconds": round(max_cell_seconds, 4),
            "total_traced_seconds": round(total_seconds, 4),
        }
        rows.append(row)

    rows.sort(key=lambda row: (ROW_INDEX.get(str(row["key"]), len(ROW_INDEX)), str(row["algorithm"])))
    ranking = compute_ranking(rows)
    return {
        "algorithms": rows,
        "ranking": ranking,
        "metric_columns": METRIC_COLUMNS,
        "traced_columns": TRACED_COLUMNS,
        "overall_max_cell_seconds": round(overall_max_cell, 4),
    }


def _format_number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format_number(row[column]) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider] + body)


def render_report(results: dict[str, object]) -> str:
    algorithms = list(results["algorithms"])
    ranking = list(results["ranking"])
    overall_max_cell = results["overall_max_cell_seconds"]
    diagnostics = list(results.get("diagnostics", []))
    diagnostics_summary = str(results.get("diagnostics_summary_tsv", ""))
    rho_winner = max(ranking, key=lambda row: float(row["spearman_rho"]))
    mape_winner = min(ranking, key=lambda row: float(row["scaled_mape"]))

    algorithm_rows = [
        {
            "Algorithm": row["algorithm"],
            "Workload": row["workload"],
            "Implementation": row["notes"],
        }
        for row in algorithms
    ]
    grid_rows = [
        {"Algorithm": row["algorithm"], **{metric: row[metric] for metric in METRIC_COLUMNS}}
        for row in algorithms
    ]
    ranking_rows = [
        {
            "Heuristic": row["metric"],
            "Spearman rho": f"{float(row['spearman_rho']):.3f}",
            "Scaled MAPE": f"{100.0 * float(row['scaled_mape']):.1f}%",
        }
        for row in ranking
    ]
    diagnostic_rows = [
        {
            "Algorithm": row["algorithm"],
            "Peak live": int(row["peak_live"]),
            "Max reuse": int(row["max_reuse"]),
            "Median reuse": int(row["median_reuse"]),
            "Working-set plot": f"[link]({row['liveset_plot']})",
            "Reuse-distance plot": f"[link]({row['reuse_distance_plot']})",
        }
        for row in diagnostics
    ]
    runtime_rows = [
        {
            "Algorithm": row["algorithm"],
            "Max traced cell (s)": f"{float(row['max_cell_seconds']):.3f}",
            "Total traced time (s)": f"{float(row['total_traced_seconds']):.3f}",
        }
        for row in algorithms
    ]

    lines = [
        "# Heuristic Grid for ByteDMD-Style Metrics",
        "",
        "This experiment compares a concrete no-free-compaction 2D cost against SpaceDMD and the two abstract ByteDMD heuristics on a small suite of workloads.",
        "",
        f"Every traced metric cell finished under {overall_max_cell:.3f} seconds on this run.",
        "",
        "## Algorithms",
        "",
        "Rows are grouped to follow the dev-branch-style ordering: matmul, attention/vector fusion, matvec/traversal/sparse, FFT, stencil, convolution, sorting/DP/APSP, dense solve, LU, Cholesky, and QR.",
        "",
        _markdown_table(algorithm_rows, ["Algorithm", "Workload", "Implementation"]),
        "",
        "## Measures",
        "",
        f"- `{SPACE}`: density-ranked spatial liveness, now with inputs first read from a separate argument stack and only later re-read from the geometric stack.",
        f"- `{LIVE}`: aggressive live-only compaction on the geometric stack, with the same separate argument-stack first-touch rule.",
        f"- `{TARGET}`: hand-scheduled fixed-address implementations with separate scratch and argument/output regions under the 2D `ceil(sqrt(addr))` cost model.",
        f"- `{CLASSIC}`: graveyard model with no reclamation on the geometric stack, again after the first-touch argument-stack read.",
        "",
        "All four columns now include a terminal readback of the full returned value, so the table prices both computation and the final result extraction.",
        "",
        "SpaceDMD globally ranks geometric-stack variables by access density (`access_count / lifespan`) and then charges each read by that variable's rank among the currently live variables; untouched inputs are priced separately on the argument stack until their first promotion.",
        "",
        "## Interpretation Notes",
        "",
        "- The trace models now have an explicit first-touch boundary: inputs are priced on an argument stack on first use, then promoted into the geometric stack for later re-use. Manual kernels mirror this with separate scratch and argument/output regions.",
        "- SpaceDMD is intentionally order-blind once data is in the geometric stack: pure permutations with the same multiset of reads, such as `Matvec` vs `Vecmat` or `Row Scan` vs `Column Scan`, can collapse to identical SpaceDMD costs even when `Manual-2D` separates them strongly.",
        "- Single-touch kernels such as the transpose trio are a deliberate failure mode for SpaceDMD. When every cell is read once, the metric collapses to the read count (`n^2` here) rather than the physical `ceil(sqrt(addr))` placement cost.",
        "- The blocked LU and blocked QR rows are panel-update variants, not cosmetic loop chunking. If they still land close to their unblocked counterparts, that should be read as an empirical result rather than a placeholder implementation.",
        "- `Recursive LU` and `Recursive Cholesky` here are copy-based block decompositions built out of `_slice_copy`, triangular solves, and Schur complements. Their costs therefore include explicit materialization traffic and should not be read as in-place communication-optimal factorizations.",
        "- `Matrix Powers (CA)` and `Stencil (Time-Diamond)` are locality proxies rather than full communication-optimal solvers. They preserve the intended block-local dataflow but should be read as stress tests for the heuristics, not numerically tuned production kernels.",
        "- These numbers are implementation-specific to this branch. Comparing them directly to other branches that use different schedules, such as right-looking versus left-looking factorizations or different Strassen fusions, can change the measured locality substantially even when the math is the same.",
        "- SpaceDMD can mis-rank virtual/intermediate-heavy traces such as `Strassen` versus `Fused Strassen`, because it scores density-ranked liveness rather than concrete placement.",
        f"- The ranking table has a split verdict: `{rho_winner['metric']}` has the best rank correlation while `{mape_winner['metric']}` has the best scaled MAPE. In other words, the heuristic that orders rows best is not the same one that matches magnitudes best.",
        "",
        "Attention uses proxy `max`, `exp`, and reciprocal operators with the same read arity as the real kernels, so the table focuses on data movement rather than numerical fidelity.",
        "",
        "## Results Grid",
        "",
        _markdown_table(grid_rows, ["Algorithm"] + METRIC_COLUMNS),
        "",
        "## Heuristic Ranking Against Manual-2D",
        "",
        _markdown_table(ranking_rows, ["Heuristic", "Spearman rho", "Scaled MAPE"]),
        "",
        "## Trace Diagnostics",
        "",
        "These follow the dev-branch style plots for the current `ByteDMD-live` path: each algorithm gets a reuse-distance-per-load scatter plot and a working-set-size-over-time step plot under [`diagnostics/`](./diagnostics/).",
        "",
        f"A tab-separated summary is also saved as [`{diagnostics_summary}`](./{diagnostics_summary})." if diagnostics_summary else "",
        "",
        _markdown_table(
            diagnostic_rows,
            ["Algorithm", "Peak live", "Max reuse", "Median reuse", "Working-set plot", "Reuse-distance plot"],
        ) if diagnostic_rows else "",
        "",
        "## Runtime",
        "",
        _markdown_table(runtime_rows, ["Algorithm", "Max traced cell (s)", "Total traced time (s)"]),
        "",
        "Run the experiment with:",
        "",
        "```bash",
        "uv run experiments/grid/run_experiment.py",
        "```",
    ]
    return "\n".join(lines)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = collect_results()
    diagnostics = collect_trace_diagnostics(out_dir=out_dir / "diagnostics", render_plots=True)
    diagnostics_by_key = {str(row["key"]): row for row in diagnostics["algorithms"]}
    results["diagnostics"] = [
        diagnostics_by_key[str(row["key"])]
        for row in results["algorithms"]
        if str(row["key"]) in diagnostics_by_key
    ]
    results["diagnostics_summary_tsv"] = diagnostics["summary_tsv"]

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))

    report_path = out_dir / "README.md"
    report_path.write_text(render_report(results))

    print(f"Saved {results_path}")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
