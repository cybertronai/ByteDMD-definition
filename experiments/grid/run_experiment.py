#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
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


SPACE = "SpaceDMD"
TARGET = "Manual-2D"
CLASSIC = "ByteDMD-classic"
LIVE = "ByteDMD-live"

METRIC_COLUMNS = [SPACE, LIVE, TARGET, CLASSIC]
TRACED_COLUMNS = [SPACE, LIVE, TARGET, CLASSIC]


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
    for metric in [SPACE, CLASSIC, LIVE]:
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
    ranking.sort(key=lambda row: (-float(row["spearman_rho"]), float(row["scaled_mape"]), str(row["metric"])))
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
        _markdown_table(algorithm_rows, ["Algorithm", "Workload", "Implementation"]),
        "",
        "## Measures",
        "",
        f"- `{SPACE}`: density-ranked spatial liveness, following the April 17, 2026 gist heuristic for ahead-of-time static pinning.",
        f"- `{LIVE}`: aggressive live-only compaction.",
        f"- `{TARGET}`: hand-scheduled fixed-address implementations under the 2D `ceil(sqrt(addr))` cost model.",
        f"- `{CLASSIC}`: graveyard model with no reclamation.",
        "",
        "The `Manual-2D` column uses explicit fixed-address kernels rather than the tombstone allocator. Traversal-only variants can collapse when they read the same fixed addresses exactly once; scratch-heavy kernels separate much more strongly.",
        "",
        "SpaceDMD globally ranks variables by access density (`access_count / lifespan`) and then charges each read by that variable's rank among the currently live variables.",
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

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))

    report_path = out_dir / "README.md"
    report_path.write_text(render_report(results))

    print(f"Saved {results_path}")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
