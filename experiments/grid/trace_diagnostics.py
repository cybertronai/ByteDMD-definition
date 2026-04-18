#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib"]
# ///
"""Trace diagnostics for the heuristic grid.

Generates two dev-branch-style plots for each algorithm under the current
ByteDMD-live measurement path:

  - `<key>_liveset.png`: geometric-stack working-set size over time
  - `<key>_reuse_distance.png`: reuse distance per load
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from experiments.grid.algorithms import AlgorithmSpec, build_algorithm_specs
from experiments.grid.measure import measure_function_diagnostics


def plot_liveset(times: list[int], sizes: list[int], title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(times, sizes, color="tab:blue", linewidth=0.8, drawstyle="steps-post", rasterized=True)
    ax.fill_between(times, 0, sizes, color="tab:blue", alpha=0.18, linewidth=0, step="post", rasterized=True)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Live values on geometric stack")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if times:
        ax.set_xlim(0, times[-1] + 1)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reuse_distance(times: list[int], distances: list[int], title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.scatter(times, distances, s=0.8, c="tab:purple", alpha=0.35, linewidths=0, rasterized=True)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Reuse distance (ByteDMD-live depth)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if times:
        ax.set_xlim(0, times[-1] + 1)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_summary_tsv(rows: list[dict[str, object]], out_path: Path) -> None:
    header = "\t".join(
        ["key", "algorithm", "peak_live", "max_reuse", "median_reuse", "liveset_plot", "reuse_distance_plot"]
    )
    lines = [header]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["key"]),
                    str(row["algorithm"]),
                    str(row["peak_live"]),
                    str(row["max_reuse"]),
                    str(row["median_reuse"]),
                    str(row["liveset_plot"]),
                    str(row["reuse_distance_plot"]),
                ]
            )
        )
    out_path.write_text("\n".join(lines) + "\n")


def collect_trace_diagnostics(
    specs: Iterable[AlgorithmSpec] | None = None,
    *,
    out_dir: Path | None = None,
    render_plots: bool = True,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    grid_dir = Path(__file__).resolve().parent
    out_root = out_dir if out_dir is not None else (Path(__file__).resolve().parent / "diagnostics")
    out_root.mkdir(parents=True, exist_ok=True)

    for spec in (build_algorithm_specs() if specs is None else list(specs)):
        measurement = measure_function_diagnostics(spec.func, spec.args_factory(), strategy="aggressive")
        liveset_name = f"{spec.key}_liveset.png"
        reuse_name = f"{spec.key}_reuse_distance.png"
        if render_plots:
            plot_liveset(
                list(measurement["live_times"]),
                list(measurement["live_sizes"]),
                f"{spec.label} — working-set size over time (peak = {int(measurement['peak_live']):,})",
                out_root / liveset_name,
            )
            plot_reuse_distance(
                list(measurement["read_times"]),
                list(measurement["read_depths"]),
                f"{spec.label} — reuse distance per load (max = {int(measurement['max_reuse']):,})",
                out_root / reuse_name,
            )

        rows.append(
            {
                "key": spec.key,
                "algorithm": spec.label,
                "peak_live": int(measurement["peak_live"]),
                "max_reuse": int(measurement["max_reuse"]),
                "median_reuse": int(measurement["median_reuse"]),
                "liveset_plot": f"diagnostics/{liveset_name}",
                "reuse_distance_plot": f"diagnostics/{reuse_name}",
            }
        )

    summary_path = out_root / "diagnostics_summary.tsv"
    _write_summary_tsv(rows, summary_path)
    try:
        summary_rel = str(summary_path.relative_to(grid_dir))
    except ValueError:
        summary_rel = str(summary_path)
    return {
        "algorithms": rows,
        "summary_tsv": summary_rel,
    }


def main() -> None:
    diagnostics = collect_trace_diagnostics()
    print(f"Saved diagnostics for {len(diagnostics['algorithms'])} algorithms")
    print(f"Saved summary: {Path(__file__).resolve().parent / diagnostics['summary_tsv']}")


if __name__ == "__main__":
    main()
