#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib"]
# ///
"""Generate reuse-distance envelope plots and summary tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.matmul_hierarchy.hierarchy import (  # noqa: E402
    abstract_reuse_depths,
    bytedmd_cost,
    compile_concrete_trace,
    concrete_reuse_depths,
    format_accesses,
    memory_curve,
    trace_matmul_program,
)

CLASSIC = "ByteDMD-classic"
LIVE = "ByteDMD-live"
NEVER_REUSE = "never-reuse"
LIFO = "lifo"
EDF = "edf"
BELADY = "belady"


def make_matrix(n: int, offset: int) -> list[list[int]]:
    return [[offset + i * n + j + 1 for j in range(n)] for i in range(n)]


def powers_of_two_up_to(limit: int) -> list[int]:
    sizes: list[int] = []
    current = 1
    while current < limit:
        sizes.append(current)
        current *= 2
    sizes.append(limit)
    return sorted(set(sizes))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    n = 16
    tile_size = 4
    a = make_matrix(n, 0)
    b = make_matrix(n, n * n)

    algorithms = [
        ("tiled", {"tile_size": tile_size}, f"Naive Tiled (tile={tile_size})"),
        ("recursive", {}, "Vanilla Recursive"),
        ("strassen", {}, "Strassen"),
    ]

    figure, axes = plt.subplots(len(algorithms), 1, figsize=(11, 11), sharex=True)
    if len(algorithms) == 1:
        axes = [axes]

    results: dict[str, object] = {
        "matrix_size": n,
        "tile_size": tile_size,
        "algorithms": {},
        "summary_table": [],
    }

    for axis, (algorithm, kwargs, title) in zip(axes, algorithms):
        program = trace_matmul_program(algorithm, a, b, **kwargs)
        classic_depths = abstract_reuse_depths(program, live_only=False)
        live_depths = abstract_reuse_depths(program, live_only=True)
        never_reuse_depths = concrete_reuse_depths(program, policy=NEVER_REUSE)
        belady_trace = compile_concrete_trace(program, policy=BELADY)
        belady_depths = [access.address for access in belady_trace if access.kind == "load"]
        lifo_depths = concrete_reuse_depths(program, policy=LIFO)
        edf_depths = concrete_reuse_depths(program, policy=EDF)
        max_depth = max(classic_depths + live_depths + belady_depths + lifo_depths + edf_depths)
        memory_sizes = powers_of_two_up_to(max_depth)

        curves = {
            CLASSIC: memory_curve(classic_depths, memory_sizes),
            LIVE: memory_curve(live_depths, memory_sizes),
            BELADY: memory_curve(belady_depths, memory_sizes),
            LIFO: memory_curve(lifo_depths, memory_sizes),
            EDF: memory_curve(edf_depths, memory_sizes),
        }
        costs = {
            CLASSIC: bytedmd_cost(classic_depths),
            LIVE: bytedmd_cost(live_depths),
            NEVER_REUSE: bytedmd_cost(never_reuse_depths),
            BELADY: bytedmd_cost(belady_depths),
            LIFO: bytedmd_cost(lifo_depths),
            EDF: bytedmd_cost(edf_depths),
        }

        axis.plot(memory_sizes, curves[CLASSIC], label=CLASSIC, color="#b22222", linewidth=2)
        axis.plot(memory_sizes, curves[LIVE], label=LIVE, color="#228b22", linewidth=2)
        axis.plot(memory_sizes, curves[BELADY], label=BELADY, color="#6a3d9a", linewidth=2)
        axis.plot(memory_sizes, curves[LIFO], label=LIFO, color="#1f77b4", linewidth=2)
        axis.plot(memory_sizes, curves[EDF], label=EDF, color="#ff8c00", linewidth=2)
        axis.set_xscale("log", base=2)
        axis.set_ylabel("loads above cache size")
        axis.set_title(
            f"{title} (N={n})\n"
            f"{CLASSIC}/{LIVE}/{BELADY}/{LIFO}/{EDF} = "
            f"{costs[CLASSIC]:,} / {costs[LIVE]:,} / {costs[BELADY]:,} / {costs[LIFO]:,} / {costs[EDF]:,}"
        )
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)

        results["algorithms"][algorithm] = {
            "title": title,
            "result_checksum": sum(sum(row) for row in program.result),
            "result_corner": program.result[0][0],
            "abstract_access_count": len(program.abstract_accesses),
            "memory_sizes": memory_sizes,
            "curves": curves,
            "bytedmd_costs": costs,
            "abstract_access_preview": format_accesses(program.abstract_accesses, limit=12),
            "concrete_never_reuse_preview": format_accesses(
                compile_concrete_trace(program, policy=NEVER_REUSE), limit=12
            ),
            "concrete_belady_preview": format_accesses(belady_trace, limit=12),
            "concrete_lifo_preview": format_accesses(
                compile_concrete_trace(program, policy=LIFO), limit=12
            ),
            "concrete_edf_preview": format_accesses(
                compile_concrete_trace(program, policy=EDF), limit=12
            ),
        }
        results["summary_table"].append(
            {
                "algorithm": title,
                CLASSIC: costs[CLASSIC],
                LIVE: costs[LIVE],
                NEVER_REUSE: costs[NEVER_REUSE],
                BELADY: costs[BELADY],
                LIFO: costs[LIFO],
                EDF: costs[EDF],
            }
        )

    axes[-1].set_xlabel("cache size (scalar slots)")
    figure.suptitle("Reuse-distance envelopes for three traced matmul hierarchies", fontsize=14, y=0.995)
    figure.tight_layout(rect=[0, 0, 1, 0.985])

    figure_path = out_dir / "reuse_envelope_n16.png"
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    results["figure"] = figure_path.name
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))

    print(f"Saved {figure_path}")
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
