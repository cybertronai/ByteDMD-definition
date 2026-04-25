#!/usr/bin/env -S /Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Regenerate the per-algorithm self-contained reproducers under
`experiments/grid/scripts/` from the live source tree.

Each generated script inlines:
  * `bytedmd_ir.py` (L2 IR, tracer, every byte-dmd cost metric,
    including `bytedmd_opt`, `static_opt_lb`, and the recently-added
    `opt_reuse_distances` / `static_opt_floor_curve`),
  * `spacedmd.py` (`space_dmd`),
  * `copy_space_dmd` (from `run_grid.py`),
  * `manual.py` (`Allocator`, every `manual_*` schedule),
  * `algorithms.py` (every traceable algorithm),
  * the plotting helpers from `generate_traces.py` and
    `trace_diagnostics.py`,
  * the size constants and shape helpers from `run_grid.py`.

The driver section per script picks the right `FN` / `ARGS` / `MANUAL`
expressions out of `run_grid.ALGOS` (parsed via `ast.unparse`).

Re-run anytime the metric set changes:

    ./generate_scripts.py        # regenerates scripts/<slug>.py for every ALGOS row
"""
from __future__ import annotations

import ast
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import run_grid as rg  # noqa: E402


# ---------------------------------------------------------------------------
# Source-extraction helpers
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _node_start_line(node: ast.AST) -> int:
    """First line of `node` *including* any decorators attached to it."""
    decos = getattr(node, "decorator_list", None) or []
    if decos:
        return min(d.lineno for d in decos)
    return node.lineno


def file_body(path: pathlib.Path) -> str:
    """Return the file's content with the leading docstring + import
    lines stripped; we re-emit a unified import block at the top of the
    generated script so duplicates don't matter, but it's still cleaner
    to remove them here."""
    src = path.read_text()
    tree = ast.parse(src)
    body_start_line = None
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            continue  # module docstring
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        body_start_line = _node_start_line(node)
        break
    if body_start_line is None:
        return ""
    return "".join(src.splitlines(keepends=True)[body_start_line - 1:])


def extract_named(src: str, name: str) -> str:
    """Return the source text of a top-level def/class with the given name."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node,
                      (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.name == name:
            seg = ast.get_source_segment(src, node)
            if seg is not None:
                return seg
    raise KeyError(name)


def extract_size_constants(rg_src: str) -> str:
    """Block of `N_MM = 16` / `N_ATT, D_ATT, BK = 64, 2, 8` ... up to ALGOS."""
    m = re.search(r"^N_MM\s*=", rg_src, re.M)
    n = re.search(r"^# Each algorithm is", rg_src, re.M)
    if not (m and n):
        raise RuntimeError("constants block not found in run_grid.py")
    return rg_src[m.start():n.start()].rstrip() + "\n"


def get_algos_tuple_sources() -> list[tuple[str, str, str]]:
    """Return per-algo (fn_src, args_src, manual_src) parsed out of run_grid."""
    src = (HERE / "run_grid.py").read_text()
    tree = ast.parse(src)
    list_node = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "ALGOS":
                    list_node = node.value
                    break
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "ALGOS":
            list_node = node.value
        if list_node is not None:
            break
    if list_node is None or not isinstance(list_node, ast.List):
        raise RuntimeError("ALGOS literal not found")
    return [
        (
            ast.unparse(tup.elts[1]).replace("alg.", ""),
            ast.unparse(tup.elts[2]).replace("alg.", ""),
            ast.unparse(tup.elts[3]).replace("man.", ""),
        )
        for tup in list_node.elts
    ]


# ---------------------------------------------------------------------------
# Build the inline runtime once
# ---------------------------------------------------------------------------

IR_BODY = file_body(ROOT / "bytedmd_ir.py")
SPACE_DMD_BODY = file_body(HERE / "spacedmd.py")
MANUAL_BODY = file_body(HERE / "manual.py")
ALG_BODY = file_body(HERE / "algorithms.py")

RG_SRC = (HERE / "run_grid.py").read_text()
COPY_SPACE_DMD_SRC = extract_named(RG_SRC, "copy_space_dmd")
SHAPE_HELPERS = "\n\n".join(
    extract_named(RG_SRC, n) for n in ("mat", "rect", "vec", "cube", "tensor4")
)
SIZE_CONSTS = extract_size_constants(RG_SRC)

GT_SRC = (HERE / "generate_traces.py").read_text()
PLOT_TRACE_SRC = extract_named(GT_SRC, "plot_trace")
SLUGIFY_SRC = extract_named(GT_SRC, "slugify")

TD_SRC = (HERE / "trace_diagnostics.py").read_text()
TD_HELPERS = "\n\n".join(
    extract_named(TD_SRC, n) for n in (
        "walk_live_and_reuse",
        "plot_liveset",
        "pick_wss_window",
        "working_set_over_time",
        "plot_wss",
        "plot_reuse_distance",
        "plot_opt_reuse_distance",
        "plot_reuse_distance_combined",
        "_miss_curve",
        "plot_mrc_combined",
        "_per_event_cost_and_ops",
        "_op_max_operand_costs",
        "plot_rolling_intensity",
        "plot_phase_diagram",
        "plot_gravity_well",
        "plot_locality_cdf",
        "plot_static_opt_floor",
    )
)
# trace_diagnostics' helpers do `from bytedmd_ir import L2Op` lazily inside
# the function bodies; in the generated script L2Op is a top-level class
# from the inlined IR, so the lazy `from bytedmd_ir import` lines fail.
# Strip them.
TD_HELPERS = re.sub(
    r"^\s*from bytedmd_ir import.*\n", "", TD_HELPERS, flags=re.M
)


# ---------------------------------------------------------------------------
# Per-algo template
# ---------------------------------------------------------------------------

HEADER = """#!/usr/bin/env -S /Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
\"\"\"Self-contained reproducer for {NAME}.

Generated by experiments/grid/generate_scripts.py — DO NOT EDIT BY HAND.
Re-run the generator (./generate_scripts.py) instead.

This single file inlines every byte-dmd cost metric (bytedmd_opt,
static_opt_lb, space_dmd, copy_space_dmd, bytedmd_live, bytedmd_classic,
manual), every plot helper, and the algorithm + manual schedule it
exercises. Hand it to a collaborator and run it directly:

    uv run --script {SLUG}.py

Outputs the cost summary for {NAME} plus six diagnostic PNGs into
../traces/ (or alongside the script if that directory does not exist):
{SLUG}.png, {SLUG}_liveset.png, {SLUG}_reuse_distance.png,
{SLUG}_mrc.png, {SLUG}_static_opt_floor.png, {SLUG}_wss.png.
\"\"\"
from __future__ import annotations

import bisect
import heapq
import math
import operator
import os as _os
import re as _re
import sys as _sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""

TEMPLATE = HEADER + """
# ===========================================================================
# bytedmd_ir.py — L2 IR, tracer, all cost metrics
# ===========================================================================

{IR_BODY}

# ===========================================================================
# spacedmd.py — space_dmd density-ranked static allocator
# ===========================================================================

{SPACE_DMD_BODY}

# ===========================================================================
# copy_space_dmd (from run_grid.py)
# ===========================================================================

{COPY_SPACE_DMD_SRC}

# ===========================================================================
# manual.py — Allocator + every hand-placed schedule
# ===========================================================================

{MANUAL_BODY}

# ===========================================================================
# algorithms.py — every traceable algorithm
# ===========================================================================

{ALG_BODY}

# ===========================================================================
# plot_trace + slugify (from generate_traces.py)
# ===========================================================================

{SLUGIFY_SRC}


{PLOT_TRACE_SRC}

# ===========================================================================
# Trace-diagnostic plot helpers (from trace_diagnostics.py)
# ===========================================================================

{TD_HELPERS}

# ===========================================================================
# Input-shape helpers + size constants (from run_grid.py)
# ===========================================================================

{SHAPE_HELPERS}


{SIZE_CONSTS}

# ===========================================================================
# Driver — algorithm-specific
# ===========================================================================

NAME   = {NAME!r}
SLUG   = {SLUG!r}
FN     = {FN_EXPR}
ARGS   = {ARGS_EXPR}
MANUAL = {MANUAL_EXPR}


def _traces_dir():
    here = _os.path.dirname(_os.path.abspath(__file__))
    sibling = _os.path.normpath(_os.path.join(here, "..", "traces"))
    if _os.path.isdir(sibling):
        return sibling
    return here


def main() -> None:
    events, input_vars = trace(FN, ARGS)
    iidx = {{v: i + 1 for i, v in enumerate(input_vars)}}

    costs = {{
        "bytedmd_opt":     bytedmd_opt(events, iidx),
        "static_opt_lb":   static_opt_lb(events, iidx),
        "space_dmd":       space_dmd(events, iidx),
        "copy_space_dmd":  copy_space_dmd(events, iidx),
        "bytedmd_live":    bytedmd_live(events, iidx),
        "manual":          MANUAL(),
        "bytedmd_classic": bytedmd_classic(events, iidx),
    }}

    ls_t, ls_s, rd_t, rd_d = walk_live_and_reuse(events, input_vars)
    opt_t, opt_d = opt_reuse_distances(events, iidx)
    peak_live = max(ls_s) if ls_s else 0
    max_reuse = max(rd_d) if rd_d else 0
    max_opt = max(opt_d) if opt_d else 0
    median_reuse = sorted(rd_d)[len(rd_d) // 2] if rd_d else 0

    logged = Allocator(logging=True)
    set_allocator(logged)
    try:
        MANUAL()
    finally:
        set_allocator(None)

    out_dir = _traces_dir()
    plot_trace(logged.log, logged.writes, logged.output_writes,
               logged.peak, logged.arg_peak,
               f"{{NAME}}  —  cost = {{logged.cost:,}}",
               _os.path.join(out_dir, f"{{SLUG}}.png"))
    plot_liveset(ls_t, ls_s,
                 f"{{NAME}} — live working-set size (peak = {{peak_live:,}})",
                 _os.path.join(out_dir, f"{{SLUG}}_liveset.png"))
    plot_reuse_distance_combined(
        rd_t, rd_d, opt_d,
        f"{{NAME}} — reuse distance per load "
        f"(LRU max = {{max_reuse:,}}; OPT max = {{max_opt:,}})",
        _os.path.join(out_dir, f"{{SLUG}}_reuse_distance.png"))
    plot_mrc_combined(
        rd_d, opt_d,
        f"{{NAME}} — miss-ratio curve (LRU vs Bélády OPT)",
        _os.path.join(out_dir, f"{{SLUG}}_mrc.png"))

    sof_t, sof_v = static_opt_floor_curve(events, iidx)
    sof_total = costs["static_opt_lb"]
    plot_static_opt_floor(
        sof_t, sof_v, sof_total, len(events),
        f"{{NAME}} — per-tick TU LP floor "
        f"(static_opt_lb = {{sof_total:,.0f}})",
        _os.path.join(out_dir, f"{{SLUG}}_static_opt_floor.png"))

    window = pick_wss_window(rd_d, len(events))
    wss_t, wss_s = working_set_over_time(events, window)
    wss_max = max(wss_s) if wss_s else 0
    plot_wss(
        wss_t, wss_s, window,
        f"{{NAME}} — WSS over time (τ = {{window:,}}, max = {{wss_max:,}})",
        _os.path.join(out_dir, f"{{SLUG}}_wss.png"))

    print(f"{{NAME}}")
    print(f"  events          {{len(events):>12,}}")
    for k in ("bytedmd_opt", "static_opt_lb", "space_dmd",
              "copy_space_dmd", "bytedmd_live", "manual",
              "bytedmd_classic"):
        v = costs[k]
        if isinstance(v, float):
            print(f"  {{k:<15}} {{v:>12,.0f}}")
        else:
            print(f"  {{k:<15}} {{v:>12,}}")
    print(f"  peak_live       {{peak_live:>12,}}")
    print(f"  max_reuse_lru   {{max_reuse:>12,}}")
    print(f"  max_reuse_opt   {{max_opt:>12,}}")
    print(f"  median_reuse    {{median_reuse:>12,}}")


if __name__ == "__main__":
    main()
"""


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = HERE / "scripts"
    out_dir.mkdir(exist_ok=True)

    runtime_kwargs = dict(
        IR_BODY=IR_BODY.rstrip(),
        SPACE_DMD_BODY=SPACE_DMD_BODY.rstrip(),
        COPY_SPACE_DMD_SRC=COPY_SPACE_DMD_SRC,
        MANUAL_BODY=MANUAL_BODY.rstrip(),
        ALG_BODY=ALG_BODY.rstrip(),
        SLUGIFY_SRC=SLUGIFY_SRC,
        PLOT_TRACE_SRC=PLOT_TRACE_SRC,
        TD_HELPERS=TD_HELPERS,
        SHAPE_HELPERS=SHAPE_HELPERS,
        SIZE_CONSTS=SIZE_CONSTS,
    )

    algo_sources = get_algos_tuple_sources()
    assert len(algo_sources) == len(rg.ALGOS), \
        "ALGOS runtime / source mismatch"

    written = 0
    for (name, _fn, _args, _man), (fn_src, args_src, manual_src) \
            in zip(rg.ALGOS, algo_sources):
        slug = slugify(name)
        # Strip a no-op `lambda: ` wrapper if it just calls a function so
        # MANUAL = function_ref reads cleaner.
        manual_clean = manual_src.strip()
        # Keep as-is; lambdas are fine for callers.
        text = TEMPLATE.format(
            NAME=name,
            SLUG=slug,
            FN_EXPR=fn_src,
            ARGS_EXPR=args_src,
            MANUAL_EXPR=manual_clean,
            **runtime_kwargs,
        )
        path = out_dir / f"{slug}.py"
        path.write_text(text)
        os.chmod(path, 0o755)
        written += 1
        print(f"  wrote {path.name}  ({len(text):,} bytes)")
    print(f"\nTotal: {written} scripts in {out_dir}")


if __name__ == "__main__":
    main()
