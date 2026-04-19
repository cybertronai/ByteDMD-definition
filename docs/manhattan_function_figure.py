"""Two-arena Manhattan-Diamond figure.

Extends manhattan_figure.py to show both sides of the model used by
a single function call:

  * Argument arena (above the core, y > 0): inputs packed outward
    from the ALU, labelled with positive depths 1, 2, 3, ...
  * Scratch arena (below the core, y < 0): the LRU/bump-pointer
    region for intermediates and outputs, labelled with positive
    depths 1, 2, 3, ...

Each arena has its own Manhattan-disc origin at logical depth 1, but
together they form a single diamond around the processor. In the
right-side address table, arg addresses are displayed with a leading
minus sign (-1, -2, ...) to distinguish them from scratch addresses,
matching the convention that args sit "above" the core while scratch
sits "below".
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ==================================================
# CORE LOGIC
# ==================================================
def isqrt_ceil(x: int) -> int:
    if x <= 0:
        return 0
    return math.isqrt(x - 1) + 1


def lower_half_spiral(n: int):
    """Scratch spiral: depths 1..n packed below the origin (y<0)."""
    for i in range(1, n + 1):
        k = isqrt_ceil(i)
        start_i = (k - 1) ** 2 + 1
        idx = i - start_i
        if k % 2 == 1:
            x = -(k - 1) + idx
        else:
            x = (k - 1) - idx
        y = -(k - abs(x))           # scratch below (y<0)
        yield x, y


def upper_half_spiral(n: int):
    """Arg spiral: depths 1..n packed above the origin (y>0).

    Mirrors the scratch spiral through y=0 so the two arenas together
    form a diamond around the processor at the origin."""
    for x, y in lower_half_spiral(n):
        yield x, -y                  # reflect across x-axis


N_PTS = 400
SCRATCH_PTS = np.array(list(lower_half_spiral(N_PTS)))
ARG_PTS = np.array(list(upper_half_spiral(N_PTS)))


# ==================================================
# RENDERING
# ==================================================
RING_COLORS = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']  # rings 1..4
MAX_RING = len(RING_COLORS)


def render_frame(d: int = 8,
                 arg_d: int = 6,
                 n_short_scratch: int = 16,
                 n_short_arg: int = 16,
                 table_size: int = 16,
                 show_access: bool = True) -> plt.Figure:
    """Render the two-arena diamond.

    Args:
        d: which scratch depth to highlight (1-indexed).
        arg_d: which arg depth to highlight (1-indexed).
        n_short_scratch / n_short_arg: how many cells to draw per arena.
        table_size: the right-side table lists scratch depths 1..table_size.
    """
    fig, ax1 = plt.subplots(figsize=(10, 10))

    # --- helper to draw one arena as scatter + labels ---
    def draw_arena(pts: np.ndarray, n_short: int, highlight_idx: int,
                   arrow_color: str, label_neg: bool):
        short = pts[:n_short]
        dists = np.array([abs(x) + abs(y) for x, y in short])
        colors = [RING_COLORS[min(int(v), MAX_RING) - 1] for v in dists]
        ax1.scatter(short[:, 0], short[:, 1], c=colors, s=55,
                    zorder=2, edgecolors='black', linewidths=0.4)

        # Label first 16 points; number sits above each cell in both arenas.
        for i in range(min(16, n_short)):
            cur = (i + 1 == highlight_idx)
            weight = 'bold' if cur else 'normal'
            alpha = 1.0 if cur else 0.4
            x_off = 5 if (i + 1) in (1, 3) else 0
            y_off = 7
            va = 'bottom'
            lbl = f"-{i + 1}" if label_neg else f"{i + 1}"
            ax1.annotate(
                lbl, (short[i, 0], short[i, 1]),
                textcoords="offset points", xytext=(x_off, y_off),
                ha='center', va=va, fontsize=9, fontweight=weight,
                color='black', alpha=alpha, zorder=4,
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec='none', alpha=0.7 if cur else 0.3),
            )

    # --- Hypothetical memory-access wires (H-tree infrastructure) ---
    # One central vertical spine + a horizontal branch per row, spanning
    # exactly the populated cells on that row. Drawn in faint gray so
    # the ring colours of the cells and the red access path both pop.
    LW = 2.0
    WIRE_COLOR = '#bdbdbd'

    def _branch_extent(pts, n_short, row_y):
        xs = [x for x, y in pts[:n_short] if y == row_y]
        return (min(xs), max(xs)) if xs else None

    # Single vertical spine spanning the full arena height.
    arg_y_max = int(ARG_PTS[:n_short_arg, 1].max())
    scr_y_min = int(SCRATCH_PTS[:n_short_scratch, 1].min())
    ax1.plot([0, 0], [scr_y_min, arg_y_max], color=WIRE_COLOR,
             lw=LW, zorder=0.3, solid_capstyle='round')

    # Horizontal branches on every populated row of each arena.
    for k in range(1, MAX_RING + 1):
        ext = _branch_extent(ARG_PTS, n_short_arg, k)
        if ext is not None:
            ax1.plot([ext[0], ext[1]], [k, k], color=WIRE_COLOR, lw=LW,
                     zorder=0.3, solid_capstyle='round')
        ext = _branch_extent(SCRATCH_PTS, n_short_scratch, -k)
        if ext is not None:
            ax1.plot([ext[0], ext[1]], [-k, -k], color=WIRE_COLOR, lw=LW,
                     zorder=0.3, solid_capstyle='round')

    # Arg arena (above) — label numbers prefixed with minus sign.
    draw_arena(ARG_PTS, n_short_arg, arg_d,
               arrow_color='steelblue', label_neg=True)
    # Scratch arena (below) — standard positive numbers.
    draw_arena(SCRATCH_PTS, n_short_scratch, d,
               arrow_color='gray', label_neg=False)

    _ = arg_d  # arg target is no longer drawn as a wire path

    # --- ALU at origin ---
    ax1.plot(0, 0, marker='o', color='red', ms=14, mec='black', zorder=5)
    ax1.annotate("core", (0, 0), textcoords="offset points",
                 xytext=(14, 0), ha='left', va='center', fontsize=10,
                 fontweight='bold', color='red')

    # --- Optional wire path from the core to the highlighted scratch target ---
    if show_access:
        sx, sy = SCRATCH_PTS[d - 1]
        ax1.plot([0, 0], [0, sy], color='red', lw=3, zorder=3)
        ax1.plot([0, sx], [sy, sy], color='red', lw=3, zorder=3)
        # Hollow circle at the target so the cell's ring colour shows through.
        ax1.plot(sx, sy, marker='o', markerfacecolor='none',
                 markeredgecolor='red', markeredgewidth=3, ms=16, zorder=4)

    # --- Axis dividing line (visual hint of the "arg | scratch" split) ---
    ax1.axhline(0, color='black', alpha=0.25, lw=1, zorder=0)

    # --- Arena labels ---
    y_top = ARG_PTS[:n_short_arg, 1].max() + 0.5
    y_bot = SCRATCH_PTS[:n_short_scratch, 1].min() - 0.5
    ax1.text(0, y_top + 0.5, 'read only (arguments)',
             ha='center', va='bottom', fontsize=11,
             color='steelblue', fontweight='bold')
    ax1.text(0, y_bot - 0.5, 'scratch arena (intermediates + output)',
             ha='center', va='top', fontsize=11,
             color='dimgray', fontweight='bold')

    ax1.set(aspect='equal')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(1))
    ax1.tick_params(bottom=False, left=False,
                    labelbottom=False, labelleft=False)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Symmetric y-limits so the whole thing is a diamond.
    y_ext = max(abs(y_top), abs(y_bot)) + 1
    ax1.set_ylim(-y_ext, y_ext)
    x_ext = max(
        abs(SCRATCH_PTS[:n_short_scratch, 0]).max(),
        abs(ARG_PTS[:n_short_arg, 0]).max(),
    ) + 1
    ax1.set_xlim(-x_ext, x_ext)

    # --- Right-side address table with ring-colour strip ---
    divider = make_axes_locatable(ax1)
    ax2 = divider.append_axes("right", size="50%", pad=0.6)
    ax2.axis('off')

    table_data = []
    wire_lengths = []
    for t in range(1, table_size + 1):
        x, y = SCRATCH_PTS[t - 1]
        wl = int(abs(x) + abs(y))
        wire_lengths.append(wl)
        # First column is a blank cell that we'll colour below; second
        # and third columns hold d and its wire length.
        table_data.append(['', t, wl])

    table = ax2.table(
        cellText=table_data,
        colLabels=['', 'd', 'Wire length'],
        cellLoc='center',
        bbox=[0, 0, 1, 1],
        colWidths=[0.08, 0.4, 0.52],   # thin colour strip in col 0
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            # Header row — keep the first (strip) cell matching header grey
            # so it doesn't stick out.
            cell.set_text_props(weight='medium')
            cell.set_facecolor('#f2f2f2')
        elif col == 0:
            # Ring-colour strip: wire_lengths[row-1] -> RING_COLORS[...]
            wl = wire_lengths[row - 1]
            cell.set_facecolor(RING_COLORS[min(wl, MAX_RING) - 1])
            cell.set_edgecolor('white')
        elif show_access and row == d:
            cell.set_facecolor('#ffcccc')
            cell.set_text_props(weight='bold')
        else:
            cell.set_text_props(weight='normal')

    return fig


# ==================================================
# GENERATE OUTPUT
# ==================================================
if __name__ == "__main__":
    # With a concrete access example highlighted (d=8 in scratch).
    fig = render_frame(d=8, arg_d=6, show_access=True)
    fig.savefig('manhattan_function_figure.svg', bbox_inches='tight')
    fig.savefig('manhattan_function_figure.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved manhattan_function_figure.svg and .png")

    # Layout-only variant used in docs/manhattan-diamond.md — no red
    # access path, no highlighted table row; just the two-arena
    # ring-coloured diamond plus its H-tree infrastructure.
    fig = render_frame(show_access=False)
    fig.savefig('manhattan_diamond.svg', bbox_inches='tight')
    fig.savefig('manhattan_diamond.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved manhattan_diamond.svg and .png")
