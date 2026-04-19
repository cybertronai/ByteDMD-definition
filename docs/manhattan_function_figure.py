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
def render_frame(d: int = 8,
                 arg_d: int = 6,
                 n_short_scratch: int = 40,
                 n_short_arg: int = 40,
                 table_size: int = 13) -> plt.Figure:
    """Render the two-arena diamond.

    Args:
        d: which scratch depth to highlight (1-indexed).
        arg_d: which arg depth to highlight (1-indexed).
        n_short_scratch / n_short_arg: how many cells to draw per arena.
        table_size: the right-side table lists scratch depths 1..table_size.
    """
    fig, ax1 = plt.subplots(figsize=(10, 10))

    # --- helper to draw one arena as a quiver + scatter + labels ---
    def draw_arena(pts: np.ndarray, n_short: int, highlight_idx: int,
                   arrow_color: str, label_neg: bool):
        short = pts[:n_short]
        u = np.diff(short[:, 0])
        v = np.diff(short[:, 1])
        ax1.quiver(short[:n_short - 1, 0], short[:n_short - 1, 1], u, v,
                   angles='xy', scale_units='xy', scale=1,
                   color=arrow_color, alpha=0.6, width=0.005,
                   headwidth=4, headlength=6, zorder=1)

        dists = np.array([abs(x) + abs(y) for x, y in short])
        max_d = int(dists.max()) or 1
        cmap = plt.cm.plasma
        colors = [cmap(v / max_d) for v in dists]
        ax1.scatter(short[:, 0], short[:, 1], c=colors, s=40, zorder=2)

        # Label first 15 points; put the number on the outside
        for i in range(min(15, n_short)):
            cur = (i + 1 == highlight_idx)
            weight = 'bold' if cur else 'normal'
            alpha = 1.0 if cur else 0.4
            x_off = 5 if (i + 1) in (1, 3) else 0
            # Put the arg-side labels ABOVE each point, scratch-side BELOW.
            y_off = -10 if label_neg else 7
            va = 'top' if label_neg else 'bottom'
            lbl = f"-{i + 1}" if label_neg else f"{i + 1}"
            ax1.annotate(
                lbl, (short[i, 0], short[i, 1]),
                textcoords="offset points", xytext=(x_off, y_off),
                ha='center', va=va, fontsize=9, fontweight=weight,
                color='black', alpha=alpha, zorder=4,
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec='none', alpha=0.7 if cur else 0.3),
            )

    # Arg arena (above) — label numbers prefixed with minus sign.
    draw_arena(ARG_PTS, n_short_arg, arg_d,
               arrow_color='steelblue', label_neg=True)
    # Scratch arena (below) — standard positive numbers.
    draw_arena(SCRATCH_PTS, n_short_scratch, d,
               arrow_color='gray', label_neg=False)

    # --- ALU at origin ---
    ax1.plot(0, 0, marker='o', color='red', ms=14, mec='black', zorder=5)
    ax1.annotate("core", (0, 0), textcoords="offset points",
                 xytext=(14, 0), ha='left', va='center', fontsize=10,
                 fontweight='bold', color='red')

    # --- Wire paths from the core to each highlighted target ---
    # Scratch target (below)
    sx, sy = SCRATCH_PTS[d - 1]
    ax1.plot([0, 0], [0, sy], color='red', lw=3, zorder=3)
    ax1.plot([0, sx], [sy, sy], color='red', lw=3, zorder=3)
    ax1.plot(sx, sy, marker='o', color='red', ms=12, mec='black', zorder=4)

    # Arg target (above)
    ax, ay = ARG_PTS[arg_d - 1]
    ax1.plot([0, 0], [0, ay], color='red', lw=3, zorder=3, linestyle='--')
    ax1.plot([0, ax], [ay, ay], color='red', lw=3, zorder=3, linestyle='--')
    ax1.plot(ax, ay, marker='o', color='red', ms=12, mec='black', zorder=4)

    # --- Axis dividing line (visual hint of the "arg | scratch" split) ---
    ax1.axhline(0, color='black', alpha=0.25, lw=1, zorder=0)

    # --- Arena labels ---
    y_top = ARG_PTS[:n_short_arg, 1].max() + 0.5
    y_bot = SCRATCH_PTS[:n_short_scratch, 1].min() - 0.5
    ax1.text(0, y_top + 0.5, 'argument arena (inputs)',
             ha='center', va='bottom', fontsize=11,
             color='steelblue', fontweight='bold')
    ax1.text(0, y_bot - 0.5, 'scratch arena (intermediates + output)',
             ha='center', va='top', fontsize=11,
             color='dimgray', fontweight='bold')

    ax1.set(aspect='equal',
            title="Manhattan-Diamond: two arenas around one core")
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

    # --- Right-side address table (same format as manhattan_figure.py) ---
    divider = make_axes_locatable(ax1)
    ax2 = divider.append_axes("right", size="50%", pad=0.6)
    ax2.axis('off')

    table_data = []
    for t in range(1, table_size + 1):
        x, y = SCRATCH_PTS[t - 1]
        table_data.append([t, int(abs(x) + abs(y))])

    table = ax2.table(
        cellText=table_data,
        colLabels=['d', 'Wire length'],
        cellLoc='center',
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='medium')
            cell.set_facecolor('#f2f2f2')
        elif row == d:
            cell.set_facecolor('#ffcccc')
            cell.set_text_props(weight='bold')
        else:
            cell.set_text_props(weight='normal')

    return fig


# ==================================================
# GENERATE OUTPUT
# ==================================================
if __name__ == "__main__":
    fig = render_frame(d=8, arg_d=6)
    fig.savefig('manhattan_function_figure.svg', bbox_inches='tight')
    fig.savefig('manhattan_function_figure.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved manhattan_function_figure.svg and .png")
