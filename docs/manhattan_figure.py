import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

# ==================================================
# CORE LOGIC
# ==================================================
def isqrt_ceil(x):
    if x <= 0: return 0
    return math.isqrt(x - 1) + 1

def upper_half_spiral(n):
    for i in range(1, n + 1):
        k = isqrt_ceil(i)
        start_i = (k - 1)**2 + 1
        idx = i - start_i
        if k % 2 == 1:
            x = (k - 1) - idx
        else:
            x = -(k - 1) + idx
        y = k - abs(x)
        yield x, y

N_PTS = 400
T_VALS = np.arange(1, N_PTS + 1)
PTS = np.array(list(upper_half_spiral(N_PTS)))


def render_frame(d, n_short=40, table_size=13):
    fig, ax1 = plt.subplots(figsize=(10, 8))

    short_pts = PTS[:n_short]

    # Draw directed arrows showing the path order
    u = np.diff(short_pts[:, 0])
    v = np.diff(short_pts[:, 1])
    ax1.quiver(short_pts[:n_short-1, 0], short_pts[:n_short-1, 1], u, v,
               angles='xy', scale_units='xy', scale=1,
               color='gray', alpha=0.6, width=0.005, headwidth=4, headlength=6, zorder=1)

    # Plot the nodes
    ax1.scatter(short_pts[:, 0], short_pts[:, 1], c=T_VALS[:n_short],
                cmap='plasma', s=40, zorder=2)

    # Label the first 15 points
    for i in range(15):
        is_current = (i + 1 == d)
        weight = 'bold' if is_current else 'normal'
        alpha = 1.0 if is_current else 0.4
        x_offset = 5 if (i + 1) in (1, 3) else 0
        ax1.annotate(f"{i+1}", (short_pts[i, 0], short_pts[i, 1]),
                     textcoords="offset points", xytext=(x_offset, 10),
                     ha='center', va='bottom', fontsize=9, fontweight=weight, color='black', alpha=alpha, zorder=4,
                     bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7 if is_current else 0.3))

    # Highlight the processor at origin
    ax1.plot(0, 0, marker='o', color='red', ms=12, mec='black', zorder=5)

    # Draw path from origin to target
    target_x, target_y = PTS[d - 1]
    ax1.plot([0, 0], [0, target_y], color='red', lw=3, zorder=3)
    ax1.plot([0, target_x], [target_y, target_y], color='red', lw=3, zorder=3)
    ax1.plot(target_x, target_y, marker='o', color='red', ms=12, mec='black', zorder=4)

    ax1.set(aspect='equal', title="2D LRU Stack")
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(1))
    ax1.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax1.grid(True, linestyle=':', alpha=0.5)

    bottom_lim = min(0, short_pts[:, 1].min()) - 1
    ax1.set_ylim(bottom=bottom_lim)

    divider = make_axes_locatable(ax1)
    ax2 = divider.append_axes("right", size="50%", pad=0.6)
    ax2.axis('off')

    table_data = []
    for t in range(1, table_size + 1):
        x, y = PTS[t - 1]
        raw_m = int(abs(x) + abs(y))
        table_data.append([t, raw_m])

    table = ax2.table(
        cellText=table_data,
        colLabels=['d', 'Wire length'],
        cellLoc='center',
        bbox=[0, 0, 1, 1]
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
# GENERATE PNG
# ==================================================
fig = render_frame(12)
fig.savefig('manhattan_figure.png', format='png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("Saved manhattan_figure.png")
