import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import math
import io

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

    ax1.scatter(short_pts[:, 0], short_pts[:, 1], c=T_VALS[:n_short],
                cmap='plasma', s=40, zorder=2)

    spine_y_vals = short_pts[short_pts[:, 0] == 0][:, 1]
    if len(spine_y_vals) > 0:
        ax1.plot([0, 0], [spine_y_vals.min(), spine_y_vals.max()],
                 color='gray', alpha=0.4, lw=1.5, zorder=1)

    unique_ys = np.unique(short_pts[:, 1])
    for y_lvl in unique_ys:
        x_at_y = short_pts[short_pts[:, 1] == y_lvl][:, 0]
        if len(x_at_y) > 0:
            ax1.plot([x_at_y.min(), x_at_y.max()], [y_lvl, y_lvl],
                     color='gray', alpha=0.4, lw=1.5, zorder=1)

    for i in range(15):
        is_current = (i + 1 == d)
        weight = 'bold' if is_current else 'normal'
        alpha = 1.0 if is_current else 0.4
        ax1.annotate(f"{i+1}", (short_pts[i, 0], short_pts[i, 1]),
                     textcoords="offset points", xytext=(6, 6),
                     ha='left', va='bottom', fontsize=9, fontweight=weight, color='black', alpha=alpha, zorder=4,
                     bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7 if is_current else 0.3))

    ax1.plot(0, 0, marker='o', color='red', ms=12, mec='black', zorder=5)

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
        colLabels=['Stack Depth', 'Wire length'],
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
# GENERATE ANIMATED GIF
# ==================================================
d_values = list(range(1, 14)) + list(range(12, 1, -1))

frames = []
for d in d_values:
    fig = render_frame(d)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    buf.close()

frames[0].save('illustration.gif', save_all=True, append_images=frames[1:],
               duration=1000, loop=0)
print("Saved illustration.gif")
