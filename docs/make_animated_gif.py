import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import io
from make_gif import render_frame

# Render frames for d=1..13 forward, then 12..2 backward
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

frames[0].save('dmd_animated.gif', save_all=True, append_images=frames[1:],
               duration=1000, loop=0)
print("Saved dmd_animated.gif")
