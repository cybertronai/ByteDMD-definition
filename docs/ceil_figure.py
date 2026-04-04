import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import MultipleLocator

def usqrt(x):
    """Ceiling of square root."""
    return math.isqrt(x - 1) + 1

# Generate points only up to 13
t_vals = np.arange(1, 14)

# Calculate exact square root and the upper integer square root
sqrt_vals = np.sqrt(t_vals)
usqrt_vals = np.array([usqrt(t) for t in t_vals])

fig, ax = plt.subplots(figsize=(8, 5))

# Plot both
ax.plot(t_vals, sqrt_vals, label=r'$\sqrt{d}$', color='blue', lw=2)
ax.scatter(t_vals, usqrt_vals, label=r'$\lceil\sqrt{d}\rceil$', color='darkorange', s=60, zorder=3)

ax.set_title("Read cost", fontsize=14)
ax.set_xlabel("d", fontsize=12)
ax.set_ylabel("Cost", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, linestyle=':', alpha=0.7)

# Force x and y axes to show integer ticks for better readability
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))

plt.savefig('ceil_figure.png', bbox_inches='tight')
plt.close()
print("Saved ceil_figure.png")
