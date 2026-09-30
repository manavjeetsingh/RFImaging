import sys
import numpy as np
import matplotlib.pyplot as plt

# Usage: python plotPoints.py path/to/points.csv
# Plots every sphere center from 3dDistances.py in 3D, labeled with the same index (1, 2, 3, ...) as the GUI.
path = sys.argv[1] if len(sys.argv) > 1 else "/Users/manavjeet/git/RFImaging/multitag/1/points.csv"
pts = np.loadtxt(path, delimiter=",", skiprows=1).reshape(-1, 3)

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(projection="3d")
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=40, c="tab:red", depthshade=False)
for i, (x, y, z) in enumerate(pts):
    ax.text(x, y, z, f" {i + 1}", fontsize=12)

ax.set_box_aspect(np.ptp(pts, axis=0) + 1e-9)  # equal scale on all axes so geometry isn't distorted
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title(f"{len(pts)} points — {path}")
plt.tight_layout()
plt.show()
