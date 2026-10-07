"""Python version of localize_lkhd.m: generate dummy 2D sensor/anchor data and
localize it with the MATLAB LikelyHoodLocalization via the MATLAB Engine API."""
from pathlib import Path

import matlab
import matlab.engine
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent

# Settings (2D only: the method draws circles on an image)
noOfSensors = 10
noOfAnchors = 5
areaSize = 4         # sensors/anchors lie in [0, areaSize]^2 (m)
radioRange = 2       # only pairs closer than this are measured (m)
noisyFac = 0.0      # coarse distance noise: d * (1 + noisyFac*randn)
phaseNoise = 0.0    # phase noise std (rad)
freq = 915e6         # main frequency (Hz)
rng = np.random.default_rng(1)  # reproducible

halfLambda = 3e8 / freq / 2  # phase is wrapped mod pi -> ambiguity of lambda/2 (m)

# Generate dummy locations
# 2 x (noOfSensors + noOfAnchors); anchors are the last noOfAnchors columns
sensors = areaSize * rng.random((2, noOfSensors))
anchors = areaSize * rng.random((2, noOfAnchors))
trueLocationMatrix = np.hstack([sensors, anchors])

# Generate measurements
# All matrices are n x n, upper triangular (i < j), distances in metres.
n = noOfSensors + noOfAnchors
trueDist = np.zeros((n, n))  # true distance for every pair
linkDist = np.zeros((n, n))  # true distance, only for measured links
distEst = np.zeros((n, n))   # coarse (noisy) distance estimate
phase = np.zeros((n, n))     # phase at main frequency, wrapped to [0, pi)
kEst = np.zeros((n, n))      # estimated number of lambda/2 cycles
for i in range(n):
    for j in range(i + 1, n):
        d = np.linalg.norm(trueLocationMatrix[:, i] - trueLocationMatrix[:, j])
        trueDist[i, j] = d
        if d < radioRange:
            linkDist[i, j] = d
            distEst[i, j] = d * max(1 + noisyFac * rng.standard_normal(), 0.1)
            phase[i, j] = np.mod(2 * np.pi * d / (2 * halfLambda)
                                 + phaseNoise * rng.standard_normal(), np.pi)
            wrapped = phase[i, j] / np.pi * halfLambda
            kEst[i, j] = max(round((distEst[i, j] - wrapped) / halfLambda), 0)
print(f"{np.count_nonzero(linkDist)} distance measurements")


def mat(x):
    return matlab.double(np.asarray(x, dtype=float).tolist())


# Parameters (positions, lim, radio_range, lambda in cm; numbers as floats
# so MATLAB receives doubles)
pars = {
    "num_tags": float(noOfSensors),
    "num_anchors": float(noOfAnchors),
    "groundDistances0": mat(trueDist),
    "groundDistances": mat(linkDist),
    "dist_ests": mat(distEst),
    "dist_ests_k": mat(distEst),     # unused by the method, but must exist
    "mainfreqPhase": mat(phase),
    "seedK": mat(kEst),
    "scale": 3.0,                    # pixels per cm
    "lambda": halfLambda * 100,      # cm
    "radio_range": radioRange * 100.0,  # cm
    "lim": areaSize * 100.0,         # cm
    "circle_ind": mat(3 * np.ones((n, n))),  # circles drawn per link (3, 5 or 7)
    "all_circles": 0.0,
    "main_runs": 1.0,
    "savefile": False,
}

numFreqs = 1.0  # only used for the output folder name
anchType = 2.0  # 2 = random anchors (only used for the output folder name)

# Localize
eng = matlab.engine.start_matlab()
eng.addpath(str(THIS_DIR), nargout=0)
X_hat_tags, orig_tag_idx, orig_anchor_idx = eng.LikelyHoodLocalization(
    pars, mat(sensors * 100), mat(anchors * 100), numFreqs, anchType,
    nargout=3,
)
eng.quit()

# The function moves confidently-located tags to the end of X_hat_tags;
# put the estimates back in the original sensor order and convert to m.
orig_tag_idx = np.array(orig_tag_idx).ravel().astype(int) - 1
orig_anchor_idx = np.array(orig_anchor_idx).ravel().astype(int) - 1
order = np.concatenate([orig_tag_idx, orig_anchor_idx[noOfAnchors:]])
estS = np.zeros((2, noOfSensors))
estS[:, order] = np.array(X_hat_tags) / 100

# Results
trueS = sensors
locErr = np.linalg.norm(estS - trueS, axis=0)
print(f"Location error: mean {locErr.mean():.4f} m, "
      f"median {np.median(locErr):.4f} m, max {locErr.max():.4f} m")

fig, ax = plt.subplots()
ax.plot(*anchors, "ks", markerfacecolor="k", label="Anchors")
ax.plot(*trueS, "bo", label="True")
ax.plot(*estS, "r*", label="Estimated")
for s in range(noOfSensors):
    ax.plot(*np.column_stack([trueS[:, s], estS[:, s]]), "r-")
ax.set_aspect("equal")
ax.grid(True)
ax.legend()
ax.set_title(f"Likelihood localization, mean error {locErr.mean():.3f} m")
plt.show()
