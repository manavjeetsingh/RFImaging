"""Python version of localize.m: generate dummy sensor/anchor data and
localize it with the MATLAB SDPwrapper via the MATLAB Engine API."""
from pathlib import Path

import matlab
import matlab.engine
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent

# Settings
sDim = 2            # 2 or 3
noOfSensors = 8
noOfAnchors = 5
areaSize = 4         # sensors/anchors lie in [0, areaSize]^sDim (m)
radioRange = 2       # only pairs closer than this are measured (m)
noisyFac = 0.00      # distance noise: d * (1 + noisyFac*randn)
rng = np.random.default_rng(1)  # reproducible

# Generate dummy locations
# sDim x (noOfSensors + noOfAnchors); anchors are the last noOfAnchors columns
sensors = areaSize * rng.random((sDim, noOfSensors))
anchors = areaSize * rng.random((sDim, noOfAnchors))
trueLocationMatrix = np.hstack([sensors, anchors])

# Generate noisy distance measurements
# noOfSensors x (noOfSensors + noOfAnchors). Entry (i,j) is the noisy distance
# between sensor i and node j if they are within radio range, else 0.
# Sensor-sensor block is upper triangular (i < j), as SFSDP expects.
n = noOfSensors + noOfAnchors
distanceMeasurements = np.zeros((noOfSensors, n))
for i in range(noOfSensors):
    for j in range(i + 1, n):
        d = np.linalg.norm(trueLocationMatrix[:, i] - trueLocationMatrix[:, j])
        if d < radioRange:
            distanceMeasurements[i, j] = d * max(1 + noisyFac * rng.standard_normal(), 0.1)
print(f"{np.count_nonzero(distanceMeasurements)} distance measurements")

# Parameters (numbers as floats so MATLAB receives doubles)
pars = {
    "sDim": float(sDim),
    "noOfSensors": float(noOfSensors),
    "noOfAnchors": float(noOfAnchors),
    "noisyFac": float(noisyFac),
    "edgeAlgo": "all",          # use every measured edge
    "SDPsolver": "sedumi",
    "localMethodSW": 1.0,       # refine SDP solution with gradient method
    "show_plots": 0.0,
    "verbose": 0.0,
}

# Localize
eng = matlab.engine.start_matlab()
eng.addpath(str(THIS_DIR), nargout=0)
eng.addpath(str(THIS_DIR / "SFSDP" / "SFSDP"), nargout=0)
eng.addpath(str(THIS_DIR / "SFSDP" / "SFSDP" / "subprograms"), nargout=0)

cost, estimatedlocationMatrix, before_degreeList, degreeList = eng.SDPwrapper(
    pars,
    matlab.double(distanceMeasurements.tolist()),
    matlab.double(trueLocationMatrix.tolist()),
    nargout=4,
)
eng.quit()
estimatedlocationMatrix = np.array(estimatedlocationMatrix)

# Results
trueS = trueLocationMatrix[:, :noOfSensors]
estS = estimatedlocationMatrix[:, :noOfSensors]
locErr = np.linalg.norm(estS - trueS, axis=0)
print(f"Distance RMSE (cost): {cost:.4f} m")
print(f"Location error: mean {locErr.mean():.4f} m, "
      f"median {np.median(locErr):.4f} m, max {locErr.max():.4f} m")

fig = plt.figure()
ax = fig.add_subplot(projection="3d" if sDim == 3 else None)
ax.plot(*anchors, "ks", markerfacecolor="k", label="Anchors")
ax.plot(*trueS, "bo", label="True")
ax.plot(*estS, "r*", label="Estimated")
for s in range(noOfSensors):
    ax.plot(*np.column_stack([trueS[:, s], estS[:, s]]), "r-")
if sDim == 2:
    ax.set_aspect("equal")
ax.grid(True)
ax.legend()
ax.set_title(f"SDP localization ({sDim}D), mean error {locErr.mean():.3f} m")
plt.show()
