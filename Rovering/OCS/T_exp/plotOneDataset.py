import argparse
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Folder name to plot, e.g. northeast6")
args = parser.parse_args()

current_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(current_dir, args.dataset)

if not os.path.exists(folder):
    raise FileNotFoundError(f"Dataset folder not found: {folder}")

FINAL_DATA_ALL = np.load(os.path.join(folder, "X_processed_all.npy"))
FINAL_DATA     = np.load(os.path.join(folder, "X_processed.npy"))

with open(os.path.join(folder, "processedDF.pkl"), "rb") as f:
    processedDF = pickle.load(f)

# --- Distribution plots ---
for idx in range(FINAL_DATA_ALL.shape[0]):
    data = FINAL_DATA_ALL[idx, 5, 1, :]
    data = data[data != 0]

    plt.figure()
    plt.hist(data, bins=50, edgecolor='k', alpha=0.7)
    plt.axvline(np.mean(data),   color='r', linestyle='--', label=f"mean={np.mean(data):.3f}")
    plt.axvline(np.median(data), color='g', linestyle='-',  label=f"median={np.median(data):.3f}")
    plt.xlabel("Phase")
    plt.ylabel("Count")
    plt.title(f"FINAL_DATA_ALL[{idx}, 5, 1, :]  (n={len(data)})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    if idx>=3:
        break

# --- Phase vs frequency fit ---
phases = np.unwrap(FINAL_DATA[0, :, 1], period=np.pi)
freqs  = np.array(processedDF["Frequency (MHz)"].unique(), dtype=float) * 1e6

p = np.polyfit(freqs, phases, 1)
slope, intercept = p[0], p[1]
print("Slope:", slope, "Dist:", (3e8 * slope) / (2 * np.pi))

plt.figure()
plt.plot(freqs, phases, 'o', label='data')
plt.plot(freqs, np.polyval(p, freqs), '-', label='linear fit')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Phase (rad)')
plt.legend()
plt.tight_layout()
plt.show()
