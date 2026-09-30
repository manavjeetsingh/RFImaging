import argparse
import os
import pickle
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("dataset", nargs="?", default=None, help="Folder name to process (default: all subdirectories)")
args = parser.parse_args()

current_dir = os.path.dirname(os.path.abspath(__file__))

if args.dataset:
    folders = [os.path.join(current_dir, args.dataset)]
else:
    folders = [os.path.join(current_dir, d) for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]

all_dfs = []
for folder in folders:
    required = ["X_processed.npy", "Y_dist.npy", "processedDF.pkl"]
    if not all(os.path.exists(os.path.join(folder, f)) for f in required):
        print(f"Skipping {folder} — missing required files")
        continue

    print(folder)

    FINAL_DATA = np.load(os.path.join(folder, "X_processed.npy"))  # (n_exp, n_freq, 2)
    Y_dist     = np.load(os.path.join(folder, "Y_dist.npy"))        # (n_exp, 4)

    with open(os.path.join(folder, "processedDF.pkl"), "rb") as f:
        processedDF = pickle.load(f)

    # Frequencies in the order they appear in FINAL_DATA
    freqs_mhz = processedDF["Frequency (MHz)"].unique().astype(int)  # e.g. [845, 855, ..., 945]

    rows = {}
    rows["dataset"] = os.path.basename(folder)
    for freq_idx, freq in enumerate(freqs_mhz):
        rows[f"phase_{freq}"] = FINAL_DATA[:, freq_idx, 1]
        rows[f"atten_{freq}"] = FINAL_DATA[:, freq_idx, 0]

    rows["min_dist"] = np.min(Y_dist, axis=1)

    all_dfs.append(pd.DataFrame(rows))
    print(f"  Added {len(FINAL_DATA)} rows from {os.path.basename(folder)}")

out_path = os.path.join(current_dir, "singleMinDist.csv")
if all_dfs:
    pd.concat(all_dfs, ignore_index=True).to_csv(out_path, index=False)
    print(f"\nSaved {sum(len(d) for d in all_dfs)} total rows to {out_path}")
else:
    print("No datasets processed.")
