import os, glob, sys
import numpy as np
import pandas as pd

# Usage: python linkFeatures.py [root_dir] [out.csv]
#   Reads mpp_testing_multifeq_775-995*.csv from every subdirectory of root_dir, combines the two
#   unidirectional measurements of each tag pair into one link row (per frequency and run), and
#   attaches the tag-to-tag distance from that subdirectory's distances.csv (or points.csv).
#
# Conventions follow SenSys'21 Eqs. 4.5-4.9. A row with Tx=A, Rx=B is A backscattering to B and
# yields the receiver-side estimates of that direction:
#   Theta_B = Unidirectional Phase,  V_B = Unidirectional V,  beta_B = Unidirectional beta (= V_A*alpha)
# and the Tx=B, Rx=A row yields Theta_A, V_A, beta_A (= V_B*alpha). Then
#   theta_AB = ((Theta_A + Theta_B) / 2) mod pi        [rad]   (4.9)
#   theta_theoretical = (2*pi*f*d_AB / c) mod pi       [rad]   (d_AB in meters)
#   alpha_AB = 0.5 * (beta_A / V_B + beta_B / V_A)              (4.8)
#   avgV_A/avgV_B: mean of the raw "Voltages (mV)" trace recorded at that tag as Rx
#                  (Unidirectional V is used only for alpha)
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "link_features.csv")
PATTERN = "mpp_testing_multifeq_775-995*.csv"

# Data tag name -> 1-based tag number used in distances.csv/points.csv (row index = number - 1).
ALIAS = {f"Tag{i}": i for i in range(1, 11)}
ALIAS.update({"Tag15": 11, "Tag16": 12})

FREQ, RUN = "Frequency (MHz)", "Run Exp Num"
C = 299792458.0  # speed of light (m/s)


def load_distances(d):
    """Return an NxN tag distance matrix (index 0 = tag 1) for subdirectory d."""
    dist_path, pts_path = os.path.join(d, "distances.csv"), os.path.join(d, "points.csv")
    if os.path.exists(dist_path):
        return np.loadtxt(dist_path, delimiter=",", ndmin=2)
    if os.path.exists(pts_path):
        pts = pd.read_csv(pts_path)[["x", "y", "z"]].to_numpy()
        return np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    raise FileNotFoundError(f"no distances.csv or points.csv in {d}")


def trace_mean(s):
    return np.fromstring(s.strip("[]"), sep=",").mean()


def process(csv_path):
    d = os.path.dirname(csv_path)
    D = load_distances(d)
    df = pd.read_csv(csv_path)
    df["avgV"] = df["Voltages (mV)"].map(trace_mean)
    df["th"] = np.deg2rad(df["Unidirectional Phase (deg)"])
    df = df.rename(columns={"Unidirectional V": "V", "Unidirectional beta": "beta"})
    keep = ["Tx", "Rx", FREQ, RUN, "th", "beta", "V", "avgV"]

    unknown = (set(df.Tx) | set(df.Rx)) - ALIAS.keys()
    if unknown:
        raise KeyError(f"{csv_path}: tags missing from ALIAS: {sorted(unknown)}")

    # Pair every A->B row with its B->A row; keep each unordered link once (A < B by alias number).
    fwd = df[keep].rename(columns={"Tx": "A", "Rx": "B", "th": "Theta_B", "beta": "beta_B",
                                   "V": "V_B", "avgV": "avgV_B"})
    rev = df[keep].rename(columns={"Tx": "B", "Rx": "A", "th": "Theta_A", "beta": "beta_A",
                                   "V": "V_A", "avgV": "avgV_A"})
    fwd = fwd[fwd.A.map(ALIAS) < fwd.B.map(ALIAS)]
    links = fwd.merge(rev, on=["A", "B", FREQ, RUN], how="inner")

    links["theta"] = np.mod((links.Theta_A + links.Theta_B) / 2, np.pi)
    links["alpha"] = 0.5 * (links.beta_A / links.V_B + links.beta_B / links.V_A)
    ia, ib = links.A.map(ALIAS).to_numpy() - 1, links.B.map(ALIAS).to_numpy() - 1
    links["distance"] = D[ia, ib]
    links["theta_theoretical"] = np.mod(2 * np.pi * links[FREQ] * 1e6 * links.distance / C, np.pi)
    links.insert(0, "experiment", os.path.basename(d))
    links["tagA_num"], links["tagB_num"] = ia + 1, ib + 1

    cols = ["experiment", "A", "B", "tagA_num", "tagB_num", FREQ, RUN, "theta", "theta_theoretical", "alpha",
            "avgV_A", "avgV_B", "V_A", "V_B", "Theta_A", "Theta_B", "beta_A", "beta_B", "distance"]
    return links[cols].rename(columns={"A": "tagA", "B": "tagB"})


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(ROOT, "*", PATTERN)))
    files = [f for f in files if f.endswith(".csv")]
    out = pd.concat([process(f) for f in files], ignore_index=True)
    # Link-wise: per experiment, each link's rows are contiguous, runs in order, frequencies within a run.
    out["experiment"] = out["experiment"].astype(int)
    out = out.sort_values(["experiment", "tagA_num", "tagB_num", RUN, FREQ], ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"{len(files)} files -> {len(out)} link rows saved to {OUT_PATH}")
