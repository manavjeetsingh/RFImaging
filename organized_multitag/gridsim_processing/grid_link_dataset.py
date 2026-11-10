"""
grid_link_dataset.py — per-link wall-distance dataset from a gridTagDatagen.py run.

Turns the simulated unidirectional Θ̂ / V̂ / β̂ of every grid-aligned link into the
features that RFImaging/organized_multitag/roomsweeps_processing/roomsweeps_wall_distance.py and
roomsweeps_wall_distance_fit.py train on, one CSV row per (simulation, exciter, link):

    α̂12 = ½ (β̂1 / V̂2 + β̂2 / V̂1)        θ̂12 = ((Θ̂1 + Θ̂2) / 2) mod π

with row 1 = Rx T1 / Tx T2 and T1 the lower tag index, exactly as in
roomsweeps_wall_distance.bidirectional. The simulation is noise-free, so there is
one "run" and the median over runs is the value itself.

Columns
-------
meta      sim_idx, exc_idx, T1, T2, cal_T1, cal_T2 (VNA tag MAC), x/y of both tags,
          mid_x, mid_y, z, direction (0 = along X, 1 = along Y), n_tiles,
          spacing_cm, length_cm, exc_x, exc_y, exc_z, exc_dist_cm (exciter → link midpoint)
target    wall_cm — perpendicular distance of the link's midpoint from the closer of
          the two walls parallel to it; wall — which one (x_min/x_max/y_min/y_max)
theta_{f} θ̂ in [0, π) per frequency (MHz)                 → --no-unwrap features
theta_uw_{f}  θ̂ unwrapped along frequency (period π), starting in [0, π)  → default features
alpha_{f} α̂ per frequency
delta_{f}, std_err, sine_A, sine_period_mhz
          the phase fit of variance_processing.ipynb over --fit-min…--fit-max MHz:
          residuals of a least-squares line through the unwrapped θ̂, its standard
          error, and the sine fit to the residuals  → roomsweeps_wall_distance_fit.py features

To build the X of roomsweeps_wall_distance.py: theta_uw_* (+ alpha_*) + one-hot(direction)
+ one-hot(n_tiles), filtered to the link lengths of interest (length_cm). Rows of one
sim share a room layout, so split train/test by sim_idx.

Example
-------
python MLScripts/grid_link_dataset.py \\
    --run-dir datagen_realistic/GeneratedData/run_grid500tag_50sims_ExpRoom_z1_...
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # RFImaging
sys.path.insert(0, os.path.join(_REPO_ROOT, "datagen_realistic"))

from gridTagDatagen import link_wall_distance


def sine_fit_bank(x, n_periods=400, max_cond=100):
    """Pseudo-inverses of the variance_processing.ipynb fit_sine design matrices for grid x.

    Returns (periods (Q,), M (Q, n, 3), pinv (Q, 3, n)) over the trial periods whose
    design matrix is well conditioned; the same for every link, so precomputed once."""
    x = np.asarray(x, float)
    periods, Ms = [], []
    for P in np.linspace(2 * np.min(np.diff(x)), 2 * np.ptp(x), n_periods):
        w = 2 * np.pi * x / P
        M = np.column_stack([np.sin(w), np.cos(w), np.ones_like(x)])
        if np.linalg.cond(M) > max_cond:
            continue
        periods.append(P)
        Ms.append(M)
    Ms = np.array(Ms)
    return np.array(periods), Ms, np.linalg.pinv(Ms)


def phase_fit(theta_uw, freqs, bank):
    """Linear fit + sine fit of each row of theta_uw (L, n) over freqs, as in variance_processing.ipynb."""
    X = np.column_stack([freqs, np.ones_like(freqs)])
    coef = np.linalg.lstsq(X, theta_uw.T, rcond=None)[0]
    delta = theta_uw - (X @ coef).T
    std_err = np.sqrt(np.sum(delta ** 2, axis=1) / (len(freqs) - 2))
    periods, M, pinv = bank
    c = np.einsum("qkn,ln->lqk", pinv, delta)                     # (L, Q, 3): sin, cos, const
    sse = np.sum((np.einsum("qnk,lqk->lqn", M, c) - delta[:, None, :]) ** 2, axis=2)
    best = np.argmin(sse, axis=1)
    cb = c[np.arange(len(delta)), best]
    return delta, std_err, np.hypot(cb[:, 0], cb[:, 1]), periods[best]


def sim_rows(sim, config, freq_sel, fit_sel, bank):
    walls = config["WALLS"]
    freqs = np.asarray(sim["freqs_hz"]) / 1e6
    pos, pairs = sim["positions"], sim["pairs"]
    wall_m, wall_name = link_wall_distance(pos, pairs, sim["direction"], walls)
    mid = (pos[pairs[:, 0]] + pos[pairs[:, 1]]) / 2
    macs = np.asarray(config["CAL_MACS"])

    th = np.deg2rad(sim["theta_deg"][..., freq_sel])                   # (E, P, 2, F)
    V, beta = sim["V"][..., freq_sel], sim["beta"][..., freq_sel]
    theta = ((th[:, :, 0] + th[:, :, 1]) / 2) % np.pi                  # (E, P, F)
    alpha = 0.5 * (beta[:, :, 0] / V[:, :, 1] + beta[:, :, 1] / V[:, :, 0])
    fq = freqs[freq_sel]

    frames = []
    for e, exc in enumerate(sim["exciters"]):
        ok = np.isfinite(theta[e]).all(axis=1) & np.isfinite(alpha[e]).all(axis=1)
        t, a = theta[e][ok], alpha[e][ok]
        p = pairs[ok]
        t_fit = np.unwrap(theta[e][ok][:, fit_sel[freq_sel]], period=np.pi, axis=1)
        delta, std_err, A, period = phase_fit(t_fit, freqs[fit_sel], bank)
        meta = pd.DataFrame({
            "sim_idx": sim["sim_idx"], "exc_idx": e, "T1": p[:, 0], "T2": p[:, 1],
            "cal_T1": macs[sim["tag_cal"][p[:, 0]]], "cal_T2": macs[sim["tag_cal"][p[:, 1]]],
            "x1": pos[p[:, 0], 0], "y1": pos[p[:, 0], 1], "x2": pos[p[:, 1], 0], "y2": pos[p[:, 1], 1],
            "mid_x": mid[ok, 0], "mid_y": mid[ok, 1], "z": mid[ok, 2],
            "direction": sim["direction"][ok], "n_tiles": sim["n_tiles"][ok],
            "spacing_cm": 100 * sim["spacing"], "length_cm": 100 * sim["spacing"] * sim["n_tiles"][ok],
            "exc_x": exc[0], "exc_y": exc[1], "exc_z": exc[2],
            "exc_dist_cm": 100 * np.linalg.norm(mid[ok] - exc, axis=1),
            "wall": wall_name[ok], "wall_cm": 100 * wall_m[ok],
            "std_err": std_err, "sine_A": A, "sine_period_mhz": period,
        })
        name = lambda prefix, fs: [f"{prefix}_{f:g}" for f in fs]
        feats = pd.DataFrame(np.hstack([t, np.unwrap(t, period=np.pi, axis=1), a, delta]),
                             columns=name("theta", fq) + name("theta_uw", fq) + name("alpha", fq)
                             + name("delta", freqs[fit_sel]))
        frames.append(pd.concat([meta, feats], axis=1))
    return pd.concat(frames, ignore_index=True), int((~np.isfinite(theta).all(axis=2)).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--freq-min", type=float, default=775, help="MHz, inclusive")
    ap.add_argument("--freq-max", type=float, default=995)
    ap.add_argument("--fit-min", type=float, default=800, help="phase-fit band (MHz), as variance_processing.ipynb")
    ap.add_argument("--fit-max", type=float, default=960)
    ap.add_argument("--out", default=None, help="default: <run-dir>/grid_links_f{min}-{max}.csv")
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "run_config.pkl"), "rb") as f:
        config = pickle.load(f)
    freqs = np.asarray(config["FREQ_RANGE_HZ"]) / 1e6
    freq_sel = (freqs >= args.freq_min) & (freqs <= args.freq_max)
    fit_sel = (freqs >= args.fit_min) & (freqs <= args.fit_max)
    if (fit_sel & ~freq_sel).any():
        ap.error("the fit band must lie inside --freq-min … --freq-max")
    bank = sine_fit_bank(freqs[fit_sel])

    files = sorted(glob.glob(os.path.join(args.run_dir, "sim_*.pkl")))
    if not files:
        raise SystemExit(f"no sim_*.pkl in {args.run_dir}")
    frames, dropped = [], 0
    for path in tqdm(files, desc="sims"):
        with open(path, "rb") as f:
            df, n_bad = sim_rows(pickle.load(f), config, freq_sel, fit_sel, bank)
        frames.append(df)
        dropped += n_bad
    df = pd.concat(frames, ignore_index=True)

    out = args.out or os.path.join(args.run_dir, f"grid_links_f{args.freq_min:g}-{args.freq_max:g}.csv")
    df.to_csv(out, index=False, float_format="%.6g")
    print(f"{len(df)} link rows from {len(files)} sims ({dropped} dropped for a failed frequency), "
          f"{freq_sel.sum()} freqs {freqs[freq_sel].min():g}–{freqs[freq_sel].max():g} MHz → {out}")
    print(df.groupby("direction").wall_cm.describe().round(1).to_string())


if __name__ == "__main__":
    main()
