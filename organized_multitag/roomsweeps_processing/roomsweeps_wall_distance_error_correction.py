"""Predict a link's distance from the closest parallel wall from its bidirectional phase and attenuation.

Uses every tag pair, in every room sweep experiment that has a CSV, whose two floor labels are exactly
`--length-cm` apart (default 92 cm = 2 tiles; several lengths can be pooled, e.g. 92,138) along a grid direction, i.e. parallel to a pair of walls:

- Direction 1 links (same column) run parallel to the Edge02 / Edge01 walls (beyond columns 0 and 16).
- Direction 0 links (same row) run parallel to the walls beyond row 1 and row 8.

The target is the perpendicular distance of the link's midpoint (in the merged room scan) from the
closer of its two parallel walls. A label pair measured in more than one experiment (the sweeps overlap by
one column) gives one sample per experiment (--dedup: only the first). θ̂ and α̂ are computed as in roomsweeps_processing.ipynb:

    α̂12 = ½ (β̂1 / V̂2 + β̂2 / V̂1)        θ̂12 = ((Θ̂1 + Θ̂2) / 2) mod π

One sample = one link, with θ̂ and α̂ the median over its runs at the frequencies in --freq-min … --freq-max
(default 775–955 MHz, the same as roomsweeps_wall_distance_fit.py). Features: θ̂ unwrapped along frequency (mod-π jumps removed, starting in
[0, π) at the lowest frequency, or at 0 with --zero-start; raw mod-π values with --no-unwrap), α̂ per frequency (unless --no-attn); the link direction and length are never features. The label pairs are split 70 / 30 at random
into train and test, so every experiment's copy of a link lands on the same side. With --loo every experiment is held
out in turn, and the other experiments' copies of its label pairs are dropped from training (--no-loo-dedup: kept). Fits an SVR and an MLP and compares them with predicting the training mean.

Range-dependent error correction (Error_correction.pdf). Each model's estimate r̂ = r + e has an error that is
correlated with the true distance r (it pulls towards the mean of train). The correction is learned from calibration
pairs (r, r̂) on the training set only. r̂ there is an out-of-fold prediction (--calib-folds GroupKFold by label pair),
because in-sample predictions are overfit and would hide the bias. Then it is applied to the test predictions:

    linear     (eq. 6)   e = α + βr + ε fitted by least squares,  r̃ = (r̂ − α) / (1 + β)
    affine     (eq. 14)  MMSE affine estimator,  r̃ = μ_r + (σ_r² + σ_re) / (σ_r² + σ_e² + 2σ_re) · (r̂ − μ_r̂)
    nonlinear  (eq. 21)  E[e | r̂] regressed with a cubic spline (ridge),  r̃ = r̂ − Ê[e | r̂] = Ê[r | r̂]

Every corrected model is reported next to its uncorrected one. The fitted parameters are printed per fold, and
with the random split a calibration plot is written.

    python roomsweeps_wall_distance_error_correction.py                 # 92 cm links, all three corrections
    python roomsweeps_wall_distance_error_correction.py --correction affine     # one correction only
    python roomsweeps_wall_distance_error_correction.py --correction none       # no correction (as roomsweeps_wall_distance.py)
    python roomsweeps_wall_distance_error_correction.py --length-cm 46  # 1-tile links
    python roomsweeps_wall_distance_error_correction.py --length-cm 92,138  # 2- and 3-tile links in one dataset
    python roomsweeps_wall_distance_error_correction.py --no-attn       # phase only
    python roomsweeps_wall_distance_error_correction.py --zero-start    # unwrapped phase shifted to 0 at the lowest frequency
    python roomsweeps_wall_distance_error_correction.py --no-unwrap     # raw θ̂ in [0, π)
    python roomsweeps_wall_distance_error_correction.py --dedup         # each label pair once, from its first experiment
    python roomsweeps_wall_distance_error_correction.py --loo           # leave one experiment out instead of the random split
    python roomsweeps_wall_distance_error_correction.py --loo --no-loo-dedup  # … keeping the held-out label pairs' other copies in training
    python roomsweeps_wall_distance_error_correction.py --max-wall-cm 200   # only links within 200 cm of their closest parallel wall
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVR

HERE = Path(__file__).resolve().parent   # organized_multitag/roomsweeps_processing: code and *_out folders
ROOT = HERE.parent                        # organized_multitag: data, scans and cache shared by every subfolder
DATA_DIR = ROOT / "roomsweeps_data"
SCAN_DIR = ROOT / "roomscans"
MESH_PATH = SCAN_DIR / "merged.ply"
MARKERS_PATH = SCAN_DIR / "merged_markers.csv"
CACHE_DIR = ROOT / ".cache"
OUT_DIR = HERE / "roomsweeps_wall_distance_error_correction_out"
FREQ_RANGE = (775, 955)   # MHz, inclusive; default band of this script and roomsweeps_wall_distance_fit.py

TILE_CM = 46.0
N_ROWS, N_COLS = 8, 17
TAG_GROUPS = {"T1–T5": range(1, 6), "T6–T10": range(6, 11), "T15–T19": range(15, 20)}
WALL_Z = (1.0, 2.0)       # m; wall points are taken above furniture height
DIRECTION_MARKERS = {0: "o", 1: "^"}
# regions, see roomsweeps_convention.md
DIR0_REGIONS = {"Edge02": range(0, 3), "Center0": range(3, 14), "Edge01": range(14, 17)}   # grid columns
DIR1_REGIONS = {"Edge10": range(0, 4), "Center1": range(4, 8)}                            # grid rows
PART_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]   # fixed categorical order (as in the notebook)

markers = pd.read_csv(MARKERS_PATH).set_index("index").sort_index()


def grid_xy(col, row):
    """(x, y) in metres of the marker at grid (col, row)."""
    return markers.loc[col * 8 + row + 1, ["x", "y"]].to_numpy(float)


def grid_region(regions, x):
    """Region of grid coordinate x (column or row, may be a half-integer midpoint); "A/B" if x is on a boundary."""
    names = [next(name for name, idx in regions.items() if i in idx) for i in (int(np.floor(x)), int(np.ceil(x)))]
    return names[0] if names[0] == names[1] else "/".join(names)


def link_part(ca, ra, cb, rb):
    """Part of the floor a link belongs to: (direction-0 region, direction-1 region) of its midpoint."""
    return f"{grid_region(DIR0_REGIONS, (ca + cb) / 2)}, {grid_region(DIR1_REGIONS, (ra + rb) / 2)}"


def label_grid(n):
    """Floor label -> (col, row)."""
    return divmod(int(n) - 1, 8)


# ---------------------------------------------------------------- tag positions (same as the notebook)

def positions_frame(tags, labels, source):
    df = pd.DataFrame({"tag": [int(str(t).strip()) for t in tags], "label": [int(str(l).strip()) for l in labels]})
    df["source"] = source
    return df.set_index("tag")


def load_positions(exp_dir):
    path = exp_dir / "positions.txt"
    if not path.exists():
        return None
    df = pd.read_csv(path, skipinitialspace=True, dtype=str).dropna()
    return positions_frame(df.iloc[:, 0], df.iloc[:, 1], "file")


def infer_positions(exp_num):
    """Leapfrog sweep: exp n occupies grid columns n-1, n, n+1 (rows 1-5); the tag-group order rotates by one each step."""
    groups = [list(g) for g in TAG_GROUPS.values()]
    k = (exp_num - 1) % 3
    groups = groups[k:] + groups[:k]
    tags, labels = [], []
    for i, group in enumerate(groups):
        col = exp_num - 1 + i
        tags += group
        labels += [col * 8 + row for row in range(1, 6)]
    return positions_frame(tags, labels, "inferred")


N_LEAPFROG = N_COLS - 2   # exps 1…15 are the leapfrog sweep (rows 1-5); later exps have their own layout in positions.txt


def all_positions():
    exp_dirs = sorted((d for d in DATA_DIR.iterdir() if d.is_dir()), key=lambda d: int(d.name))
    out = {}
    for d in exp_dirs:
        n = int(d.name)
        pos = load_positions(d)
        if n > N_LEAPFROG:
            assert pos is not None, f"exp {d.name} is outside the leapfrog sweep and needs a positions.txt"
        elif pos is not None:
            guess = infer_positions(n)["label"]
            assert (guess.loc[pos.index] == pos["label"]).all(), f"leapfrog pattern doesn't match exp {d.name}"
        out[d.name] = pos if pos is not None else infer_positions(n)
    return out


# ---------------------------------------------------------------- θ̂, α̂ (same as the notebook)

def tag_num(name):
    return int(str(name).replace("Tag", ""))


def load_csv(exp_dir):
    f = next(Path(exp_dir).glob("*.csv"))
    cols = pd.read_csv(f, nrows=0).columns
    return pd.read_csv(f, usecols=[c for c in cols if "voltages" not in c.lower()])


def bidirectional(df):
    """Join each Rx=a/Tx=b row with its Rx=b/Tx=a counterpart (a < b) and compute THETA, alpha."""
    keep = ["Rx", "Tx", "Frequency (MHz)", "Run Exp Num",
            "Unidirectional Phase (deg)", "Unidirectional V", "Unidirectional beta"]
    d = df[keep]
    fwd = d[d["Rx"].map(tag_num) < d["Tx"].map(tag_num)]
    rev = d[d["Rx"].map(tag_num) > d["Tx"].map(tag_num)]
    m = fwd.merge(rev, left_on=["Rx", "Tx", "Frequency (MHz)", "Run Exp Num"],
                  right_on=["Tx", "Rx", "Frequency (MHz)", "Run Exp Num"],
                  suffixes=("_a", "_b"), validate="one_to_one")
    th_a = np.deg2rad(m["Unidirectional Phase (deg)_a"])
    th_b = np.deg2rad(m["Unidirectional Phase (deg)_b"])
    m["THETA"] = ((th_a + th_b) / 2) % np.pi
    m["alpha"] = 0.5 * (m["Unidirectional beta_a"] / m["Unidirectional V_b"]
                        + m["Unidirectional beta_b"] / m["Unidirectional V_a"])
    m = m.rename(columns={"Rx_a": "T1", "Tx_a": "T2", "Frequency (MHz)": "freq", "Run Exp Num": "run"})
    m["T1"], m["T2"] = m["T1"].map(tag_num), m["T2"].map(tag_num)
    return m[["T1", "T2", "freq", "run", "THETA", "alpha"]]


def load_bidir(exp):
    """Bidirectional table for one experiment, cached (the CSVs are ~500 MB each)."""
    exp_dir = DATA_DIR / exp
    csv = next(exp_dir.glob("*.csv"))
    cache = CACHE_DIR / f"bidir_{exp}_{csv.stem}.pkl"
    if cache.exists():
        try:
            return pd.read_pickle(cache)
        except ModuleNotFoundError:  # written under another numpy major version
            print(f"  stale cache {cache.name}, rebuilding")
    print(f"  reading {csv.relative_to(ROOT)} …")
    m = bidirectional(load_csv(exp_dir))
    CACHE_DIR.mkdir(exist_ok=True)
    m.to_pickle(cache)
    return m


# ---------------------------------------------------------------- walls

def find_walls():
    """The four walls around the grid as {name: (point, unit normal into the room, links parallel to it)}.

    For each side of the grid, wall points (WALL_Z) beyond that edge and within its length are projected on
    the edge's outward normal; the wall is the median offset around the densest 10 cm bin."""
    verts = np.asarray(o3d.io.read_triangle_mesh(str(MESH_PATH)).vertices)
    pts = verts[(verts[:, 2] > WALL_Z[0]) & (verts[:, 2] < WALL_Z[1])][:, :2]
    sides = {   # name: (edge start, edge end, a point inside the grid, direction of the links parallel to it)
        "Edge02 wall (beyond col 0)": (grid_xy(0, 0), grid_xy(0, N_ROWS - 1), grid_xy(1, 0), 1),
        "Edge01 wall (beyond col 16)": (grid_xy(N_COLS - 1, 0), grid_xy(N_COLS - 1, N_ROWS - 1), grid_xy(N_COLS - 2, 0), 1),
        "Row-1 wall (beyond row 1)": (grid_xy(0, 0), grid_xy(N_COLS - 1, 0), grid_xy(0, 1), 0),
        "Row-8 wall (beyond row 8)": (grid_xy(0, N_ROWS - 1), grid_xy(N_COLS - 1, N_ROWS - 1), grid_xy(0, N_ROWS - 2), 0),
    }
    walls = {}
    for name, (a, b, inside, direction) in sides.items():
        u = (b - a) / np.linalg.norm(b - a)
        n = np.array([u[1], -u[0]])
        if (inside - a) @ n > 0:          # n points out of the grid
            n = -n
        q = pts - a
        s, t = q @ n, q @ u
        near = (t > -0.3) & (t < (b - a) @ u + 0.3) & (s > 0.1) & (s < 4.0)
        h, e = np.histogram(s[near], bins=np.arange(0.1, 4.0, 0.1))
        peak = e[np.argmax(h)] + 0.05
        offset = np.median(s[near & (np.abs(s - peak) < 0.15)])
        walls[name] = (a + offset * n, -n, direction)
    return walls


def wall_distance_cm(xy, wall):
    point, normal, _ = wall
    return 100 * (np.asarray(xy) - point) @ normal


# ---------------------------------------------------------------- links and dataset

def parallel_links(positions, exps, n_tiles, walls, dedup=False, max_wall_cm=None):
    """Every tag pair whose labels are n tiles apart (n in n_tiles) along one grid direction, with its closest parallel wall,
    leaving out links more than max_wall_cm from that wall. A label pair measured in several experiments gives one row
    per experiment; with dedup only the first one in `exps`."""
    rows, seen = [], set()
    for exp in exps:
        pos = positions[exp]
        tags = sorted(pos.index)
        for i, ta in enumerate(tags):
            for tb in tags[i + 1:]:
                la, lb = pos.loc[ta, "label"], pos.loc[tb, "label"]
                (ca, ra), (cb, rb) = label_grid(la), label_grid(lb)
                if ca == cb and abs(ra - rb) in n_tiles:
                    direction, n = 1, abs(ra - rb)
                elif ra == rb and abs(ca - cb) in n_tiles:
                    direction, n = 0, abs(ca - cb)
                else:
                    continue
                mid = (grid_xy(ca, ra) + grid_xy(cb, rb)) / 2
                dists = {name: wall_distance_cm(mid, w) for name, w in walls.items() if w[2] == direction}
                wall = min(dists, key=dists.get)
                if max_wall_cm is not None and dists[wall] > max_wall_cm:
                    continue
                key = tuple(sorted((la, lb)))
                if dedup and key in seen:
                    continue
                seen.add(key)
                rows.append({"exp": exp, "T1": ta, "T2": tb, "label_a": la, "label_b": lb,
                             "part": link_part(ca, ra, cb, rb), "n_tiles": n, "direction": direction, "wall": wall, "wall_cm": dists[wall],
                             "scan_length_cm": 100 * np.linalg.norm(grid_xy(ca, ra) - grid_xy(cb, rb))})
    return pd.DataFrame(rows)


def circmean_mod_pi(x, axis=None):
    return (np.angle(np.mean(np.exp(2j * np.asarray(x)), axis=axis)) / 2) % np.pi


def median_phase(theta, unwrap=True):
    """(n_runs, n_freq) mod-π θ̂ -> median over runs. The circular mean over runs is (if unwrap) unwrapped along
    frequency, every run is moved onto that π branch, then the median is taken per frequency.
    Without unwrap the median is returned mod π, i.e. in [0, π)."""
    ref = circmean_mod_pi(theta, axis=0)
    if unwrap:
        ref = np.unwrap(ref, period=np.pi)
    runs = theta + np.pi * np.round((ref - theta) / np.pi)
    med = np.median(runs, axis=0)
    return med if unwrap else med % np.pi


def link_runs(bidir, link, freq_range):
    """(θ̂, α̂) of a link as (run × freq) frames over freq_range, keeping the runs with every frequency in both."""
    m = bidir[link.exp]
    m = m[(m["T1"] == link.T1) & (m["T2"] == link.T2) & (m["freq"] >= freq_range[0]) & (m["freq"] <= freq_range[1])]
    theta = m.pivot(index="run", columns="freq", values="THETA").dropna()
    alpha = m.pivot(index="run", columns="freq", values="alpha").loc[theta.index].dropna()
    return theta.loc[alpha.index], alpha


def build_dataset(links, bidir, freq_range, use_attn=True, zero_start=False, unwrap=True):
    """One row per link (median over its runs): θ̂ unwrapped along frequency if unwrap, else mod π (minus its
    first value if zero_start), α̂ per frequency (if use_attn). No direction or length features."""
    X, y, meta = [], [], []
    for _, link in links.iterrows():
        theta, alpha = link_runs(bidir, link, freq_range)
        if theta.empty:
            continue
        freqs = theta.columns.to_numpy()
        th = median_phase(theta.to_numpy(), unwrap)
        if zero_start:
            th = th - th[0]
        al = np.median(alpha.to_numpy(), axis=0)
        X.append(np.concatenate([th, al] if use_attn else [th]))
        y.append(link.wall_cm)
        meta.append({**link.to_dict(), "n_runs": len(theta)})
    return np.array(X), np.array(y), pd.DataFrame(meta), freqs


# ---------------------------------------------------------------- models

def models(seed):
    scaled = lambda reg: TransformedTargetRegressor(make_pipeline(StandardScaler(), reg), transformer=StandardScaler())
    return {
        "SVR": scaled(SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale")),
        "MLP": scaled(MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, learning_rate_init=1e-3,
                                   max_iter=5000, random_state=seed)),
    }


def pair_groups(meta):
    """Label pair of every sample as a string (tuples would become a 2-D array)."""
    return np.array(["-".join(map(str, sorted(p))) for p in zip(meta["label_a"], meta["label_b"])])


def split_by_label_pair(meta, train_frac, seed):
    """Train/test sample indices with every sample of a label pair (one per experiment) on the same side."""
    groups = pair_groups(meta)
    return next(GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
                .split(np.zeros(len(meta)), groups=groups))


def describe_split(meta, tr, te):
    n_pairs = lambda idx: len({tuple(sorted(p)) for p in zip(meta["label_a"].iloc[idx], meta["label_b"].iloc[idx])})
    return f"train: {len(tr)} samples ({n_pairs(tr)} label pairs)    test: {len(te)} samples ({n_pairs(te)} label pairs)"


def metrics(y, p):
    return {"MAE (cm)": mean_absolute_error(y, p), "RMSE (cm)": np.sqrt(mean_squared_error(y, p)), "R²": r2_score(y, p)}


def fit_predict(X, y, tr, te, seed):
    """{model name: predictions on te} incl. the training mean, and each model's training MAE."""
    preds, train_mae = {"mean of train": np.full(len(te), y[tr].mean())}, {}
    for name, model in models(seed).items():
        model.fit(X[tr], y[tr])
        preds[name] = model.predict(X[te])
        train_mae[name] = mean_absolute_error(y[tr], model.predict(X[tr]))
    return preds, train_mae


# ---------------------------------------------------------------- range-dependent error correction (Error_correction.pdf)

CORRECTIONS = {"linear": "linear (6)", "affine": "MMSE affine (14)", "nonlinear": "nonlinear (21)"}
NL_KNOTS, NL_RIDGE = 5, 1.0   # spline for E[e | r̂]: knots at quantiles of r̂, ridge penalty on the basis weights
MIN_LINEAR_GAIN = 0.1         # the linear correction (eq. 6) is skipped (r̃ = r̂) when |1 + β| is below this


def out_of_fold(X, y, tr, groups, seed, n_folds):
    """{model: out-of-fold predictions on the training samples tr}. Each one comes from a model fitted without that
    sample's label pair (GroupKFold), so the predictions carry the same range-dependent error as test predictions."""
    k = min(n_folds, len(set(groups[tr])))
    assert k >= 2, "need at least two label pairs in training for the calibration folds"
    oof = {name: np.full(len(tr), np.nan) for name in models(seed)}
    for itr, ite in GroupKFold(n_splits=k).split(tr, groups=groups[tr]):
        for name, model in models(seed).items():
            model.fit(X[tr[itr]], y[tr[itr]])
            oof[name][ite] = model.predict(X[tr[ite]])
    return oof


def fit_correction(method, r, r_hat):
    """Fit a correction on calibration pairs (true r, estimate r̂).
    Returns (function r̂ -> r̃, {parameter: value})."""
    e = r_hat - r
    if method == "linear":                                 # eq. 3, 6
        beta, alpha = np.polyfit(r, e, 1)
        gain = 1 + beta
        params = {"α (cm)": alpha, "β": beta, "1+β": gain, "applied": abs(gain) >= MIN_LINEAR_GAIN}
        if not params["applied"]:   # dividing by 1+β would amplify ε more than 1/MIN_LINEAR_GAIN times: leave r̂
            return (lambda x: np.asarray(x, float)), params
        return (lambda x: (np.asarray(x, float) - alpha) / gain), params
    if method == "affine":                                 # eq. 11-15 (moments over the calibration set)
        mu_r, mu_h = r.mean(), r_hat.mean()
        var_r, var_e = r.var(), e.var()
        s_re = np.mean((r - mu_r) * (e - e.mean()))
        k = (var_r + s_re) / (var_r + var_e + 2 * s_re)   # Cov(r, r̂) / Var(r̂)
        return (lambda x: mu_r + k * (np.asarray(x, float) - mu_h)), {"σ_re (cm²)": s_re, "gain": k}
    if method == "nonlinear":                              # eq. 21-23
        n_knots = int(min(NL_KNOTS, max(2, len(np.unique(r_hat)) // 4)))
        reg = make_pipeline(SplineTransformer(n_knots=n_knots, degree=3, knots="quantile", extrapolation="linear"),
                            Ridge(alpha=NL_RIDGE))
        reg.fit(r_hat[:, None], e)
        return (lambda x: np.asarray(x, float) - reg.predict(np.asarray(x, float)[:, None])), {}
    raise ValueError(method)


def make_corrected_fit(meta, corrections, n_folds, log, calib):
    """A fit_fn for evaluate(). It fits the models as fit_predict does and also adds "<model> + <correction>"
    predictions, using corrections fitted on out-of-fold training predictions. log gets one row of fitted
    parameters per fold and model. calib gets the calibration data (r, r̂ and the correction functions) per fold."""
    groups = pair_groups(meta)
    exp_of = meta["exp"].to_numpy()

    def fit(X, y, tr, te, seed):
        raw, train_mae = fit_predict(X, y, tr, te, seed)
        oof = out_of_fold(X, y, tr, groups, seed, n_folds)
        held_out = set(exp_of[te])
        fold = next(iter(held_out)) if len(held_out) == 1 else "split"
        preds = {"mean of train": raw.pop("mean of train")}
        r = y[tr]
        for name, p in raw.items():
            preds[name] = p
            r_hat = oof[name]
            row = {"fold": fold, "model": name, "calib samples": len(tr),
                   "OOF MAE (cm)": mean_absolute_error(r, r_hat), "corr(e, r)": np.corrcoef(r_hat - r, r)[0, 1]}
            fns = {}
            for method in corrections:
                g, params = fit_correction(method, r, r_hat)
                label = f"{name} + {CORRECTIONS[method]}"
                preds[label] = g(p)
                train_mae[label] = mean_absolute_error(r, g(r_hat))   # on the out-of-fold calibration predictions
                row.update({f"{method}: {k}": v for k, v in params.items()})
                row[f"{method}: calib MAE (cm)"] = train_mae[label]
                fns[method] = g
            log.append(row)
            calib.append({"fold": fold, "model": name, "r": r, "r_hat": r_hat, "fns": fns})
        return preds, train_mae

    return fit


def plot_calibration(calib, path, title):
    """For each model: the out-of-fold error against the true distance with the linear fit e = α + βr (eq. 3), and
    the true distance against the estimate with each correction r̃ = g(r̂)."""
    colors = dict(zip(CORRECTIONS, PART_COLORS[1:]))
    fig, axes = plt.subplots(len(calib), 2, figsize=(12, 5 * len(calib)), squeeze=False)
    for (ax_e, ax_r), c in zip(axes, calib):
        r, r_hat = c["r"], c["r_hat"]
        e = r_hat - r
        lo, hi = 0, max(r.max(), r_hat.max()) * 1.05
        grid = np.linspace(lo, hi, 200)

        ax_e.axhline(0, color="#4a4a4a", lw=1)
        ax_e.scatter(r, e, s=18, color=PART_COLORS[0], alpha=0.6, linewidths=0)
        beta, alpha = np.polyfit(r, e, 1)
        ax_e.plot(grid, alpha + beta * grid, color=colors.get("linear", "#4a4a4a"), lw=2,
                  label=f"e = {alpha:.1f} + {beta:.3f} r   (corr {np.corrcoef(e, r)[0, 1]:.2f})")
        ax_e.set_xlabel("true distance r (cm)")
        ax_e.set_ylabel("out-of-fold error e = r̂ − r (cm)")
        ax_e.set_title(f"{c['model']}: range-dependent error", fontsize=11)
        ax_e.legend(frameon=False, fontsize=8)

        ax_r.plot([lo, hi], [lo, hi], color="#9a9a9a", lw=1, ls="--", label="no correction (r̃ = r̂)")
        ax_r.scatter(r_hat, r, s=18, color=PART_COLORS[0], alpha=0.6, linewidths=0)
        for method, g in c["fns"].items():
            ax_r.plot(grid, g(grid), color=colors[method], lw=2, label=CORRECTIONS[method])
        ax_r.set_xlim(lo, hi)
        ax_r.set_ylim(lo, hi)
        ax_r.set_aspect("equal")
        ax_r.set_xlabel("out-of-fold estimate r̂ (cm)")
        ax_r.set_ylabel("true distance r / corrected r̃ (cm)")
        ax_r.set_title(f"{c['model']}: corrections r̃ = g(r̂)", fontsize=11)
        ax_r.legend(frameon=False, fontsize=8, loc="upper left")
        for ax in (ax_e, ax_r):
            ax.grid(color="#e0e0e0", lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(X, y, meta, train_frac, seed, loo=False, fit_fn=fit_predict, loo_dedup=False, keep_mean=False):
    """Random split by label pair (see split_by_label_pair), or with loo leave one experiment out: train on the
    samples of every other experiment, predict the held-out one, repeat for each experiment and pool the predictions.
    With loo_dedup the training samples of a label pair that the held-out experiment also measured are dropped, so no
    test link is seen in training (a link measured again by another experiment is otherwise an easy match).
    fit_fn(X, y, tr, te, seed) -> ({model: predictions on te}, {model: train MAE}) fits the models.
    Prints the results; returns (test sample indices, {model: predictions}) without the training mean (with it, as
    "mean of train", if keep_mean)."""
    drop = set() if keep_mean else {"mean of train"}
    if not loo:
        tr, te = split_by_label_pair(meta, train_frac, seed)
        print(describe_split(meta, tr, te))
        preds, train_mae = fit_fn(X, y, tr, te, seed)
        results = {name: {**metrics(y[te], p), "train MAE (cm)": train_mae.get(name, np.nan)} for name, p in preds.items()}
        print("\ntest set:")
        print(pd.DataFrame(results).T.round(2).to_string())
        return te, {k: v for k, v in preds.items() if k not in drop}

    exp_of = meta["exp"].to_numpy()
    pair_of = np.array(["-".join(map(str, sorted(p))) for p in zip(meta["label_a"], meta["label_b"])])
    pooled, folds = {}, []
    for exp in dict.fromkeys(exp_of):
        te, tr = np.flatnonzero(exp_of == exp), np.flatnonzero(exp_of != exp)
        repeated = np.isin(pair_of[tr], pair_of[te])
        if loo_dedup:
            tr = tr[~repeated]
        preds, train_mae = fit_fn(X, y, tr, te, seed)
        for name, p in preds.items():
            pooled.setdefault(name, np.full(len(y), np.nan))[te] = p
        folds.append({"exp": exp, "test samples": len(te), "train samples": len(tr),
                      "train samples dropped" if loo_dedup else "train samples of a test pair": int(repeated.sum()),
                      "test pairs also in train": int(np.isin(pair_of[te], pair_of[tr]).sum()),
                      **{f"{name} MAE": mean_absolute_error(y[te], p) for name, p in preds.items()},
                      **{f"{name} train MAE": v for name, v in train_mae.items()}})
    print(f"\nleave one experiment out ({len(folds)} folds"
          f"{', training samples of the held-out label pairs dropped' if loo_dedup else ''}):")
    print(pd.DataFrame(folds).set_index("exp").round(1).to_string())
    print("\nall held-out predictions pooled:")
    print(pd.DataFrame({name: metrics(y, p) for name, p in pooled.items()}).T.round(2).to_string())
    return np.arange(len(y)), {k: v for k, v in pooled.items() if k not in drop}


def plot_predictions(y_test, preds, meta_test, path, title, ncols=None):
    ncols = ncols or len(preds)
    nrows = -(-len(preds) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols + 2.5, 5 * nrows), squeeze=False)
    for ax in axes.flat[len(preds):]:
        ax.set_visible(False)
    order = [f"{a}, {b}" for a in ["Edge02", "Edge02/Center0", "Center0", "Center0/Edge01", "Edge01"]
             for b in ["Edge10", "Edge10/Center1", "Center1"]]
    present = set(meta_test["part"])
    part_colors = {part: PART_COLORS[i % len(PART_COLORS)]
                   for i, part in enumerate(p for p in order if p in present)}
    hi = max(y_test.max(), *(p.max() for p in preds.values()))
    pad = 0.05 * hi                              # axes start at 0 (distances are non-negative)
    for ax, (name, p) in zip(axes.flat, preds.items()):
        ax.plot([0, hi + pad], [0, hi + pad], color="#9a9a9a", lw=1, ls="--", zorder=1)
        for part, color in part_colors.items():
            for d, marker in DIRECTION_MARKERS.items():
                sel = (meta_test["part"].to_numpy() == part) & (meta_test["direction"].to_numpy() == d)
                ax.scatter(y_test[sel], p[sel], s=30, color=color, marker=marker, alpha=0.85,
                           linewidths=0, zorder=2)
        mae = mean_absolute_error(y_test, p)
        ax.set_title(f"{name}  (test MAE {mae:.1f} cm)", fontsize=11)
        ax.set_xlabel("true distance from closest parallel wall (cm)")
        ax.set_ylabel("predicted (cm)")
        ax.set_xlim(0, hi + pad)
        ax.set_ylim(0, hi + pad)
        ax.set_aspect("equal")
        ax.grid(color="#e0e0e0", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, markersize=8, label=part)
               for part, c in part_colors.items()]
    handles += [plt.Line2D([], [], marker=m, ls="", color="#555555", markersize=7, label=f"direction {d} link")
                for d, m in DIRECTION_MARKERS.items()]
    axes[0, -1].legend(handles=handles, title="part (dir-0, dir-1 region)", frameon=False, fontsize=8,
                         title_fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


GOAL_CM = (10, 15)        # target accuracy band, drawn on the error plots
BASELINE_COLOR = "#8a8a8a"


def plot_errors(y_test, preds, baseline, meta_test, path, title, bin_cm=25):
    """Where the models are good or bad, against predicting the training mean (baseline), in three panels:
    the cumulative distribution of |error| (with the GOAL_CM band), the signed error (median and interquartile range)
    per bin_cm bin of the true distance, and the MAE per closest wall and link length. Also prints the numbers."""
    colors = dict(zip(preds, PART_COLORS))
    err = {name: p - y_test for name, p in preds.items()}
    base_err = baseline - y_test
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1, 1.15, 1.1]})

    # 1. cumulative |error|
    ax = axes[0]
    ax.axvspan(*GOAL_CM, color="#1baf7a", alpha=0.12, lw=0, zorder=0)
    ax.text(np.mean(GOAL_CM), 1.0, f"goal\n{GOAL_CM[0]}–{GOAL_CM[1]} cm", ha="center", va="top", fontsize=8,
            color="#4a4a4a")
    hi = max(np.abs(base_err).max(), *(np.abs(e).max() for e in err.values()))
    for name, e, color, ls in [("mean of train", base_err, BASELINE_COLOR, "--"),
                               *((n, e, colors[n], "-") for n, e in err.items())]:
        a = np.sort(np.abs(e))
        within = 100 * (a <= GOAL_CM[1]).mean()
        ax.step(np.r_[0, a], np.r_[0, np.arange(1, len(a) + 1) / len(a)], where="post", color=color, ls=ls, lw=2,
                label=f"{name}: {within:.0f}% ≤ {GOAL_CM[1]} cm, median {np.median(a):.0f} cm")
    ax.set_xlim(0, hi * 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("absolute error (cm)")
    ax.set_ylabel("fraction of test links with at most this error")
    ax.set_title("Cumulative absolute error (left / higher is better)", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # 2. signed error by true distance
    ax = axes[1]
    edges = np.arange(0, y_test.max() + bin_cm, bin_cm)
    b = np.digitize(y_test, edges) - 1
    used = [k for k in range(len(edges) - 1) if (b == k).sum() >= 3]
    centers = np.array([(edges[k] + edges[k + 1]) / 2 for k in used])
    ax.axhline(0, color="#4a4a4a", lw=1, zorder=1)
    ax.axhspan(-GOAL_CM[1], GOAL_CM[1], color="#1baf7a", alpha=0.12, lw=0, zorder=0)
    ax.plot(centers, [np.median(base_err[b == k]) for k in used], color=BASELINE_COLOR, ls="--", lw=2,
            label="mean of train (median)", zorder=2)
    width = 0.6 * bin_cm
    for i, (name, e) in enumerate(err.items()):
        x = centers + (i - (len(err) - 1) / 2) * width / max(len(err), 1)
        q = np.array([np.percentile(e[b == k], [25, 50, 75]) for k in used])
        ax.vlines(x, q[:, 0], q[:, 2], color=colors[name], lw=2, zorder=3)
        ax.plot(x, q[:, 1], "o", color=colors[name], ms=7, mec="white", mew=1.5, zorder=4,
                label=f"{name} (median, bar = middle 50 %)")
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{edges[k]:.0f}–{edges[k + 1]:.0f}\n({(b == k).sum()})" for k in used], fontsize=8)
    ax.set_xlabel("true distance from closest parallel wall (cm), (number of test links)")
    ax.set_ylabel("predicted − true (cm)")
    ax.set_title("Signed error by true distance (a model that learned nothing follows the dashed line)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # 3. MAE per closest wall and link length
    ax = axes[2]
    wall = meta_test["wall"].str.split(" (", n=1, regex=False).str[0].to_numpy()
    tiles = meta_test["n_tiles"].to_numpy()
    groups = [(w, n) for w in dict.fromkeys(wall) for n in sorted(set(tiles[wall == w]))]
    rows = []
    for w, n in groups:
        sel = (wall == w) & (tiles == n)
        rows.append({"group": f"{w}, {n * TILE_CM:.0f} cm ({sel.sum()})", "mean of train": np.abs(base_err[sel]).mean(),
                     **{name: np.abs(e[sel]).mean() for name, e in err.items()}})
    tab = pd.DataFrame(rows).set_index("group")
    y_pos = np.arange(len(tab))[::-1]
    h = 0.8 / max(len(err), 1)
    for i, name in enumerate(err):
        ax.barh(y_pos + (len(err) - 1) / 2 * h - i * h, tab[name], height=h * 0.85, color=colors[name], label=name,
                zorder=2)
    ax.scatter(tab["mean of train"], y_pos, marker="|", s=900, color="#4a4a4a", lw=2.5, zorder=3)
    ax.axvspan(*GOAL_CM, color="#1baf7a", alpha=0.12, lw=0, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tab.index, fontsize=8)
    ax.set_xlabel("MAE (cm)")
    ax.set_title("MAE by closest wall and link length (bar left of the tick = beats the mean)", fontsize=10)
    handles = ax.get_legend_handles_labels()[0] + [
        plt.Line2D([], [], marker="|", ls="", color="#4a4a4a", markersize=12, mew=2.5, label="mean of train")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=len(handles))

    for ax in axes:
        ax.grid(color="#e0e0e0", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    beats = {name: 100 * (np.abs(e) < np.abs(base_err)).mean() for name, e in err.items()}
    print(f"\nerror distribution (test set; links within {GOAL_CM[1]} cm, links better than the mean of train):")
    print(pd.DataFrame({name: {f"≤ {GOAL_CM[1]} cm (%)": 100 * (np.abs(e) <= GOAL_CM[1]).mean(),
                               "median |error| (cm)": np.median(np.abs(e)),
                               "90th pct |error| (cm)": np.percentile(np.abs(e), 90),
                               "beats mean of train (%)": beats.get(name, np.nan)}
                        for name, e in {"mean of train": base_err, **err}.items()}).T.round(1).to_string())
    print("\nMAE (cm) by closest wall and link length (number of test links):")
    print(tab.round(1).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length-cm", default="92",
                    help="link length(s) in cm, comma-separated (e.g. 92,138); each a whole number of tiles")
    ap.add_argument("--freq-min", type=float, default=FREQ_RANGE[0], help="MHz, inclusive")
    ap.add_argument("--freq-max", type=float, default=FREQ_RANGE[1], help="MHz, inclusive")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", action=argparse.BooleanOptionalAction, default=True,
                    help="use α̂ as features alongside θ̂ (--no-attn: phase only)")
    ap.add_argument("--zero-start", action="store_true",
                    help="shift each run's unwrapped θ̂ so it is 0 at the lowest frequency")
    ap.add_argument("--unwrap", action=argparse.BooleanOptionalAction, default=True,
                    help="unwrap θ̂ along frequency (--no-unwrap: raw θ̂ in [0, π))")
    ap.add_argument("--max-wall-cm", type=float, default=None,
                    help="ignore links whose midpoint is more than this many cm from the closest parallel wall")
    ap.add_argument("--dedup", action="store_true",
                    help="keep each label pair once, from the first experiment that measured it")
    ap.add_argument("--loo", action="store_true",
                    help="leave one experiment out: train on the other experiments, test on it, for every experiment")
    ap.add_argument("--loo-dedup", action=argparse.BooleanOptionalAction, default=True,
                    help="with --loo, drop the training samples of every label pair the held-out experiment measured "
                         "(--no-loo-dedup: keep them)")
    ap.add_argument("--correction", default=",".join(CORRECTIONS),
                    help=f"range-dependent error corrections, comma-separated from {', '.join(CORRECTIONS)} "
                         "(Error_correction.pdf eq. 6, 14, 21), or 'none'")
    ap.add_argument("--calib-folds", type=int, default=5,
                    help="GroupKFold folds (by label pair) for the out-of-fold training predictions that the "
                         "corrections are fitted on")
    args = ap.parse_args()
    if args.zero_start and not args.unwrap:
        ap.error("--zero-start only works with unwrapped phase (drop --no-unwrap)")
    corrections = [] if args.correction.strip() == "none" else [c.strip() for c in args.correction.split(",")]
    if unknown := [c for c in corrections if c not in CORRECTIONS]:
        ap.error(f"unknown --correction {', '.join(unknown)} (choose from {', '.join(CORRECTIONS)} or none)")

    lengths = [float(x) for x in args.length_cm.split(",")]
    n_tiles = sorted({int(round(l / TILE_CM)) for l in lengths})
    for l in lengths:
        assert round(l / TILE_CM) > 0 and abs(round(l / TILE_CM) * TILE_CM - l) < 1, \
            f"{l:g} cm is not a whole number of {TILE_CM:g} cm tiles"
    lengths_cm = [f"{n * TILE_CM:.0f}" for n in n_tiles]

    positions = all_positions()
    exps = [e for e in positions if any((DATA_DIR / e).glob("*.csv"))]
    print(f"experiments with a CSV: {', '.join(exps)}  "
          f"(positions inferred for: {', '.join(e for e in exps if positions[e]['source'].iloc[0] == 'inferred') or 'none'})")

    walls = find_walls()
    print("walls (distance of the nearest grid edge from the wall):")
    for name, w in walls.items():
        edge = grid_xy(0, 0) if "col 0" in name or "row 1" in name else grid_xy(N_COLS - 1, N_ROWS - 1)
        print(f"  {name:<30} {wall_distance_cm(edge, w):6.1f} cm")

    links = parallel_links(positions, exps, n_tiles, walls, args.dedup, args.max_wall_cm)
    if links.empty:
        raise SystemExit("no links match --length-cm / --max-wall-cm")
    print(f"\n{len(links)} links ({links.groupby(['label_a', 'label_b']).ngroups} label pairs) of {'/'.join(map(str, n_tiles))} tiles ({'/'.join(lengths_cm)} cm; "
          f"{links.scan_length_cm.min():.0f}–{links.scan_length_cm.max():.0f} cm in the scan)")
    print(links.groupby(["n_tiles", "direction", "wall"]).agg(links=("exp", "size"), min_cm=("wall_cm", "min"),
                                                   max_cm=("wall_cm", "max")).round(1).to_string())

    print("\nloading θ̂, α̂ …")
    bidir = {e: load_bidir(e) for e in exps}
    X, y, meta, freqs = build_dataset(links, bidir, (args.freq_min, args.freq_max), args.attn, args.zero_start, args.unwrap)
    print(f"{len(X)} samples (one per link and experiment, median over {meta.n_runs.min()}–{meta.n_runs.max()} runs), {X.shape[1]} features: θ̂{'' if args.unwrap else ' (wrapped)'}{' (start at 0)' if args.zero_start else ''}{' + α̂' if args.attn else ''} at "
          f"{len(freqs)} freqs {freqs.min()}–{freqs.max()} MHz")

    corr_log, calib = [], []
    fit_fn = make_corrected_fit(meta, corrections, args.calib_folds, corr_log, calib) if corrections else fit_predict
    te, preds = evaluate(X, y, meta, args.train_frac, args.seed, args.loo, fit_fn=fit_fn, loo_dedup=args.loo_dedup,
                         keep_mean=True)
    baseline = preds.pop("mean of train")

    if corrections:
        log = pd.DataFrame(corr_log).set_index(["fold", "model"])
        print(f"\nerror correction, fitted on out-of-fold training predictions ({args.calib_folds}-fold by label pair); "
              "corr(e, r) ≠ 0 means a range-dependent error, |1+β| < 1 means the linear correction amplifies noise:")
        print(log.round(3).to_string())
        if args.loo:
            print("\nmean over folds:")
            print(log.groupby(level="model").mean(numeric_only=True).round(3).to_string())

    OUT_DIR.mkdir(exist_ok=True)
    tag = "-".join(lengths_cm) + "cm" + ("" if args.attn else "_phase_only") + ("_zero_start" if args.zero_start else "") + ("" if args.unwrap else "_no_unwrap") + ("_dedup" if args.dedup else "") + ("_loo" + ("-dedup" if args.loo_dedup else "") if args.loo else "") + (f"_max{args.max_wall_cm:g}cm" if args.max_wall_cm is not None else "") + ("_corr-" + "-".join(corrections) if corrections else "")
    out = meta.iloc[te].reset_index(drop=True).assign(**{f"pred_{k}": v for k, v in preds.items()})
    out.to_csv(OUT_DIR / f"test_predictions_{tag}.csv", index=False)
    plot_predictions(y[te], preds, meta.iloc[te],
                     OUT_DIR / f"pred_vs_true_{tag}.png",
                     f"{', '.join(lengths_cm)} cm links, θ̂{'' if args.unwrap else ' (wrapped)'}{' (start at 0)' if args.zero_start else ''}{' + α̂' if args.attn else ' only'}: "
                     "distance from closest parallel wall (test set)", ncols=len(corrections) + 1)
    if corrections and not args.loo:
        plot_calibration(calib, OUT_DIR / f"calibration_{tag}.png",
                         f"{', '.join(lengths_cm)} cm links: error correction fitted on out-of-fold training predictions")
        print(f"wrote {(OUT_DIR / f'calibration_{tag}.png').relative_to(ROOT)}")
    plot_errors(y[te], preds, baseline, meta.iloc[te], OUT_DIR / f"errors_{tag}.png",
                f"{', '.join(lengths_cm)} cm links, θ̂{'' if args.unwrap else ' (wrapped)'}{' + α̂' if args.attn else ' only'}"
                f"{' (leave one experiment out' + (', held-out pairs dropped from training)' if args.loo_dedup else ')') if args.loo else ''}"
                ": error distribution (test set)")
    print(f"\nwrote {(OUT_DIR / f'test_predictions_{tag}.csv').relative_to(ROOT)}, "
          f"{(OUT_DIR / f'pred_vs_true_{tag}.png').relative_to(ROOT)}, {(OUT_DIR / f'errors_{tag}.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
