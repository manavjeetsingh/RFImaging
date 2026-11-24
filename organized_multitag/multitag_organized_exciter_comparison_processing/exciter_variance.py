"""Compare run-to-run standard deviation of the multitag measurement between exciter types (bladeRF vs RF generator).

Data: multitag_organized_exciter_comparison_data/<exciter>/*.csv, one CSV per exciter, 15 tags, every ordered pair,
915 MHz only, 50 runs. Each link (unordered tag pair) gets θ̂ and α̂ per run as in rfgen_processing.ipynb /
roomsweeps_convention.md §4:

    α̂12 = ½ (β̂1 / V̂2 + β̂2 / V̂1)        θ̂12 = ((Θ̂1 + Θ̂2) / 2) mod π

and its standard deviation over the runs of that link (nothing else is aggregated):

- θ̂ SD (deg): θ̂ is only defined mod π, so each run is first moved onto the π branch closest to the link's
  circular mean (no value changes by more than a branch shift), then the sample SD over runs is taken.
- α̂ SD: sample SD over runs.

Tag positions: <exciter>/positions.txt (`tag, position`, position = floor label) if present, otherwise DEFAULT_POSITIONS,
the layout of rf_gen experiment 1 (T1–T5 @ 1–5, T6–T10 @ 9–13, T15–T19 @ 17–21). The comparison was recorded the same
evening right before that sweep, so the setup is assumed to be the same. Tag-to-tag distance =
TILE_CM · √(Δcol² + Δrow²) on the floor grid (roomsweeps_convention.md §1).

Plots (in exciter_variance_out/):
1. std_per_link_{theta,alpha}.png – SD of every link, one subplot per tag-to-tag distance, the two exciters of a
   link joined by a line.
2. std_per_distance.png – the same per-link SDs as a box plot per tag-to-tag distance and exciter (links overlaid as dots).

    python exciter_variance.py
    python exciter_variance.py --runs 0-24    # only some runs
    python exciter_variance.py --data-version improvedv1   # read <name>_improvedv1.csv (robust MPP + p70)

--data-version picks each exciter's CSV: the collected one (default) or the one rewritten by
phase_calc_improvements/reprocess_improvedv1.py; improvedv1 outputs get an _improvedv1 suffix.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent    # organized_multitag/multitag_organized_exciter_comparison_processing
ROOT = HERE.parent                         # organized_multitag
DATA_DIR = ROOT / "multitag_organized_exciter_comparison_data"
CACHE_DIR = ROOT / ".cache"
OUT_DIR = HERE / "exciter_variance_out"

TILE_CM = 46.0
DEFAULT_POSITIONS = {**{t: t for t in range(1, 6)},             # T1–T5  @ 1–5    (column 0)
                     **{t: t + 3 for t in range(6, 11)},        # T6–T10 @ 9–13   (column 1)
                     **{t: t + 2 for t in range(15, 20)}}       # T15–T19 @ 17–21 (column 2)
EXCITERS = {"bladerf": "bladeRF", "rf_gen": "RF generator"}      # folder -> label, fixed order = fixed colour
COLORS = ["#2a78d6", "#eb6834"]                                  # categorical slots 1, 2
METRICS = [("theta_sd_deg", "θ̂ standard deviation over runs (deg)"),
           ("alpha_sd", "α̂ standard deviation over runs")]


# ---------------------------------------------------------------- loading

def tag_num(name):
    return int(str(name).replace("Tag", ""))


def label_grid(n):
    """Floor label -> (col, row)."""
    return divmod(int(n) - 1, 8)


def load_positions(exc_dir):
    f = exc_dir / "positions.txt"
    if not f.exists():
        print(f"  {exc_dir.name}: no positions.txt, using the rf_gen exp-1 layout")
        return dict(DEFAULT_POSITIONS)
    df = pd.read_csv(f, skipinitialspace=True, dtype=str).dropna()
    return {int(str(t).strip()): int(str(l).strip()) for t, l in zip(df.iloc[:, 0], df.iloc[:, 1])}


DATA_VERSIONS = {"original": "", "improvedv1": "_improvedv1"}   # --data-version -> CSV stem suffix


def experiment_csv(exc_dir, version="original"):
    """The exciter's CSV for `version`: the collected one, or <name>_improvedv1.csv from reprocess_improvedv1.py."""
    suffix = DATA_VERSIONS[version]
    tagged = [s for s in DATA_VERSIONS.values() if s]
    csvs = [c for c in Path(exc_dir).glob("*.csv")
            if (c.stem.endswith(suffix) if suffix else not any(c.stem.endswith(s) for s in tagged))]
    if len(csvs) != 1:
        raise FileNotFoundError(f"{exc_dir}: expected one {version} CSV, found {[c.name for c in csvs]}")
    return csvs[0]


def load_csv(exc_dir, version="original"):
    f = experiment_csv(exc_dir, version)
    cols = pd.read_csv(f, nrows=0).columns
    keep = ["Rx", "Tx", "Frequency (MHz)", "Run Exp Num",
            "Unidirectional Phase (deg)", "Unidirectional V", "Unidirectional beta"]
    missing = set(keep) - set(cols)
    assert not missing, f"{f.name}: missing columns {missing}"
    return f, pd.read_csv(f, usecols=keep)


def bidirectional(df):
    """Join each Rx=a/Tx=b row with its Rx=b/Tx=a counterpart (a < b) -> θ̂, α̂ per link, frequency and run."""
    fwd = df[df["Rx"].map(tag_num) < df["Tx"].map(tag_num)]
    rev = df[df["Rx"].map(tag_num) > df["Tx"].map(tag_num)]
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


def load_bidir(exc_dir, version="original"):
    """Bidirectional table of one exciter, cached per CSV (the CSVs are ~375 MB)."""
    csv = experiment_csv(exc_dir, version)
    cache = CACHE_DIR / f"exciter_bidir_{exc_dir.name}_{csv.stem}.pkl"
    if cache.exists():
        try:
            return pd.read_pickle(cache)
        except ModuleNotFoundError:   # written under another numpy major version
            pass
    print(f"  reading {csv.relative_to(ROOT)} …")
    m = bidirectional(load_csv(exc_dir, version)[1])
    CACHE_DIR.mkdir(exist_ok=True)
    m.to_pickle(cache)
    return m


# ---------------------------------------------------------------- per-link SD

def theta_sd_deg(theta):
    """Sample SD (deg) of mod-π angles: each run moved to the π branch nearest the circular mean first."""
    theta = np.asarray(theta, float)
    ref = np.angle(np.mean(np.exp(2j * theta))) / 2
    runs = theta + np.pi * np.round((ref - theta) / np.pi)
    return np.rad2deg(runs).std(ddof=1)


def link_variance(bidir, positions, exciter):
    rows = []
    for (t1, t2, freq), g in bidir.groupby(["T1", "T2", "freq"]):
        if t1 not in positions or t2 not in positions:
            continue
        (c1, r1), (c2, r2) = label_grid(positions[t1]), label_grid(positions[t2])
        rows.append(dict(exciter=exciter, T1=t1, T2=t2, freq=freq, n_runs=len(g),
                         label1=positions[t1], label2=positions[t2],
                         dist_cm=TILE_CM * np.hypot(c1 - c2, r1 - r2),
                         theta_sd_deg=theta_sd_deg(g["THETA"]),
                         alpha_sd=g["alpha"].to_numpy(float).std(ddof=1)))
    return pd.DataFrame(rows)


def parse_runs(s):
    if s is None:
        return None
    out = set()
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


# ---------------------------------------------------------------- plots

def style(ax):
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9992")
    ax.tick_params(colors="#55544f")


def plot_per_link(res, path, col, xlabel, title, ncols=4):
    """One subplot per tag-to-tag distance; inside it one row per link with the two exciters' SDs as dots joined by a
    line (dumbbell), so each link's exciters are compared directly. Shared x axis across subplots."""
    dists = np.sort(res["dist_cm"].unique())
    wide = res.pivot_table(index=["dist_cm", "T1", "T2"], columns="exciter", values=col).reset_index()
    groups = [wide[wide["dist_cm"] == d].sort_values(["T1", "T2"]).reset_index(drop=True) for d in dists]
    nrows = int(np.ceil((len(dists) + 1) / ncols))          # +1 cell for the legend
    row_n = [max(len(g) for g in groups[r * ncols:(r + 1) * ncols]) for r in range(nrows)]
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 0.22 * sum(row_n) + 1.2 * nrows + 1),
                             sharex=True, gridspec_kw=dict(height_ratios=row_n), squeeze=False)
    xmax = 1.05 * np.nanmax(wide[list(EXCITERS)].to_numpy())
    keys = list(EXCITERS)
    for k, ax in enumerate(axes.flat):
        if k >= len(dists):
            ax.axis("off")
            continue
        g, r = groups[k], k // ncols
        y = np.arange(len(g))
        ax.hlines(y, g[keys].min(axis=1), g[keys].max(axis=1), color="#b8b7b0", linewidth=1.2, zorder=2)
        for i, (key, label) in enumerate(EXCITERS.items()):
            ax.scatter(g[key], y, s=26, color=COLORS[i], edgecolor="white", linewidth=0.6, label=label, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"T{a}–T{b}" for a, b in zip(g["T1"], g["T2"])], fontsize=7)
        ax.set_ylim(row_n[r] - 0.5, -0.5)                   # same row pitch in every subplot of a row, top-aligned
        ax.set_xlim(0, xmax)
        ax.set_title(f"{dists[k]:.0f} cm  (n={len(g)})", fontsize=9, loc="left")
        ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#9a9992")
        ax.tick_params(colors="#55544f", labelbottom=True)
        if r == nrows - 1 or k + ncols >= len(dists):
            ax.set_xlabel(xlabel, fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_ax = axes.flat[len(dists)] if len(dists) < axes.size else axes.flat[-1]
    legend_ax.legend(handles, labels, frameon=False, loc="center", title="exciter")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_distance(res, path, title):
    """Box plot of the per-link SDs at each tag-to-tag distance, exciters side by side; links drawn as dots."""
    dists = np.sort(res["dist_cm"].unique())
    pos = {d: i for i, d in enumerate(dists)}
    n_exc, width = len(EXCITERS), 0.36
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(max(9, 0.75 * len(dists) + 2), 8), sharex=True)
    for ax, (col, ylabel) in zip(axes, METRICS):
        for i, (key, label) in enumerate(EXCITERS.items()):
            d = res[res["exciter"] == key]
            off = (i - (n_exc - 1) / 2) * width
            groups = [d.loc[d["dist_cm"] == dist, col].to_numpy() for dist in dists]
            xs = np.arange(len(dists)) + off
            ax.boxplot(groups, positions=xs, widths=width * 0.8, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="none", edgecolor=COLORS[i], linewidth=1.2),
                       medianprops=dict(color=COLORS[i], linewidth=2),
                       whiskerprops=dict(color=COLORS[i]), capprops=dict(color=COLORS[i]), zorder=2)
            x = d["dist_cm"].map(pos).to_numpy() + off + rng.uniform(-width * 0.25, width * 0.25, len(d))
            ax.scatter(x, d[col], s=12, color=COLORS[i], alpha=0.55, linewidth=0, label=label, zorder=3)
        for j in range(1, len(dists), 2):
            ax.axvspan(j - 0.5, j + 0.5, color="#f3f2ee", zorder=0)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel)
        style(ax)
    counts = res[res["exciter"] == next(iter(EXCITERS))].groupby("dist_cm").size()
    axes[-1].set_xticks(range(len(dists)))
    axes[-1].set_xticklabels([f"{d:.0f}\nn={counts.get(d, 0)}" for d in dists], fontsize=8)
    axes[-1].set_xlim(-0.5, len(dists) - 0.5)
    axes[-1].set_xlabel("tag-to-tag distance (cm); n = links per exciter")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.12), ncol=2)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", help="runs to use, e.g. 0-24 or 0,2,4 (default: all)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--data-version", choices=list(DATA_VERSIONS), default="original",
                    help="exciter CSV to read: the original collection, or <name>_improvedv1.csv from "
                         "phase_calc_improvements/reprocess_improvedv1.py (outputs then get an _improvedv1 suffix)")
    args = ap.parse_args()
    runs = parse_runs(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)

    parts = []
    for key in EXCITERS:
        exc_dir = DATA_DIR / key
        bidir = load_bidir(exc_dir, args.data_version)
        if runs is not None:
            bidir = bidir[bidir["run"].isin(runs)]
        parts.append(link_variance(bidir, load_positions(exc_dir), key))
    res = pd.concat(parts, ignore_index=True)

    for freq, r in res.groupby("freq"):
        tag = (f"_{freq}MHz" if res["freq"].nunique() > 1 else "") + DATA_VERSIONS[args.data_version]
        n_runs = int(r["n_runs"].median())
        title = f"Per-link standard deviation over runs per exciter, {freq} MHz, {n_runs} runs" \
                + ("" if args.data_version == "original" else f" [{args.data_version}]")
        r.to_csv(args.out / f"std_per_link{tag}.csv", index=False)
        for col, xlabel in METRICS:
            name = col.split("_")[0]
            plot_per_link(r, args.out / f"std_per_link_{name}{tag}.png", col, xlabel, title)
        plot_per_distance(r, args.out / f"std_per_distance{tag}.png", title)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
