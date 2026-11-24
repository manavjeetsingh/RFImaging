"""Recompute link phases from the raw MPP traces with a chosen MPP-extraction and phase-calc algorithm, then plot the
run-to-run θ̂ standard deviation per tag-to-tag distance for each exciter (the θ panel of
multitag_organized_exciter_comparison_processing/exciter_variance_out/std_per_distance.png).

Pipeline, per row (one Rx <- Tx MPP) of multitag_organized_exciter_comparison_data/<exciter>/*.csv:

    mpp = MPP_EXTRACTORS[--mpp](voltages, meta, channels)          {"medians": {channel: mV}, "samples": ...}
    Θ̂   = PHASE_CALCS[--phase](mpp, rx_mac, tx_mac, cfg, freq)     unidirectional phase (deg)

then per link (unordered pair) and run:   θ̂ = COMBINERS[--combine](Θ̂_ab, Θ̂_ba)   (rad, mod π)
and per link: SD over runs (exciter_variance.theta_sd_deg).

Adding an algorithm: write a function with the same signature (in its own module or here) and register it under a
new name in MPP_EXTRACTORS / PHASE_CALCS / COMBINERS. It then becomes a valid --mpp / --phase / --combine choice.

    python phase_pipeline.py                                   # original / original
    python phase_pipeline.py --mpp original --phase original --runs 0-24
    python phase_pipeline.py --mpp robust --phase pct70          # 70th percentile per channel
    python phase_pipeline.py --mpp robust --phase sections       # median of 100 section-combination phases
    python phase_pipeline.py --list                            # show registered algorithms

Outputs go to out/<mpp>__<phase>__<combine>/ : uni_phases.csv (every MPP), std_per_link.csv, std_per_distance_theta.png
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                  # organized_multitag/phase_calc_improvements
ROOT = HERE.parent                                       # organized_multitag
DATA_DIR = ROOT / "multitag_organized_exciter_comparison_data"
CACHE_DIR = ROOT / ".cache"
OUT_ROOT = HERE / "out"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "multitag_organized_exciter_comparison_processing"))

import mpp_extraction as mppx                            # noqa: E402
import mpp_extraction_robust as mppr                     # noqa: E402
import phase_calc_improved as pci                        # noqa: E402
import phase_calc_original as pco                        # noqa: E402
from exciter_variance import (COLORS, EXCITERS, TILE_CM, experiment_csv, label_grid, load_positions,  # noqa: E402
                              style, tag_num, theta_sd_deg)

FREQ_COL = "Frequency (MHz)"
RUN_COL = "Run Exp Num"
REP_COL = "MPP Repetition"
VOLT_COL = "Voltages (mV)"


# ================================================================ algorithm registries
#
# MPP extractor:  f(voltages: np.ndarray, meta: dict, channels: list[int]) -> dict with
#                   "medians": {channel: median mV}                      (required; what phase "original" uses)
#                   "samples": {channel: 1-D mV array}                   (edge-trimmed samples the median is over;
#                                                                         required by phase "pct70" / "sections")
#                   "windows": {channel: (lo, hi)} sample index ranges   (optional; drawn by plot_traces.py)
#                   "spikes":  bool mask of samples the method replaced  (optional; drawn by plot_traces.py)
#                 meta = {"dwell_samples": float|None, "rate_hz": float|None, "connection": str}
# Phase calc:     f(mpp: dict (extractor output), rx_mac: str, tx_mac: str, cfg: dict, freq_hz: float) -> unidirectional phase (deg)
# Combiner:       f(phase_ab_deg: ndarray, phase_ba_deg: ndarray) -> link θ̂ (rad, defined mod π)

def grid_kwargs(meta):
    """Dwell source as in the live pipeline: the recorded dwell if there is one, else the transport table."""
    dwell = meta.get("dwell_samples")
    if dwell is not None and np.isfinite(dwell):
        return {"dwell_samples": dwell}
    return {"transport": meta["connection"]}


def mpp_original(voltages, meta, channels):
    """mpp_segment.segment_capture, dwell as in the live pipeline (recorded dwell, else transport table)."""
    medians, samples, (start, dwell, _) = mppx.segment_capture(voltages, channels, **grid_kwargs(meta))
    return {"medians": medians, "samples": samples, "windows": mppx.channel_windows(start, dwell, channels)}


def mpp_robust(voltages, meta, channels):
    """despike + trimmed-mean grid score (mpp_extraction_robust.segment_capture_robust)."""
    medians, samples, (start, dwell, _), spikes = mppr.segment_capture_robust(voltages, channels, **grid_kwargs(meta))
    return {"medians": medians, "samples": samples, "windows": mppx.channel_windows(start, dwell, channels), "spikes": spikes}


def phase_original(mpp, rx_mac, tx_mac, cfg, freq_hz):
    """phase_cal.cal_theta_et_al (PV polynomial + Tx S11, least squares) on the channel medians."""
    return pco.cal_theta_et_al(mpp["medians"], rxName=rx_mac, txName=tx_mac, cfg=cfg, freq=freq_hz)[0]


def make_phase_pct(q):
    """Phase calc that aggregates each channel by its q-th percentile instead of the median."""
    def phase_pct(mpp, rx_mac, tx_mac, cfg, freq_hz):
        return pci.cal_theta_percentile(mpp["samples"], rx_mac, tx_mac, cfg, freq_hz, q=float(q))
    phase_pct.__doc__ = f"as original, but each channel aggregated by its {q}th percentile instead of the median."
    return phase_pct


def phase_sections(mpp, rx_mac, tx_mac, cfg, freq_hz):
    """10 section medians per channel, 100 fixed section combinations -> circular median of the 100 phases."""
    return pci.cal_theta_sections(mpp["samples"], rx_mac, tx_mac, cfg, freq_hz, n_sections=10, n_combos=100)


def combine_original(phase_ab_deg, phase_ba_deg):
    """((Θ̂_ab + Θ̂_ba) / 2) mod π  (post_processing.load_combined_phases / exciter_variance.bidirectional)."""
    return np.deg2rad(pco.combine_bidirectional_phase(np.asarray(phase_ab_deg), np.asarray(phase_ba_deg)))


MPP_EXTRACTORS = {
    "original": mpp_original,
    "robust": mpp_robust,
}
PHASE_CALCS = {
    "original": phase_original,
    **{f"pct{q}": make_phase_pct(q) for q in (30, 40, 60, 70, 80, 90)},
    "sections": phase_sections,
}
COMBINERS = {
    "original": combine_original,
}


# ================================================================ loading

def fast_parse(text):
    """'[1.0, 2.0, ...]' -> ndarray (much faster than ast.literal_eval); falls back to mppx.parse_float_array.

    Repairs a decimal point stored as a comma ('105,4398' instead of '105.4398', seen once in rf_gen): every
    sample is written as a float repr and so contains '.', and two adjacent tokens without one can only be a
    split number.
    """
    tokens = [t.strip() for t in text.strip("[] \n").split(",")]
    if not all("." in t or "e" in t for t in tokens):
        fixed, i = [], 0
        while i < len(tokens):
            t = tokens[i]
            if "." not in t and i + 1 < len(tokens) and "." not in tokens[i + 1] and t.lstrip("-").isdigit():
                fixed.append(f"{t}.{tokens[i + 1]}")
                i += 2
            else:
                fixed.append(t)
                i += 1
        print(f"    repaired a comma-for-decimal-point in a trace ({len(tokens)} -> {len(fixed)} samples)")
        tokens = fixed
    try:
        return np.array(tokens, dtype=float)
    except ValueError:
        return mppx.parse_float_array(text)


def data_csv(path):
    """A data CSV, or the collected (not _improvedv1) CSV inside a dataset folder -- the raw traces are re-extracted."""
    path = Path(path)
    return path if path.is_file() else experiment_csv(path, "original")


def load_traces(path):
    """Raw traces + metadata of one data CSV (or dataset folder), cached as pickle (parsing is the slow part)."""
    csv = data_csv(path)
    cache = CACHE_DIR / f"phase_pipeline_traces_{csv.parent.name}_{csv.stem}.pkl"
    if cache.exists():
        try:
            return pd.read_pickle(cache)
        except ModuleNotFoundError:          # pickled under another numpy major version
            pass
    print(f"  reading {csv} …")
    cols = pd.read_csv(csv, nrows=0).columns
    keep = ["Rx", "Tx", FREQ_COL, RUN_COL, REP_COL, VOLT_COL, "Connection", "Sample Rate (Hz)",
            "Dwell (samples)", "Unidirectional Phase (deg)"]
    df = pd.read_csv(csv, usecols=[c for c in keep if c in cols])
    df[VOLT_COL] = df[VOLT_COL].map(fast_parse)
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_pickle(cache)
    return df


def channels_of(path):
    cols = pd.read_csv(data_csv(path), nrows=0).columns
    ch = sorted(int(c.split("_")[1]) for c in cols if c.startswith("Channel_") and c.endswith("_median"))
    return ch or mppx.DEFAULT_CHANNELS


# ================================================================ compute

def row_meta(r):
    """meta dict for an MPP extractor from one data row (dict or Series)."""
    get = lambda k: r[k] if k in r and pd.notna(r[k]) else None   # noqa: E731
    return {"dwell_samples": get("Dwell (samples)"), "rate_hz": get("Sample Rate (Hz)"),
            "connection": get("Connection")}


def unidirectional_phases(df, mpp_fn, phase_fn, channels, cfg, tag_mac):
    """Θ̂ (deg) for every MPP row. Rows whose extraction/phase fails get NaN and are reported."""
    out = np.full(len(df), np.nan)
    failures = 0
    for i, r in enumerate(df.to_dict("records")):
        try:
            mpp = mpp_fn(np.asarray(r[VOLT_COL], float), row_meta(r), channels)
            out[i] = phase_fn(mpp, tag_mac[r["Rx"]], tag_mac[r["Tx"]], cfg, float(r[FREQ_COL]) * 1e6)
        except Exception as e:
            failures += 1
            if failures <= 5:
                print(f"    {r['Rx']} <- {r['Tx']} run {r[RUN_COL]}: {type(e).__name__}: {e}")
    if failures:
        print(f"    {failures}/{len(df)} MPPs failed -> NaN")
    return out


def bidirectional(uni, combine_fn):
    """Join Rx=a/Tx=b with Rx=b/Tx=a (a < b) per freq, run, rep -> θ̂ (rad) per link."""
    u = uni[["Rx", "Tx", FREQ_COL, RUN_COL, REP_COL, "phase_deg"]]
    fwd = u[u["Rx"].map(tag_num) < u["Tx"].map(tag_num)]
    rev = u[u["Rx"].map(tag_num) > u["Tx"].map(tag_num)]
    m = fwd.merge(rev, left_on=["Rx", "Tx", FREQ_COL, RUN_COL, REP_COL],
                  right_on=["Tx", "Rx", FREQ_COL, RUN_COL, REP_COL], suffixes=("_a", "_b"), validate="one_to_one")
    m["THETA"] = combine_fn(m["phase_deg_a"].to_numpy(), m["phase_deg_b"].to_numpy())
    m = m.rename(columns={"Rx_a": "T1", "Tx_a": "T2", FREQ_COL: "freq", RUN_COL: "run"})
    m["T1"], m["T2"] = m["T1"].map(tag_num), m["T2"].map(tag_num)
    return m.dropna(subset=["THETA"])[["T1", "T2", "freq", "run", "THETA"]]


def link_theta_sd(bidir, positions, exciter):
    rows = []
    for (t1, t2, freq), g in bidir.groupby(["T1", "T2", "freq"]):
        if t1 not in positions or t2 not in positions or len(g) < 2:
            continue
        (c1, r1), (c2, r2) = label_grid(positions[t1]), label_grid(positions[t2])
        rows.append(dict(exciter=exciter, T1=t1, T2=t2, freq=freq, n_runs=len(g),
                         dist_cm=TILE_CM * np.hypot(c1 - c2, r1 - r2),
                         theta_sd_deg=theta_sd_deg(g["THETA"])))
    return pd.DataFrame(rows)


def parse_runs(s):
    if s is None:
        return None
    out = set()
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


# ================================================================ plot

def plot_theta_per_distance(res, path, title):
    """Box plot of per-link θ̂ SD at each tag-to-tag distance, exciters side by side, links as dots."""
    dists = np.sort(res["dist_cm"].unique())
    pos = {d: i for i, d in enumerate(dists)}
    n_exc, width = len(EXCITERS), 0.36
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(max(9, 0.75 * len(dists) + 2), 4.8))
    for i, (key, label) in enumerate(EXCITERS.items()):
        d = res[res["exciter"] == key]
        off = (i - (n_exc - 1) / 2) * width
        groups = [d.loc[d["dist_cm"] == dist, "theta_sd_deg"].to_numpy() for dist in dists]
        ax.boxplot(groups, positions=np.arange(len(dists)) + off, widths=width * 0.8, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor="none", edgecolor=COLORS[i], linewidth=1.2),
                   medianprops=dict(color=COLORS[i], linewidth=2),
                   whiskerprops=dict(color=COLORS[i]), capprops=dict(color=COLORS[i]), zorder=2)
        x = d["dist_cm"].map(pos).to_numpy() + off + rng.uniform(-width * 0.25, width * 0.25, len(d))
        med = np.nanmedian(d["theta_sd_deg"]) if len(d) else np.nan
        ax.scatter(x, d["theta_sd_deg"], s=12, color=COLORS[i], alpha=0.55, linewidth=0,
                   label=f"{label} (median {med:.2f}°)", zorder=3)
    for j in range(1, len(dists), 2):
        ax.axvspan(j - 0.5, j + 0.5, color="#f3f2ee", zorder=0)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("θ̂ standard deviation over runs (deg)")
    style(ax)
    counts = res[res["exciter"] == next(iter(EXCITERS))].groupby("dist_cm").size()
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels([f"{d:.0f}\nn={counts.get(d, 0)}" for d in dists], fontsize=8)
    ax.set_xlim(-0.5, len(dists) - 0.5)
    ax.set_xlabel("tag-to-tag distance (cm); n = links per exciter")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.13), ncol=2)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ================================================================ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mpp", default="original", help=f"MPP extraction algorithm: {list(MPP_EXTRACTORS)}")
    ap.add_argument("--phase", default="original", help=f"phase calc algorithm: {list(PHASE_CALCS)}")
    ap.add_argument("--combine", default="original", help=f"bidirectional combiner: {list(COMBINERS)}")
    ap.add_argument("--runs", help="runs to use, e.g. 0-24 or 0,2,4 (default: all)")
    ap.add_argument("--out", type=Path, help="output dir (default: out/<mpp>__<phase>__<combine>)")
    ap.add_argument("--list", action="store_true", help="list registered algorithms and exit")
    args = ap.parse_args()

    if args.list:
        for name, reg in [("--mpp", MPP_EXTRACTORS), ("--phase", PHASE_CALCS), ("--combine", COMBINERS)]:
            print(name)
            for k, f in reg.items():
                print(f"  {k:12s} {(f.__doc__ or '').strip().splitlines()[0]}")
        return
    for flag, val, reg in [("--mpp", args.mpp, MPP_EXTRACTORS), ("--phase", args.phase, PHASE_CALCS),
                           ("--combine", args.combine, COMBINERS)]:
        if val not in reg:
            ap.error(f"{flag} {val!r} unknown, choose from {list(reg)}")
    mpp_fn, phase_fn, combine_fn = MPP_EXTRACTORS[args.mpp], PHASE_CALCS[args.phase], COMBINERS[args.combine]

    out = args.out or OUT_ROOT / f"{args.mpp}__{args.phase}__{args.combine}"
    out.mkdir(parents=True, exist_ok=True)
    runs = parse_runs(args.runs)
    cfg = pco.load_hw_config()
    tag_mac = pco.load_tag_mac_mapping()

    uni_parts, sd_parts = [], []
    for key in EXCITERS:
        exc_dir = DATA_DIR / key
        df = load_traces(exc_dir)
        if runs is not None:
            df = df[df[RUN_COL].isin(runs)].reset_index(drop=True)
        t0 = time.time()
        print(f"  {key}: {len(df)} MPPs, mpp={args.mpp}, phase={args.phase} …")
        df = df.assign(phase_deg=unidirectional_phases(df, mpp_fn, phase_fn, channels_of(exc_dir), cfg, tag_mac))
        print(f"    done in {time.time() - t0:.1f}s")
        uni_parts.append(df.drop(columns=[VOLT_COL]).assign(exciter=key))
        sd_parts.append(link_theta_sd(bidirectional(df, combine_fn), load_positions(exc_dir), key))

    uni = pd.concat(uni_parts, ignore_index=True)
    res = pd.concat(sd_parts, ignore_index=True)
    uni.to_csv(out / "uni_phases.csv", index=False)

    for freq, r in res.groupby("freq"):
        tag = f"_{freq}MHz" if res["freq"].nunique() > 1 else ""
        n_runs = int(r["n_runs"].median())
        title = (f"Per-link θ̂ SD over runs per exciter, {freq} MHz, {n_runs} runs  "
                 f"[mpp={args.mpp}, phase={args.phase}, combine={args.combine}]")
        r.to_csv(out / f"std_per_link{tag}.csv", index=False)
        plot_theta_per_distance(r, out / f"std_per_distance_theta{tag}.png", title)
        print(f"  {freq} MHz median θ̂ SD: " + ", ".join(
            f"{EXCITERS[k]} {g['theta_sd_deg'].median():.2f}°" for k, g in r.groupby("exciter")))

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
