"""Figures and numbers for doc/eval.tex (experiments 1-9, 775-955 MHz).

Run from the repo with the RFImaging .venv (needs the MATLAB engine for the SDP step):
    ../.venv/bin/python doc/make_figures.py            # localize (MATLAB) + figures
    ../.venv/bin/python doc/make_figures.py --cached   # figures from doc/loc_results_<band>.npz
Localization is the same pipeline as localization_merged.ipynb (SDP + likelihood, merged).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

DOC = Path(__file__).resolve().parent
ROOT = DOC.parent  # multitag/
FIG = DOC / "figs"
FIG.mkdir(exist_ok=True)
EXPS = list(range(1, 10))
BAND = "775-955"
CACHE = DOC / f"loc_results_{BAND}.npz"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.4, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL_W = 3.45  # in, IEEEtran column width


def load_estimates():
    est = pd.concat([pd.read_csv(ROOT / str(e) / "distance_estimates.csv").assign(experiment=e) for e in EXPS])
    return est[est["band (MHz)"] == BAND]


# ---------------------------------------------------------------- localization (MATLAB) or cache
def localize():
    sys.path.insert(0, str(ROOT))
    nb = json.load(open(ROOT / "localization_merged.ipynb"))
    src = "".join(nb["cells"][1]["source"]) + "\n" + "".join(nb["cells"][3]["source"])
    import os
    cwd = os.getcwd()
    os.chdir(ROOT)
    g = {}
    exec(src, g)  # setup + method definitions from the notebook (starts the MATLAB engine)
    out = {}
    for e in EXPS:
        exp = Path(str(e))
        pts = pd.read_csv(exp / "points.csv")[["x", "y"]].to_numpy()
        anchors = g["pick_anchors"](pts)
        tags = np.setdiff1d(np.arange(len(pts)), anchors)
        D_svr, R = g["load_inputs"](exp, BAND)
        lkhd = g["lkhd_localize"](R, pts, anchors, tags)
        sdp = g["sdp_localize"](D_svr, pts, anchors, tags)
        merged, _, _ = g["merge_locations"](lkhd, sdp, lam=g["MERGE_LAM"], near_mult=g["NEAR_MULT"],
                                            far_mult=g["FAR_MULT"])
        out[f"pts{e}"], out[f"anc{e}"], out[f"tag{e}"], out[f"est{e}"] = pts, anchors, tags, merged
        print(f"exp {e}: median error {np.median(np.linalg.norm(pts[tags] - merged, axis=1)) * 100:.1f} cm")
    g["eng"].quit()
    os.chdir(cwd)
    np.savez(CACHE, **out)


if "--cached" not in sys.argv:
    localize()
L = np.load(CACHE)
loc_err = {e: np.linalg.norm(L[f"pts{e}"][L[f"tag{e}"]] - L[f"est{e}"], axis=1) * 100 for e in EXPS}  # cm


def cdf(ax, x, **kw):
    x = np.sort(x)
    ax.plot(np.r_[0, x], np.r_[0, np.arange(1, len(x) + 1) / len(x)], **kw)


def style(ax, xlabel):
    ax.set_xlabel(xlabel); ax.set_ylabel("CDF")
    ax.set_ylim(0, 1); ax.set_xlim(left=0)
    ax.grid(color="0.85", lw=0.5); ax.set_axisbelow(True)


# ---------------------------------------------------------------- ranging CDF
est = load_estimates()
err_svr = np.abs(est["dist_est_svr"] - est["distance"]).to_numpy() * 100
err_unw = np.abs(est["distance_est_unwrap"] - est["distance"]).to_numpy() * 100
fig, ax = plt.subplots(figsize=(COL_W, 2.1))
cdf(ax, err_svr, color="C0", label="SVR")
cdf(ax, err_unw, color="C1", ls="--", label="Unwrap")
ax.set_xlim(0, 100)
style(ax, "Absolute ranging error (cm)")
ax.legend(loc="lower right")
fig.savefig(FIG / "ranging_cdf.pdf"); plt.close(fig)

# ---------------------------------------------------------------- localization CDF
all_loc = np.concatenate([loc_err[e] for e in EXPS])
fig, ax = plt.subplots(figsize=(COL_W, 1.8))
cdf(ax, all_loc, color="C0")
ax.set_xlim(0, np.ceil(all_loc.max() / 10) * 10)
style(ax, "Localization error (cm)")
fig.savefig(FIG / "loc_cdf.pdf"); plt.close(fig)

# ---------------------------------------------------------------- link distance distribution
links = est.groupby(["experiment", "tagA_num", "tagB_num"])["distance"].first()
fig, ax = plt.subplots(figsize=(COL_W, 1.9))
ax.hist(links.to_numpy(), bins=np.arange(0, links.max() + 0.1, 0.1), color="C0", edgecolor="white", lw=0.5)
ax.set_xlabel("Tag-to-tag distance (m)"); ax.set_ylabel("Number of links")
ax.set_xlim(left=0); ax.grid(axis="y", color="0.85", lw=0.5); ax.set_axisbelow(True)
fig.savefig(FIG / "link_dist.pdf"); plt.close(fig)

# ---------------------------------------------------------------- per-experiment layouts
# coordinates shifted per experiment so the layout is centred in a [0, span] x [0, span] panel
box = {e: np.vstack([L[f"pts{e}"], L[f"est{e}"]]) for e in EXPS}
span = max(np.ptp(box[e], axis=0).max() for e in EXPS)  # same scale on every panel (m)
span = np.ceil((span + 0.2) / 0.5) * 0.5
origin = {e: (box[e].min(0) + box[e].max(0)) / 2 - span / 2 for e in EXPS}
plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "legend.fontsize": 12})
fig, axs = plt.subplots(3, 3, figsize=(7.16, 7.6), sharex=False, sharey=False)
for ax, e in zip(axs.flat, EXPS):
    pts, anc, tag, est_xy = L[f"pts{e}"] - origin[e], L[f"anc{e}"], L[f"tag{e}"], L[f"est{e}"] - origin[e]
    for t, p in zip(pts[tag], est_xy):
        ax.plot([t[0], p[0]], [t[1], p[1]], "-", color="C0", lw=0.7, zorder=1)
    ax.plot(*pts[anc].T, "ks", ms=5, label="Anchor", clip_on=False)
    ax.plot(*pts[tag].T, "o", color="C3", ms=5, label="Tag (ground truth)", clip_on=False)
    ax.plot(*est_xy.T, "*", color="C0", ms=8, label="Tag (estimated)", clip_on=False)
    ax.set_xlim(0, span); ax.set_ylim(0, span)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5)); ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.grid(color="0.85", lw=0.5); ax.set_axisbelow(True)
    ax.set_title(f"Location {e} (median {np.median(loc_err[e]):.1f} cm)", fontsize=11, pad=3)
for ax in axs[-1]:
    ax.set_xlabel("x (m)")
for ax in axs[:, 0]:
    ax.set_ylabel("y (m)")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.6, w_pad=0.4)
fig.savefig(FIG / "layouts.pdf"); plt.close(fig)

# ---------------------------------------------------------------- photos (experiment 10)
for name, src in [("photo_a", "0771A6FC-64AC-477A-B034-6EABBB66299C.JPG"),
                  ("photo_b", "BFEC8F28-6308-4F74-B90A-EEDE599A9856.JPG")]:
    im = ImageOps.exif_transpose(Image.open(ROOT / "10" / src))
    im.thumbnail((1600, 1600))
    im.save(FIG / f"{name}.jpg", quality=88)

# ---------------------------------------------------------------- numbers for the text
rows = []
for e in EXPS:
    p3 = pd.read_csv(ROOT / str(e) / "points.csv").to_numpy()
    d = links.loc[e].to_numpy()
    sub = est[est.experiment == e]
    rows.append({"exp": e, "nodes": len(p3), "anchors": len(L[f"anc{e}"]), "tags": len(L[f"tag{e}"]),
                 "links": len(d), "dmin": d.min(), "dmed": np.median(d), "dmax": d.max(),
                 "xspan": np.ptp(p3[:, 0]), "yspan": np.ptp(p3[:, 1]), "zspan": np.ptp(p3[:, 2]),
                 "svr_med": np.median(np.abs(sub.dist_est_svr - sub.distance)) * 100,
                 "unw_med": np.median(np.abs(sub.distance_est_unwrap - sub.distance)) * 100,
                 "loc_med": np.median(loc_err[e]), "loc_mean": loc_err[e].mean()})
tab = pd.DataFrame(rows)
tab.round(3).to_csv(DOC / "summary.csv", index=False)
print(tab.round(2).to_string(index=False))
q = lambda x, p: np.percentile(x, p)
print(f"links {len(links)}, samples {len(est)}, distance {links.min():.2f}-{links.max():.2f} m, median {links.median():.2f}")
for n, x in [("SVR", err_svr), ("Unwrap", err_unw), ("Loc", all_loc)]:
    print(f"{n}: n={len(x)} median {q(x, 50):.1f} mean {x.mean():.1f} p90 {q(x, 90):.1f} max {x.max():.1f} cm")

# ---------------------------------------------------------------- generated LaTeX (table rows + numbers in the text)
f1 = lambda v: f"{v:.1f}"
body = [f"{r.exp:>2} & {r.nodes} & {r.anchors} & {r.tags} & {r.links:>3} & {r.dmin:.2f} & {r.dmed:.2f} & {r.dmax:.2f} & "
        f"{r.xspan:.2f} & {r.yspan:.2f} & {f1(r.svr_med)} & {f1(r.unw_med)} & {f1(r.loc_med)} \\\\"
        for r in tab.itertuples()]
body += [r"\midrule", f"All & {tab.nodes.sum()} & {tab.anchors.sum()} & {tab.tags.sum()} & {tab.links.sum()} & "
         f"{links.min():.2f} & {links.median():.2f} & {links.max():.2f} & & & {f1(q(err_svr, 50))} & "
         f"{f1(q(err_unw, 50))} & {f1(q(all_loc, 50))} \\\\"]
(DOC / "table_rows.tex").write_text("\n".join(body) + "\n")
best, worst = tab.loc[tab.loc_med.idxmin()], tab.loc[tab.loc_med.idxmax()]
macros = {
    "SvrMed": f1(q(err_svr, 50)), "SvrNinety": f1(q(err_svr, 90)), "UnwMed": f1(q(err_unw, 50)),
    "UnwNinety": f1(q(err_unw, 90)), "UnwOver": f"{(err_unw > 100).mean() * 100:.1f}",
    "SvrOver": f"{(err_svr > 100).mean() * 100:.2f}",
    "SvrBetterAll": "every" if (tab.svr_med < tab.unw_med).all() else f"{(tab.svr_med < tab.unw_med).sum()} of {len(EXPS)}",
    "LocMed": f1(q(all_loc, 50)), "LocMean": f1(all_loc.mean()), "LocNinety": f1(q(all_loc, 90)),
    "LocMax": f1(all_loc.max()), "LocBest": f1(best.loc_med), "LocBestExp": str(int(best.exp)),
    "LocWorst": f1(worst.loc_med), "LocWorstExp": str(int(worst.exp)),
}
(DOC / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
print(macros)
