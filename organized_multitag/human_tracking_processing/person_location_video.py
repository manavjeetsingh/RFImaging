"""Estimate where the person is from the 915 MHz links, and make a video of it next to the camera recording.

For walking915_1 / walking915_2 (one frequency, 70 runs; every run measures all 28 links within ~0.4 s):

1. Frame. Every run k gives one vector with one score per link (T1-T2, ..., T7-T8). The score is the one of the
   motion detection cell in human_tracking_processing.ipynb:
       score = sqrt((phase residual / s_phase)^2 + (log-alpha residual / s_logalpha)^2)
   where a residual is the sample minus the link's rolling median (MD_WINDOW runs, centred; the phase is compared on
   the doubled angle so there is no wrap at 0 / 180 deg), and s_phase, s_logalpha are the robust noise (MAD) of the
   same residuals of that link in baseline915.
2. Cluster. All 6-link combinations whose link centres (midpoints) are all within --radius metres of the
   combination's centroid are candidates. At every frame, the candidate with the largest cluster score is selected.
   The cluster score (--agg) must not let one very strong link carry five weak ones, so the default is the
   geometric mean of (1 + link score): it is only high when all 6 links are raised together. Other choices:
   median, trim (mean without the strongest link), mean, rms (the old behaviour, dominated by the strongest link).
3. Location. The plain average of the 6 link centres of the selected cluster (unweighted). It is shown as solid
   when the cluster score is above the --base-pct percentile of
   the same best-cluster score in baseline915 (i.e. bigger than anything the empty room produces), faded otherwise.
4. Track (--tracker, see tracking.py). The location above is decided per run on its own. The tracker links the
   runs: grid (default) is a Bayes filter on a floor grid whose likelihood is the soft cluster score at every point
   and whose motion model lets the person walk ~--walk-speed m/s between runs; kalman is a constant-velocity
   Kalman filter on the per-run cluster locations; none keeps the per-run location. --smooth uses all runs
   (forward-backward / RTS) instead of runs up to the current one.
5. Video. Camera recording (vid_start_time.txt in the experiment folder) on the left, layout on the right: links
   coloured by their score, updated when each link's measurement completes (the stop time of the later of its two
   MPPs); the selected cluster outlined and the location (with a short trail) updated when the run completes.

    python person_location_video.py walking915_1
    python person_location_video.py walking915_2 --radius 0.5 --speed 2
    python person_location_video.py walking915_1 --tracker kalman
Also writes <exp>_person_location.csv (one row per run: time, location, track, cluster, scores). Outputs of
--tracker kalman / none are named <exp>_person_location_kalman.* / _raw.*.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse, FancyArrowPatch

from link_distance_video import (DATA_DIR, HERE, HIGHLIGHT, CameraReader, bidirectional, link_curvature, load_csv,
                                 rotate_layout, tag_color, tag_num, tag_positions, upright_angle, video_start)
from tracking import grid_track, kalman_track, make_grid, soft_cluster_score, support

BASELINE = "baseline915"
MD_WINDOW = 11            # rolling-median window (runs), as in the notebook
CLUSTER_SIZE = 6
LOCATION = "#1baf7a"
TRAIL = 4                 # previous locations drawn behind the current one
RAW = "#7a7a7a"           # per-run location when a tracker is on
GRID_PAD = 0.35           # m around the tags, as the plot
SUFFIX = {"grid": "", "kalman": "_kalman", "none": "_raw"}    # output file name per tracker


# ---------------------------------------------------------------- scores

def link_matrix(m):
    # bidirectional rows -> link list, and runs × links arrays of THETA, alpha, ready time
    m = m.assign(link=list(zip(m["T1"].map(tag_num), m["T2"].map(tag_num))))
    links = sorted(m["link"].unique())
    piv = {c: m.pivot(index="Run Exp Num", columns="link", values=c)[links].to_numpy()
           for c in ("THETA", "alpha", "ready")}
    assert not any(np.isnan(v).any() for v in piv.values()), "some link is missing in some run"
    return links, piv["THETA"], piv["alpha"], piv["ready"]


def rolling_median(x, window=MD_WINDOW):
    # along the runs (axis 0) of a runs × links array
    return pd.DataFrame(x).rolling(window, center=True, min_periods=1).median().to_numpy()


def residuals(theta, attn):
    # -> phase residual (deg) and log-attenuation residual against the rolling median, runs × links
    u = np.exp(2j * theta)
    ref = rolling_median(u.real) + 1j * rolling_median(u.imag)
    d_phase = np.rad2deg(np.angle(u * np.conj(ref)) / 2)
    assert np.all(attn > 0), "alpha must be positive for log-attenuation"
    la = np.log(attn)
    return d_phase, la - rolling_median(la)


def mad_scale(x):
    # robust standard deviation of every column
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)


def link_scores(theta, attn, scales):
    d_phase, d_att = residuals(theta, attn)
    return np.hypot(d_phase / scales[0], d_att / scales[1])


# ---------------------------------------------------------------- clusters and location

def candidate_clusters(centres, radius, size=CLUSTER_SIZE):
    # all `size`-link combinations whose link centres are all within `radius` of their centroid -> combos × size
    combos = np.array(list(itertools.combinations(range(len(centres)), size)), dtype=np.int16)
    pts = centres[combos]                                          # combos × size × 2
    spread = np.linalg.norm(pts - pts.mean(axis=1, keepdims=True), axis=2).max(axis=1)
    return combos[spread <= radius]


AGGREGATES = {
    # links axis = last; every one maps runs × clusters × 6 -> runs × clusters
    "gmean": lambda s: np.expm1(np.log1p(s).mean(axis=-1)),          # geometric mean of 1 + score, minus 1
    "median": lambda s: np.median(s, axis=-1),
    "trim": lambda s: np.sort(s, axis=-1)[..., :-1].mean(axis=-1),   # mean without the strongest link
    "mean": lambda s: s.mean(axis=-1),
    "rms": lambda s: np.sqrt((s ** 2).mean(axis=-1)),
}


def best_clusters(scores, clusters, agg="gmean"):
    # -> index of the best cluster and its cluster score, per run
    cs = AGGREGATES[agg](scores[:, clusters])                         # runs × clusters
    best = cs.argmax(axis=1)
    return best, cs[np.arange(len(cs)), best]


def dominance(scores, clusters, best):
    # share of the selected cluster's total score that comes from its strongest link, per run (1/6 = even)
    s = scores[np.arange(len(best))[:, None], clusters[best]]
    return s.max(axis=1) / s.sum(axis=1)


# ---------------------------------------------------------------- pipeline

def analyse(exp, radius, base_pct, rotate, agg="gmean", max_link_length=None):
    exp_df = load_csv(DATA_DIR / exp)
    exp_m, base_m = bidirectional(exp_df), bidirectional(load_csv(DATA_DIR / BASELINE))
    for name, mm in ((exp, exp_m), (BASELINE, base_m)):
        assert mm["Frequency (MHz)"].nunique() == 1, f"{name}: expected one frequency"

    # layout (rotated for display; distances do not change)
    ref, markers = tag_positions(DATA_DIR / exp)
    rotate = upright_angle(ref) if rotate is None else rotate
    ref, markers = rotate_layout(ref, markers, rotate)
    xy = {int(t): ref.loc[t, ["x", "y"]].to_numpy(float) for t in ref.index}

    if max_link_length is not None:
        # drop links longer than max_link_length (m) from the experiment and the baseline before anything else
        def short(mm):
            length = np.array([np.linalg.norm(xy[tag_num(a)] - xy[tag_num(b)]) for a, b in zip(mm["T1"], mm["T2"])])
            return mm[length <= max_link_length]
        n_all = len(set(zip(exp_m["T1"], exp_m["T2"])))
        exp_m, base_m = short(exp_m), short(base_m)
        n_kept = len(set(zip(exp_m["T1"], exp_m["T2"])))
        print(f"  max link length {max_link_length:g} m: {n_kept} / {n_all} links kept")
        assert n_kept, f"no link is {max_link_length} m or shorter"
    links, theta, attn, ready = link_matrix(exp_m)
    b_links, b_theta, b_attn, _ = link_matrix(base_m)
    assert links == b_links, "experiment and baseline have different links"

    # empty-room noise of every link, then scores of the experiment and of the baseline itself
    b_dp, b_da = residuals(b_theta, b_attn)
    scales = (mad_scale(b_dp), mad_scale(b_da))
    scores = link_scores(theta, attn, scales)                      # runs × links
    b_scores = link_scores(b_theta, b_attn, scales)

    # link centres
    centres = np.array([(xy[a] + xy[b]) / 2 for a, b in links])

    clusters = candidate_clusters(centres, radius)
    assert len(clusters), f"no {CLUSTER_SIZE}-link cluster fits within {radius} m, use a larger --radius"
    best, best_rms = best_clusters(scores, clusters, agg)
    _, b_best_rms = best_clusters(b_scores, clusters, agg)
    threshold = float(np.percentile(b_best_rms, base_pct))

    loc = centres[clusters[best]].mean(axis=1)                     # plain average of the 6 link centres
    run_time = ready.max(axis=1)                                   # the run is complete when its last link is
    order = np.argsort(run_time)
    res = dict(exp=exp, exp_df=exp_df, links=links, scores=scores[order], ready=ready[order],
               run_time=run_time[order], clusters=clusters, best=best[order], best_rms=best_rms[order],
               threshold=threshold, loc=loc[order], agg=agg,
               dominance=dominance(scores, clusters, best)[order], centres=centres, ref=ref, markers=markers, rotate=rotate)
    print(f"{exp}: {len(links)} links, {len(run_time)} runs; {len(clusters)} candidate {CLUSTER_SIZE}-link clusters "
          f"within {radius} m; detection threshold {threshold:.2f} (p{base_pct:g} of {BASELINE}); "
          f"{(best_rms[order] > threshold).sum()} runs above it; cluster score = {agg}; "
          f"strongest link's share of the selected cluster: median {np.median(dominance(scores, clusters, best)):.2f}"
          f" (1/6 = {1 / 6:.2f} is even)")
    return res


def track(res, tracker, smooth, beta, walk_speed, rho, reach, meas_std, accel_std):
    # adds res["track"]: mean (runs × 2), cov (runs × 2 × 2), and for the grid tracker the posterior per run
    valid = res["best_rms"] > res["threshold"]
    kind = "smooth" if smooth else "filt"
    if tracker == "none":
        res["track"] = None
        return res
    if tracker == "grid":
        xs, ys = make_grid(res["ref"], GRID_PAD)
        mask = support(res["centres"], xs, ys, reach)
        soft = soft_cluster_score(res["scores"], res["centres"], xs, ys, rho, CLUSTER_SIZE)
        o = grid_track(soft, valid, res["run_time"], xs, ys, beta, walk_speed, mask, smooth)
        t = dict(post=o[kind], xs=xs, ys=ys, label=f"grid Bayes filter (beta {beta:g}, {walk_speed:g} m/s, "
                                                     f"rho {rho:g} m)")
    else:
        o = kalman_track(res["loc"], res["best_rms"], valid, res["run_time"], res["threshold"], meas_std, accel_std,
                         walk_speed, smooth)
        t = dict(label=f"constant-velocity Kalman filter (meas {meas_std:g} m, accel {accel_std:g} m/s²)")
        print(f"  kalman: {o['used'].sum()} / {len(valid)} runs used (others below threshold or gated out)")
    t.update(kind=tracker, mean=o[f"{kind}_mean"], cov=o[f"{kind}_cov"], smooth=smooth)
    step = lambda a: np.linalg.norm(np.diff(a, axis=0), axis=1).mean()
    print(f"  {tracker} track{' (smoothed)' if smooth else ''}: mean step between runs {step(t['mean']):.2f} m "
          f"(per-run location {step(res['loc']):.2f} m), median 1-sigma radius "
          f"{np.median(np.sqrt(np.trace(t['cov'], axis1=1, axis2=2))):.2f} m")
    res["track"] = t
    return res


def save_table(res, path):
    links = res["links"]
    rows = []
    for k in range(len(res["run_time"])):
        c = res["clusters"][res["best"][k]]
        rows.append({"run": k, "time (s)": res["run_time"][k], "x": res["loc"][k, 0], "y": res["loc"][k, 1],
                     **({} if res["track"] is None else
                        {"track x": res["track"]["mean"][k, 0], "track y": res["track"]["mean"][k, 1],
                         "track sd (m)": np.sqrt(np.trace(res["track"]["cov"][k]))}),
                     "cluster score": res["best_rms"][k], "strongest link share": res["dominance"][k], "above threshold": res["best_rms"][k] > res["threshold"],
                     "cluster": " ".join(f"T{links[i][0]}-T{links[i][1]}" for i in c),
                     **{f"T{a}-T{b}": res["scores"][k, i] for i, (a, b) in enumerate(links)}})
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4f")
    print(f"saved {path}  (x, y in the rotated layout frame, metres)")


# ---------------------------------------------------------------- video

def make_video(res, out, fps, speed, vmax, dpi, vid_start):
    exp, links, ref, markers = res["exp"], res["links"], res["ref"], res["markers"]
    scores, ready, run_time = res["scores"], res["ready"], res["run_time"]
    vmax = vmax or float(np.percentile(scores, 99))
    norm = Normalize(0, vmax, clip=True)
    cmap = plt.get_cmap("Blues")
    t_end = float(run_time.max()) + 1.0

    exp_df = res["exp_df"]
    t0 = exp_df["MPP Start Time (s)"].min()
    mpps = pd.DataFrame({"stop": exp_df["MPP Stop Time (s)"] - t0, "Rx": exp_df["Rx"], "Tx": exp_df["Tx"]})
    mpps = mpps.sort_values("stop").reset_index(drop=True)
    mpp_stop = mpps["stop"].to_numpy()

    start = video_start(DATA_DIR / exp, vid_start)
    movs = sorted((DATA_DIR / exp).glob("*.mov"))
    if start is None or not movs:
        raise SystemExit(f"{exp}: need a .mov and vid_start_time.txt (or --vid-start) for the camera panel")
    camera = CameraReader(movs[0], start)
    print(f"  camera: {movs[0].name}, experiment starts at {start:.2f} s in the recording")

    fig = plt.figure(figsize=(15.5, 8.4))
    cam_ax = fig.add_axes([0.01, 0.31, 0.50, 0.60])
    ax = fig.add_axes([0.52, 0.31, 0.38, 0.60])
    cax = fig.add_axes([0.915, 0.34, 0.012, 0.54])
    tax = fig.add_axes([0.06, 0.06, 0.90, 0.17])
    cam_ax.set_xticks([])
    cam_ax.set_yticks([])
    cam_img = cam_ax.imshow(camera.frame(0.0))
    cam_title = cam_ax.set_title("", fontsize=11)

    # layout: floor labels, links, tags, location
    pad = 0.35
    ax.scatter(markers["x"], markers["y"], s=6, c="#9a9a9a", zorder=1)
    xy = {int(t): ref.loc[t, ["x", "y"]].to_numpy(float) for t in ref.index}
    patches, outlines = [], []
    for a, b in links:
        cs = f"arc3,rad={link_curvature(ref, f'Tag{a}', f'Tag{b}')}"
        outlines.append(ax.add_patch(FancyArrowPatch(xy[a], xy[b], arrowstyle="-", connectionstyle=cs,
                                                     color=HIGHLIGHT, lw=0, alpha=0.0, zorder=1.5,
                                                     capstyle="round")))
        patches.append(ax.add_patch(FancyArrowPatch(xy[a], xy[b], arrowstyle="-", connectionstyle=cs,
                                                    color="#e6e6e6", lw=0.8, ls="--", zorder=2, capstyle="round")))
    centre_dots = ax.scatter(res["centres"][:, 0], res["centres"][:, 1], s=14, c="#9a9a9a", marker="x",
                             linewidths=0.8, zorder=2.5)
    for tag, r in ref.iterrows():
        ax.scatter(r.x, r.y, s=260, color=tag_color(tag), edgecolors="white", linewidths=1.5, zorder=4)
        ax.annotate(tag, (r.x, r.y), ha="center", va="center", fontsize=9, color="white", fontweight="bold",
                    zorder=5)
    trk = res["track"]
    heat = ellipse = raw = None
    if trk is not None and "post" in trk:
        step = trk["xs"][1] - trk["xs"][0]
        heat = ax.imshow(np.zeros_like(trk["post"][0]), origin="lower", zorder=0.5, interpolation="bilinear",
                         extent=(trk["xs"][0] - step / 2, trk["xs"][-1] + step / 2,
                                 trk["ys"][0] - step / 2, trk["ys"][-1] + step / 2),
                         cmap=LinearSegmentedColormap.from_list("post", [(1, 1, 1, 0), LOCATION]), vmin=0, vmax=1)
    if trk is not None:
        ellipse = ax.add_patch(Ellipse((0, 0), 0, 0, fill=False, ec=LOCATION, lw=1.5, ls="--", zorder=5.4))
        raw = ax.scatter([np.nan], [np.nan], marker="o", s=70, facecolors="none", edgecolors=RAW, linewidths=1.5,
                         zorder=5.6, label="this run alone (cluster centre)")
    est = res["loc"] if trk is None else trk["mean"]
    trail = ax.plot([], [], "-o", color=LOCATION, ms=5, lw=1.2, alpha=0.5, zorder=5.5)[0]
    person = ax.scatter([np.nan], [np.nan], marker="*", s=520, color=LOCATION, edgecolors="white", linewidths=1.5,
                        zorder=6)
    ax.set_xlim(ref["x"].min() - pad, ref["x"].max() + pad)
    ax.set_ylim(ref["y"].min() - pad, ref["y"].max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#dddddd")
    title = ax.set_title("", fontsize=11)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 label=f"link score (× {BASELINE} noise)")
    if trk is None:
        fig.suptitle(f"{exp}: person location from the {CLUSTER_SIZE}-link cluster with the largest score "
                     f"(star = average of its link centres)", fontsize=11, y=0.975)
    else:
        fig.suptitle(f"{exp}: tracked location, {trk['label']}{', smoothed over all runs' if trk['smooth'] else ''}"
                     f"\nstar = track, dashed = 1-sigma"
                     f"{', green = probability of each position' if heat is not None else ''}"
                     f", grey circle = this run's best cluster alone", fontsize=10.5, y=0.985)

    # timeline: best-cluster score per run and the empty-room threshold
    tax.step(run_time, res["best_rms"], where="post", color="#08306b", lw=1.3, label=f"best cluster score ({res['agg']})")
    tax.axhline(res["threshold"], color="#c0392b", ls="--", lw=1, label=f"{BASELINE} threshold")
    cursor = tax.axvline(0, color=HIGHLIGHT, lw=1.5)
    tax.set_xlim(0, t_end)
    tax.set_ylim(0, max(res["best_rms"].max(), res["threshold"]) * 1.08)
    tax.set_xlabel("time from the first MPP (s)")
    tax.set_ylabel("score")
    tax.legend(fontsize=8, loc="upper right", frameon=False)
    tax.grid(alpha=0.3)

    n_frames = int(np.ceil(t_end / speed * fps))
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(out), dpi=dpi):
        for f in range(n_frames):
            t = f / fps * speed
            # every link shows its latest completed measurement
            for i, patch in enumerate(patches):
                k = np.searchsorted(ready[:, i], t, side="right") - 1
                if k < 0:
                    continue
                d = norm(scores[k, i])
                patch.set_color(cmap(d))
                patch.set_linestyle("-")
                patch.set_linewidth(0.8 + 5.5 * d)
                patch.set_zorder(2 + d)
            # the latest completed run: its cluster and location
            k = np.searchsorted(run_time, t, side="right") - 1
            for o in outlines:
                o.set_alpha(0.0)
            if k >= 0:
                strong = res["best_rms"][k] > res["threshold"]
                for i in res["clusters"][res["best"][k]]:
                    outlines[i].set_alpha(0.85 if strong else 0.35)
                    outlines[i].set_linewidth(0.8 + 5.5 * norm(scores[k, i]) + 5)
                person.set_offsets([est[k]])
                person.set_alpha(1.0 if strong or trk is not None else 0.3)
                tr = est[max(0, k - TRAIL):k + 1]
                trail.set_data(tr[:, 0], tr[:, 1])
                if trk is not None:
                    raw.set_offsets([res["loc"][k]])
                    raw.set_alpha(1.0 if strong else 0.4)
                    vals, vecs = np.linalg.eigh(trk["cov"][k])
                    ellipse.set_center(est[k])
                    ellipse.set_width(2 * np.sqrt(max(vals[1], 0)))
                    ellipse.set_height(2 * np.sqrt(max(vals[0], 0)))
                    ellipse.set_angle(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))
                if heat is not None:
                    q = trk["post"][k]
                    heat.set_data(q / q.max() * 0.75)
                title.set_text(f"t = {t:6.2f} s   run {k + 1} / {len(run_time)}   cluster score "
                               f"{res['best_rms'][k]:.2f} ({'above' if strong else 'below'} threshold)")
            else:
                title.set_text(f"t = {t:6.2f} s   waiting for the first run")
            j = np.searchsorted(mpp_stop, t, side="right") - 1
            if j >= 0:
                tc = mpp_stop[j]
                cam_img.set_data(camera.frame(tc))
                cam_title.set_text(f"camera at the latest MPP: t = {tc:6.2f} s  "
                                   f"(T{tag_num(mpps['Rx'][j])} ← T{tag_num(mpps['Tx'][j])}, "
                                   f"{start + tc:.2f} s into the recording)")
            cursor.set_xdata([t, t])
            writer.grab_frame()
            if f % (fps * 10) == 0:
                print(f"  frame {f} / {n_frames}  (t = {t:.1f} s)", flush=True)
    plt.close(fig)
    camera.close()
    print(f"saved {out}")


def main():
    global DATA_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", nargs="?", default="walking915_1", help="walking915_1 or walking915_2")
    ap.add_argument("--radius", type=float, default=0.4,
                    help="max distance (m) of a link centre from its cluster's centroid (default 0.4)")
    ap.add_argument("--agg", choices=list(AGGREGATES), default="gmean",
                    help="cluster score from its 6 link scores (default gmean: all links must be raised)")
    ap.add_argument("--base-pct", type=float, default=99,
                    help="percentile of the baseline915 best-cluster score used as the threshold (default 99)")
    ap.add_argument("--tracker", choices=["grid", "kalman", "none"], default="grid",
                    help="link the runs into a track (default grid Bayes filter; none = every run on its own)")
    ap.add_argument("--smooth", action="store_true",
                    help="track from all runs (forward-backward / RTS) instead of the runs up to the current one")
    ap.add_argument("--beta", type=float, default=4.0,
                    help="grid: likelihood = (1 + soft cluster score) ** beta; larger trusts each run more (default 4)")
    ap.add_argument("--walk-speed", type=float, default=0.45,
                    help="walking speed (m/s): grid motion std between runs, kalman initial velocity std "
                         "(default 0.45 = 1 mph)")
    ap.add_argument("--rho", type=float, default=0.3,
                    help="grid: width (m) of the Gaussian weight of the nearest link centres (default 0.3)")
    ap.add_argument("--reach", type=float, default=0.3,
                    help="grid: the person can only be within this distance (m) of some link centre (default 0.3)")
    ap.add_argument("--meas-std", type=float, default=0.3,
                    help="kalman: std (m) of a cluster location whose score is at the threshold (default 0.3)")
    ap.add_argument("--accel-std", type=float, default=0.3,
                    help="kalman: std (m/s^2) of the white-noise acceleration (default 0.3: the velocity can "
                         "change by about the walking speed between runs)")
    ap.add_argument("--out", type=Path, help="output .mp4 (default <exp>_person_location.mp4 next to this script)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time")
    ap.add_argument("--vmax", type=float, help="top of the link colour scale (default: 99th percentile of scores)")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--vid-start", help="time in the .mov at which the experiment starts (default: vid_start_time.txt)")
    ap.add_argument("--rotate", type=float, default=None, help="layout rotation in degrees (default: T1 above T4)")
    ap.add_argument("--no-video", action="store_true", help="only write the CSV")
    ap.add_argument("--max-link-length", type=float,
                    help="ignore links longer than this (m, e.g. 1.5 = 150 cm) in every calculation (default: no limit)")
    ap.add_argument("--data-dir", type=Path,
                    help=f"folder holding <exp> and {BASELINE} (default {DATA_DIR})")
    args = ap.parse_args()
    if args.data_dir is not None:
        DATA_DIR = args.data_dir.resolve()

    res = analyse(args.exp, args.radius, args.base_pct, args.rotate, args.agg, args.max_link_length)
    res = track(res, args.tracker, args.smooth, args.beta, args.walk_speed, args.rho, args.reach, args.meas_std,
                args.accel_std)
    name = f"{args.exp}_person_location{SUFFIX[args.tracker]}"
    if args.max_link_length is not None:
        name += f"_maxlen{args.max_link_length:g}m"
    save_table(res, HERE / f"{name}.csv")
    if not args.no_video:
        out = args.out or HERE / f"{name}.mp4"
        make_video(res, out, args.fps, args.speed, args.vmax, args.dpi, args.vid_start)


if __name__ == "__main__":
    main()
