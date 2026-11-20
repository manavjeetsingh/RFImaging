"""Prototype: probability map of the person's position from the phase AND attenuation of every link (two-state
"fade-level" link model), and a video of it next to the camera recording.

Model (per run k, per link l between tags a and b; x = a floor cell):
    measurement   z_l = (dtheta_l, dlogA_l): bidirectional phase (mod 180 deg) and log attenuation minus the link's
                  reference (--ref median: the median over the experiment's runs; baseline: the empty room).
    geometry      excess path  D_l(x) = |a - x| + |x - b| - |a - b|  (ellipses with foci at the two tags)
    link state    disturbed with probability  pi_l(x) = p_out + (p_in - p_out) exp(-D_l(x)^2 / (2 kappa^2)),
                  else empty. p_out > 0 because a person far from the link still changes it through multipath.
    empty   f_E   phase: von Mises on the doubled angle with the link's baseline915 noise; log A: Gaussian with the
                  link's baseline915 noise. Mixed with eps * f_D (glitches), which caps one link's say at log(1/eps).
    disturbed f_D the same with the noise widened by sd_phase_D (deg) and sd_logA_D.
    likelihood    log p(z | x) = beta * sum_l log(pi_l(x) f_D + (1 - pi_l(x)) f_E)
    nobody        log p(z | empty) = beta * sum_l log(p_out f_D + (1 - p_out) f_E)  ->  P(present), prior 1/2
Map: posterior with a flat prior over the grid, every run on its own (no memory between runs). Three models side
by side: phase only, attenuation only, phase + attenuation; a fourth panel shows the links coloured by their
log likelihood ratio (disturbed / empty) under phase + attenuation.
--fit: grid search of (kappa, p_in, p_out, sd_phase_D, sd_logA_D) maximising the mean log evidence of the
experiment's runs (no ground truth needed).

    python link_likelihood_map.py 1_915 --duration 40
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
from scipy.special import i0e, logsumexp

from link_distance_video import (HERE, ROOT, CameraReader, bidirectional, load_csv, rotate_layout, tag_num,
                                 tag_positions, upright_angle, video_start)

DATA_DIR = ROOT / "human_tracking_grid_data"
BASELINE = "baseline915"
STEP = 0.05               # grid cell (m)
PAD = 0.6                 # grid margin around the tags (m)
MODELS = ("phase", "attn", "both")
MODEL_LABEL = {"phase": "phase only", "attn": "attenuation only", "both": "phase + attenuation"}
MODEL_COLOR = {"phase": "#2a78d6", "attn": "#c0392b", "both": "#1baf7a"}
LINK_MODEL = "both"        # model whose per-link likelihood ratios colour the link panel
FLOOR = (np.deg2rad(0.5), 0.01)   # minimum noise std (phase rad, log A)


# ---------------------------------------------------------------- data

def link_arrays(exp):
    # -> links [(a, b)], and runs × links arrays: theta (rad, mod pi), log attenuation, ready time (s), start time (s)
    m = bidirectional(load_csv(DATA_DIR / exp))
    m["first"] = np.minimum(m["start_a"], m["start_b"])
    assert m["Frequency (MHz)"].nunique() == 1, f"{exp}: expected one frequency"
    m["link"] = list(zip(m["T1"].map(tag_num), m["T2"].map(tag_num)))
    links = sorted(m["link"].unique())
    piv = {c: m.pivot(index="Run Exp Num", columns="link", values=c)[links].to_numpy()
           for c in ("THETA", "alpha", "ready", "first")}
    assert not any(np.isnan(v).any() for v in piv.values()), f"{exp}: some link is missing in some run"
    assert np.all(piv["alpha"] > 0)
    return links, piv["THETA"], np.log(piv["alpha"]), piv["ready"], piv["first"]


def wrap(d):
    # phase difference mod pi -> (-pi/2, pi/2]
    return (d + np.pi / 2) % np.pi - np.pi / 2


def circ_median(theta):
    # per-link reference of an angle known mod pi
    c = np.angle(np.exp(2j * theta).mean(axis=0)) / 2
    return c + np.median(wrap(theta - c), axis=0)


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)


# ---------------------------------------------------------------- model

def log_vm_half(d, sd):
    # density of a phase difference d in (-pi/2, pi/2] (mod pi): von Mises on 2d with std 2*sd
    k = 1.0 / (2 * sd) ** 2
    return k * (np.cos(2 * d) - 1) - np.log(np.pi * i0e(k))


def log_gauss(d, sd):
    return -0.5 * (d / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi))


def link_logs(dth, dla, noise, sd_phase_d, sd_loga_d, eps, use="both"):
    # -> log f_E, log f_D (runs × links); use = "phase", "attn" or "both" (which measurements enter)
    s_th, s_la = noise
    sd_th = np.sqrt(s_th ** 2 + np.deg2rad(sd_phase_d) ** 2)
    sd_la = np.sqrt(s_la ** 2 + sd_loga_d ** 2)
    ph, at = use in ("phase", "both"), use in ("attn", "both")
    log_d = ph * log_vm_half(dth, sd_th) + at * log_gauss(dla, sd_la)
    log_e0 = ph * log_vm_half(dth, s_th) + at * log_gauss(dla, s_la)
    log_e = np.logaddexp(np.log1p(-eps) + log_e0, np.log(eps) + log_d)
    return log_e, log_d


def excess_path(xy_a, xy_b, pts):
    # cells × links
    return (np.linalg.norm(pts[:, None] - xy_a[None], axis=2) + np.linalg.norm(pts[:, None] - xy_b[None], axis=2)
            - np.linalg.norm(xy_a - xy_b, axis=1)[None])


def log_maps(log_e, log_d, dpath, kappa, p_in, p_out, beta):
    # -> log p(z | x) (runs × cells) and log p(z | nobody) (runs), both up to the same constant
    pi = p_out + (p_in - p_out) * np.exp(-dpath ** 2 / (2 * kappa ** 2))           # cells × links
    lr = np.exp(np.clip(log_d - log_e, -50, 50))                                     # runs × links
    # log(pi f_D + (1 - pi) f_E) - log f_E = log(1 + pi (LR - 1))
    ll = np.log1p(pi[None] * (lr[:, None] - 1)).sum(axis=2)                          # runs × cells
    l0 = np.log1p(p_out * (lr - 1)).sum(axis=1)
    return beta * ll, beta * l0


def present_prob(ll, l0):
    # P(someone present) with prior 1/2 and a flat prior over cells
    lp = logsumexp(ll, axis=1) - np.log(ll.shape[1])
    return 1 / (1 + np.exp(np.clip(l0 - lp, -50, 50)))


def mean_evidence(ll):
    return float((logsumexp(ll, axis=1) - np.log(ll.shape[1])).mean())


def fit(dth, dla, noise, dpath, eps, beta, use):
    grid = dict(kappa=[0.1, 0.25, 0.4, 0.6, 0.9], p_in=[0.8, 0.95, 0.99], p_out=[0.3, 0.5, 0.7, 0.8, 0.9],
                sd_phase_d=[10, 25, 60], sd_loga_d=[0.1, 0.3, 0.8])
    best = None
    for sp, sl in itertools.product(grid["sd_phase_d"], grid["sd_loga_d"]):
        le, ld = link_logs(dth, dla, noise, sp, sl, eps, use)
        for k, pin, pout in itertools.product(grid["kappa"], grid["p_in"], grid["p_out"]):
            if pout >= pin:
                continue
            ll, l0 = log_maps(le, ld, dpath, k, pin, pout, beta)
            # evidence of "person somewhere" relative to f_E alone; same constant for every parameter set
            ev = mean_evidence(ll) + beta * le.sum(axis=1).mean()
            if best is None or ev > best[0]:
                best = (ev, dict(kappa=k, p_in=pin, p_out=pout, sd_phase_d=sp, sd_loga_d=sl))
    print(f"  fit: mean log evidence {best[0]:.1f} at {best[1]}")
    return best[1]


def hpd_levels(p, masses=(0.5, 0.9)):
    # probability thresholds whose superlevel sets hold the given mass (for contours), ascending
    q = np.sort(p.ravel())[::-1]
    c = np.cumsum(q)
    return sorted(q[min(np.searchsorted(c, m), len(q) - 1)] for m in masses)


def moments(p, gx, gy):
    mx, my = (p * gx).sum(), (p * gy).sum()
    return mx, my, np.sqrt((p * ((gx - mx) ** 2 + (gy - my) ** 2)).sum())


# ---------------------------------------------------------------- pipeline

def analyse(args):
    links, th, la, ready, first = link_arrays(args.exp)
    b_links, b_th, b_la, _, _ = link_arrays(BASELINE)
    assert links == b_links, "experiment and baseline have different links"

    # empty-room noise per link (residuals of the baseline against its own reference)
    noise = (np.maximum(mad(wrap(b_th - circ_median(b_th))), FLOOR[0]),
             np.maximum(mad(b_la - np.median(b_la, axis=0)), FLOOR[1]))
    ref_th, ref_la = ((circ_median(th), np.median(la, axis=0)) if args.ref == "median"
                      else (circ_median(b_th), np.median(b_la, axis=0)))
    dth, dla = wrap(th - ref_th), la - ref_la
    # empty-room check: the baseline against its own median
    b_dth, b_dla = wrap(b_th - circ_median(b_th)), b_la - np.median(b_la, axis=0)

    # layout and grid (rotated for display; distances do not change)
    ref, markers = tag_positions(DATA_DIR / args.exp)
    rot = upright_angle(ref) if args.rotate is None else args.rotate
    ref, markers = rotate_layout(ref, markers, rot)
    xy = {int(t): ref.loc[t, ["x", "y"]].to_numpy(float) for t in ref.index}
    if args.max_link_length is not None:
        # drop links longer than max_link_length (m) from the experiment and the baseline
        keep = np.array([np.linalg.norm(xy[a] - xy[b]) <= args.max_link_length for a, b in links])
        assert keep.any(), f"no link is {args.max_link_length} m or shorter"
        print(f"  max link length {args.max_link_length:g} m: {keep.sum()} / {len(links)} links kept")
        links = [l for l, k in zip(links, keep) if k]
        dth, dla, b_dth, b_dla, ready, first = (x[:, keep] for x in (dth, dla, b_dth, b_dla, ready, first))
        noise = tuple(n[keep] for n in noise)
    xs = np.arange(ref["x"].min() - PAD, ref["x"].max() + PAD + STEP / 2, STEP)
    ys = np.arange(ref["y"].min() - PAD, ref["y"].max() + PAD + STEP / 2, STEP)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    dpath = excess_path(np.array([xy[a] for a, _ in links]), np.array([xy[b] for _, b in links]), pts)

    run_time = ready.max(axis=1)
    order = np.argsort(run_time)
    print(f"{args.exp}: {len(links)} links, {len(run_time)} runs, grid {gx.shape[1]}×{gx.shape[0]} cells of "
          f"{STEP} m; reference = {args.ref}; eps {args.eps}, beta {args.beta}")
    models = {}
    for use in MODELS:
        p = dict(kappa=args.kappa, p_in=args.p_in, p_out=args.p_out, sd_phase_d=args.sd_phase_d,
                 sd_loga_d=args.sd_loga_d)
        if args.fit:
            p = fit(dth, dla, noise, dpath, args.eps, args.beta, use)
        le, ld = link_logs(dth, dla, noise, p["sd_phase_d"], p["sd_loga_d"], args.eps, use)
        ll, l0 = log_maps(le, ld, dpath, p["kappa"], p["p_in"], p["p_out"], args.beta)
        b_le, b_ld = link_logs(b_dth, b_dla, noise, p["sd_phase_d"], p["sd_loga_d"], args.eps, use)
        b_ll, b_l0 = log_maps(b_le, b_ld, dpath, p["kappa"], p["p_in"], p["p_out"], args.beta)
        ll, l0 = ll[order], l0[order]
        post = np.exp(ll - logsumexp(ll, axis=1, keepdims=True)).reshape(-1, *gx.shape)
        pres, b_pres = present_prob(ll, l0), present_prob(b_ll, b_l0)
        spread = np.array([moments(q, gx, gy)[2] for q in post])
        models[use] = dict(post=post, pres=pres, spread=spread, log_lr=(ld - le)[order], params=p)
        print(f"  {MODEL_LABEL[use]:>19}: P(present) median {np.median(pres):.3f} (min {pres.min():.3f}), "
              f"{BASELINE} median {np.median(b_pres):.3f} (max {b_pres.max():.3f}); "
              f"map spread median {np.median(spread):.2f} m" + (f"; params {p}" if args.fit else ""))
    return dict(exp=args.exp, links=links, ref=ref, markers=markers, xy=xy, xs=xs, ys=ys, gx=gx, gy=gy,
                run_time=run_time[order], run_start=first.min(axis=1)[order], ready=ready[order], models=models)


def save_table(res, path):
    rows = []
    for k, t in enumerate(res["run_time"]):
        r = {"run": k, "time (s)": t}
        for use, m in res["models"].items():
            q = m["post"][k]
            i = np.unravel_index(q.argmax(), q.shape)
            mx, my, sd = moments(q, res["gx"], res["gy"])
            r.update({f"{use} P(present)": m["pres"][k], f"{use} peak x": res["xs"][i[1]],
                      f"{use} peak y": res["ys"][i[0]], f"{use} mean x": mx, f"{use} mean y": my,
                      f"{use} sd (m)": sd})
        rows.append(r)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4f")
    print(f"saved {path}")


# ---------------------------------------------------------------- video

def draw_layout(ax, res, links_style):
    # floor labels, tags, and links (faint and fixed on the maps; coloured on the link panel)
    ref, xy = res["ref"], res["xy"]
    ax.scatter(res["markers"]["x"], res["markers"]["y"], s=4, c="#c4c4c4", zorder=1)
    lines = [ax.plot(*np.array([xy[a], xy[b]]).T, zorder=2, **links_style)[0] for a, b in res["links"]]
    for tag, r in ref.iterrows():
        ax.scatter(r.x, r.y, s=170, color="#2a3f5f", edgecolors="white", linewidths=1.2, zorder=4)
        ax.annotate(tag, (r.x, r.y), ha="center", va="center", fontsize=7.5, color="white", fontweight="bold",
                    zorder=5)
    xs, ys = res["xs"], res["ys"]
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#dddddd")
    return lines


def make_video(res, args, out):
    t0, t1 = args.start, (np.inf if args.duration is None else args.start + args.duration)
    run_time, ready, models = res["run_time"], res["ready"], res["models"]
    log_lr = models[LINK_MODEL]["log_lr"]
    lr_norm = Normalize(0, np.log(1 / args.eps), clip=True)
    cmap = plt.get_cmap("Oranges")

    start = video_start(DATA_DIR / res["exp"])
    movs = sorted((DATA_DIR / res["exp"]).glob("*.mov"))
    assert start is not None and movs, "need a .mov and vid_start_time.txt in the experiment folder"
    # one camera window per run: from the run's first MPP start - delta to its last MPP stop + delta
    runs = np.flatnonzero((res["run_time"] >= t0) & (res["run_start"] <= t1))
    assert len(runs), f"no run between {t0} and {t1} s"
    windows, prev_end = [], -np.inf
    for k in runs:
        a = max(res["run_start"][k] - args.delta, prev_end, -start)       # never before the recording starts
        b = res["run_time"][k] + args.delta
        windows.append((k, a, b))
        prev_end = b
    cam0 = windows[0][1]
    camera = CameraReader(movs[0], start + cam0)

    fig = plt.figure(figsize=(19, 10.6))
    cam_ax = fig.add_axes([0.01, 0.595, 0.47, 0.345])
    link_ax = fig.add_axes([0.50, 0.595, 0.24, 0.345])
    cax = fig.add_axes([0.705, 0.62, 0.007, 0.29])
    map_axes = {use: fig.add_axes([0.02 + 0.33 * i, 0.13, 0.30, 0.415]) for i, use in enumerate(MODELS)}
    tax = fig.add_axes([0.05, 0.05, 0.92, 0.05])
    cam_ax.set_xticks([])
    cam_ax.set_yticks([])
    cam_img = cam_ax.imshow(camera.frame(0.0))
    cam_title = cam_ax.set_title("", fontsize=10)

    link_lines = draw_layout(link_ax, res, dict(color="#cccccc", lw=0.6))
    link_title = link_ax.set_title("", fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(norm=lr_norm, cmap=cmap), cax=cax,
                 label=f"log LR, disturbed / empty\n({MODEL_LABEL[LINK_MODEL]})")

    panels = {}
    xs, ys, step = res["xs"], res["ys"], STEP
    for use, ax in map_axes.items():
        draw_layout(ax, res, dict(color="#e2e2e2", lw=0.5))
        heat = ax.imshow(np.zeros(res["gx"].shape), origin="lower", zorder=0.5, interpolation="bilinear",
                         extent=(xs[0] - step / 2, xs[-1] + step / 2, ys[0] - step / 2, ys[-1] + step / 2),
                         cmap=LinearSegmentedColormap.from_list("p", [(1, 1, 1, 0), MODEL_COLOR[use]]),
                         vmin=0, vmax=1)
        star = ax.scatter([np.nan], [np.nan], marker="*", s=280, color=MODEL_COLOR[use], edgecolors="white",
                          linewidths=1.2, zorder=6)
        panels[use] = dict(ax=ax, heat=heat, star=star, contours=[])

    p = models["both"]["params"]
    fig.suptitle(f"{res['exp']}: probability of the person's position, every run on its own (no memory between "
                 f"runs)   kappa {p['kappa']:g} m, p_in {p['p_in']:g}, p_out {p['p_out']:g}, disturbed sd "
                 f"{p['sd_phase_d']:g}° / {p['sd_loga_d']:g} log A, beta {args.beta:g}\n"
                 f"maps: shade = probability (scaled to its peak), contours = 50% / 90% regions, star = peak;  "
                 f"top right: links coloured by how disturbed they are;  camera: run start - {args.delta:g} s to "
                 f"run end + {args.delta:g} s of every run, played at {args.speed:g}× (gaps between runs skipped)",
                 fontsize=10, y=0.995)

    sel = (run_time >= t0 - 3) & (run_time <= t1 + 3)
    for use in MODELS:
        tax.step(run_time[sel], models[use]["pres"][sel], where="post", color=MODEL_COLOR[use], lw=1.3,
                 label=f"P(present), {MODEL_LABEL[use]}")
    tax.set_ylim(-0.02, 1.05)
    tax.set_xlim(t0, min(t1, windows[-1][2]))
    tax.set_xlabel("time from the first MPP (s)")
    tax.set_ylabel("P(present)")
    cursor = tax.axvline(t0, color="#f2a900", lw=1.5)
    span = tax.axvspan(t0, t0, color="#f2a900", alpha=0.25, lw=0)
    tax.legend(fontsize=8, loc="lower right", frameon=False, ncol=3)
    tax.grid(alpha=0.3)

    frames = [(k, a, b, t) for k, a, b in windows for t in np.arange(a, b, args.speed / args.fps)]
    writer = FFMpegWriter(fps=args.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    last_k = None
    print(f"  {len(windows)} runs, {len(frames)} frames ({len(frames) / args.fps:.1f} s of video)")
    with writer.saving(fig, str(out), dpi=args.dpi):
        for f, (k, a, b, t) in enumerate(frames):
            # link panel: every link shows its latest completed measurement at camera time t
            for i, line in enumerate(link_lines):
                j = np.searchsorted(ready[:, i], t, side="right") - 1
                if j < 0:
                    continue
                d = lr_norm(log_lr[j, i])
                line.set_color(cmap(0.15 + 0.85 * d) if d > 0 else "#d8d8d8")
                line.set_linewidth(0.5 + 3 * d)
                line.set_zorder(2 + d)
            link_title.set_text(f"links, latest measurement of each (t = {t:5.2f} s)")
            if k != last_k:
                for use, pn in panels.items():
                    q = models[use]["post"][k]
                    pn["heat"].set_data(q / q.max() * 0.85)
                    for c in pn["contours"]:
                        c.remove()
                    pn["contours"] = [pn["ax"].contour(res["gx"], res["gy"], q, levels=hpd_levels(q),
                                                       colors=[MODEL_COLOR[use]], linewidths=[0.8, 1.6],
                                                       linestyles=["--", "-"], zorder=5)]
                    pn["star"].set_offsets([[res["gx"].ravel()[q.argmax()], res["gy"].ravel()[q.argmax()]]])
                    pn["ax"].set_title(f"{MODEL_LABEL[use]}: run {k + 1} "
                                       f"({res['run_start'][k]:.2f}–{run_time[k]:.2f} s)\nspread "
                                       f"{models[use]['spread'][k]:.2f} m, P(present) {models[use]['pres'][k]:.2f}",
                                       fontsize=10, color=MODEL_COLOR[use])
                span.set_xy([[a, 0], [a, 1], [b, 1], [b, 0], [a, 0]])
                last_k = k
            inside = res["run_start"][k] <= t <= run_time[k]
            cam_img.set_data(camera.frame(t - cam0))
            where = ("DURING the run" if inside else
                     f"{res['run_start'][k] - t:.2f} s before the run" if t < res["run_start"][k] else
                     f"{t - run_time[k]:.2f} s after the run")
            cam_title.set_text(f"camera, t = {t:6.2f} s, run {k + 1}: {where}")
            cam_title.set_color("#c0392b" if inside else "black")
            cursor.set_xdata([t, t])
            writer.grab_frame()
            if f % (args.fps * 10) == 0:
                print(f"  frame {f} / {len(frames)}  (run {k + 1}, t = {t:.1f} s)", flush=True)
    plt.close(fig)
    camera.close()
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", nargs="?", default="1_915")
    ap.add_argument("--ref", choices=["median", "baseline"], default="median",
                    help="per-link reference: median over the experiment's runs, or the empty room (default median)")
    ap.add_argument("--kappa", type=float, default=0.15, help="ellipse width in excess path (m)")
    ap.add_argument("--p-in", type=float, default=0.9, help="P(link disturbed | person on the link)")
    ap.add_argument("--p-out", type=float, default=0.2, help="P(link disturbed | person far from the link)")
    ap.add_argument("--sd-phase-d", type=float, default=25, help="extra phase std when disturbed (deg)")
    ap.add_argument("--sd-loga-d", type=float, default=0.3, help="extra log-attenuation std when disturbed")
    ap.add_argument("--eps", type=float, default=0.05, help="glitch share of the empty distribution")
    ap.add_argument("--beta", type=float, default=0.5, help="likelihood temper (<1: links are not independent)")
    ap.add_argument("--fit", action="store_true",
                    help="fit kappa, p_in, p_out, sd_*_d by maximum evidence, separately for every model")
    ap.add_argument("--start", type=float, default=0.0, help="first run time to show (s from the first MPP)")
    ap.add_argument("--duration", type=float, default=None,
                    help="experiment time span to show (s, default: all runs)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=0.5, help="camera playback speed (default 0.5 = half speed)")
    ap.add_argument("--delta", type=float, default=0.5,
                    help="camera window around every run: run start - delta to run end + delta (s, default 0.5)")
    ap.add_argument("--dpi", type=int, default=90)
    ap.add_argument("--rotate", type=float, default=None, help="layout rotation (deg; default T1 above T4)")
    ap.add_argument("--max-link-length", type=float, default=None,
                    help="ignore links longer than this (m) in every calculation (default: all links)")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    res = analyse(args)
    name = f"{args.exp}_likelihood_map" + ("" if args.max_link_length is None else f"_maxlen{args.max_link_length:g}m")
    if args.ref != "median":
        name += f"_ref{args.ref}"
    save_table(res, HERE / f"{name}.csv")
    if not args.no_video:
        span = "all" if args.duration is None else f"{args.start:g}-{args.start + args.duration:g}s"
        make_video(res, args, HERE / f"{name}_{span}.mp4")


if __name__ == "__main__":
    main()
