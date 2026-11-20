"""Figures for link_likelihood_map_algorithm.md (written to link_likelihood_map_doc/).

    python link_likelihood_map_doc_figures.py [exp] [run]
"""

import sys
from types import SimpleNamespace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from link_likelihood_map import (BASELINE, FLOOR, HERE, MODEL_COLOR, MODEL_LABEL, MODELS, analyse, circ_median,
                                 excess_path, link_arrays, log_gauss, log_vm_half, mad, wrap)

OUT = HERE / "link_likelihood_map_doc"
TAG = "#2a3f5f"


def args_for(exp):
    return SimpleNamespace(exp=exp, ref="median", kappa=0.15, p_in=0.9, p_out=0.2, sd_phase_d=25, sd_loga_d=0.3,
                           eps=0.05, beta=0.5, fit=False, rotate=None)


def tags(ax, res, links=True, highlight=()):
    xy = res["xy"]
    if links:
        for a, b in res["links"]:
            ax.plot(*np.array([xy[a], xy[b]]).T, color="#e0e0e0", lw=0.5, zorder=2)
    for a, b in highlight:
        ax.plot(*np.array([xy[a], xy[b]]).T, color="#222222", lw=2.2, zorder=3)
    for t, p in xy.items():
        ax.scatter(*p, s=150, color=TAG, edgecolors="white", linewidths=1.2, zorder=4)
        ax.annotate(str(t), p, ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=5)
    ax.set_xlim(res["xs"][0], res["xs"][-1])
    ax.set_ylim(res["ys"][0], res["ys"][-1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def extent(res):
    s = res["xs"][1] - res["xs"][0]
    return (res["xs"][0] - s / 2, res["xs"][-1] + s / 2, res["ys"][0] - s / 2, res["ys"][-1] + s / 2)


def grid_dpath(res):
    pts = np.stack([res["gx"].ravel(), res["gy"].ravel()], axis=1)
    xy = res["xy"]
    return excess_path(np.array([xy[a] for a, _ in res["links"]]), np.array([xy[b] for _, b in res["links"]]), pts)


def fig_ellipse(res, p):
    # one link: excess path contours and P(disturbed | person at x)
    i = res["links"].index((2, 8)) if (2, 8) in res["links"] else 0
    d = grid_dpath(res)[:, i].reshape(res["gx"].shape)
    pi = p["p_out"] + (p["p_in"] - p["p_out"]) * np.exp(-d ** 2 / (2 * p["kappa"] ** 2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    ax = axes[0]
    cs = ax.contour(res["gx"], res["gy"], d, levels=[0.05, 0.15, 0.3, 0.5, 0.8], colors="#555555", linewidths=0.8)
    ax.clabel(cs, fmt=lambda v: f"{v * 100:.0f} cm", fontsize=7)
    tags(ax, res, links=False, highlight=[res["links"][i]])
    ax.set_title("extra path length via a point\n(ellipses around the link)", fontsize=10)
    ax = axes[1]
    im = ax.imshow(pi, origin="lower", extent=extent(res), cmap="Oranges", vmin=0, vmax=1, zorder=0.5)
    tags(ax, res, links=False, highlight=[res["links"][i]])
    fig.colorbar(im, ax=ax, fraction=0.04, label="P(link disturbed | person here)")
    ax.set_title(f"chance the link is disturbed if you stand there\n(p_in {p['p_in']:g} on the link, "
                 f"p_out {p['p_out']:g} far away, kappa {p['kappa']:g} m)", fontsize=10)
    a, b = res["links"][i]
    fig.suptitle(f"One link: tag {a} ↔ tag {b}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "1_link_ellipse.png", dpi=120)
    plt.close(fig)


def fig_distributions(p, eps):
    # typical link: empty vs disturbed density for phase and log attenuation
    _, b_th, b_la, _, _ = link_arrays(BASELINE)
    s_th = float(np.median(np.maximum(mad(wrap(b_th - circ_median(b_th))), FLOOR[0])))
    s_la = float(np.median(np.maximum(mad(b_la - np.median(b_la, axis=0)), FLOOR[1])))
    sd_th = np.sqrt(s_th ** 2 + np.deg2rad(p["sd_phase_d"]) ** 2)
    sd_la = np.sqrt(s_la ** 2 + p["sd_loga_d"] ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    d = np.linspace(-np.pi / 2, np.pi / 2, 2000)
    fe = (1 - eps) * np.exp(log_vm_half(d, s_th)) + eps * np.exp(log_vm_half(d, sd_th))
    fd = np.exp(log_vm_half(d, sd_th))
    for ax, x, e, dd, xl, unit in ((axes[0], np.rad2deg(d), fe * np.pi / 180, fd * np.pi / 180,
                                    "phase change from the link's usual value (deg)", "per deg"),):
        ax.plot(x, e, color="#2a78d6", lw=1.8, label=f"empty: room noise (sd {np.rad2deg(s_th):.1f}°)")
        ax.plot(x, dd, color="#c0392b", lw=1.8, label=f"disturbed: person near the link")
        ax.set_xlabel(xl)
        ax.set_ylabel(f"probability density ({unit})")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, None)
    v = np.linspace(-1.2, 1.2, 2000)
    fe = (1 - eps) * np.exp(log_gauss(v, s_la)) + eps * np.exp(log_gauss(v, sd_la))
    fd = np.exp(log_gauss(v, sd_la))
    ax = axes[1]
    ax.plot(v, fe, color="#2a78d6", lw=1.8, label=f"empty: room noise (sd {s_la:.2f})")
    ax.plot(v, fd, color="#c0392b", lw=1.8, label="disturbed: person near the link")
    ax.set_xlabel("change of log attenuation from the link's usual value")
    ax.set_ylabel("probability density")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, None)
    for ax in axes:
        ax.legend(fontsize=8, frameon=False, loc="lower center")
        ax.grid(alpha=0.3)
    fig.suptitle("A typical link: what its change looks like when nobody is near it vs when someone is "
                 "(log scale; the narrow peak is 'nothing happened')", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "2_empty_vs_disturbed.png", dpi=120)
    plt.close(fig)


def fig_votes(res, p, run):
    # one loud link, one quiet link, and the total of all links (phase + attenuation)
    m = res["models"]["both"]
    lr = np.exp(np.clip(m["log_lr"][run], -50, 50))
    dpath = grid_dpath(res)
    pi = p["p_out"] + (p["p_in"] - p["p_out"]) * np.exp(-dpath ** 2 / (2 * p["kappa"] ** 2))
    # beta = 0.5, cells × links; relative to standing far from the link (pi = p_out), which is the same constant
    # for every cell and drops out when the map is normalised
    contrib = 0.5 * (np.log1p(pi * (lr[None] - 1)) - np.log1p(p["p_out"] * (lr[None] - 1)))
    loud, quiet = int(np.argmax(m["log_lr"][run])), int(np.argmin(m["log_lr"][run]))
    shape = res["gx"].shape
    lim = max(np.abs(contrib[:, [loud, quiet]]).max(), 1e-3)
    norm = TwoSlopeNorm(0, -lim, lim)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6))
    for ax, i, word in ((axes[0], loud, "changed a lot"), (axes[1], quiet, "stayed quiet")):
        im = ax.imshow(contrib[:, i].reshape(shape), origin="lower", extent=extent(res), cmap="RdBu_r", norm=norm,
                       zorder=0.5)
        tags(ax, res, links=False, highlight=[res["links"][i]])
        a, b = res["links"][i]
        ax.set_title(f"link {a}–{b} {word}\n(log LR {m['log_lr'][run, i]:+.1f}): its vote", fontsize=10)
    fig.colorbar(im, ax=axes[:2], fraction=0.03, label="vote (log; red = more likely)")
    q = m["post"][run]
    ax = axes[2]
    ax.imshow(q / q.max(), origin="lower", extent=extent(res), zorder=0.5,
              cmap=LinearSegmentedColormap.from_list("p", [(1, 1, 1, 0), MODEL_COLOR["both"]]), vmin=0, vmax=1)
    tags(ax, res)
    j = q.argmax()
    ax.scatter(res["gx"].ravel()[j], res["gy"].ravel()[j], marker="*", s=300, color=MODEL_COLOR["both"],
               edgecolors="white", zorder=6)
    ax.set_title(f"all {len(res['links'])} votes added → probability map\n(run {run + 1}, spread "
                 f"{m['spread'][run]:.2f} m)", fontsize=10)
    fig.suptitle(f"{res['exp']}, run {run + 1}: every link votes on every floor cell", fontsize=12)
    fig.savefig(OUT / "3_link_votes.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_models(res, run):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.4))
    for ax, use in zip(axes, MODELS):
        m = res["models"][use]
        q = m["post"][run]
        ax.imshow(q / q.max(), origin="lower", extent=extent(res), zorder=0.5,
                  cmap=LinearSegmentedColormap.from_list("p", [(1, 1, 1, 0), MODEL_COLOR[use]]), vmin=0, vmax=1)
        tags(ax, res)
        j = q.argmax()
        ax.scatter(res["gx"].ravel()[j], res["gy"].ravel()[j], marker="*", s=300, color=MODEL_COLOR[use],
                   edgecolors="white", zorder=6)
        ax.set_title(f"{MODEL_LABEL[use]}\nspread {m['spread'][run]:.2f} m, P(present) {m['pres'][run]:.2f}",
                     fontsize=10, color=MODEL_COLOR[use])
    fig.suptitle(f"{res['exp']}, run {run + 1}: the same run with phase only, attenuation only, and both",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "4_three_models.png", dpi=120)
    plt.close(fig)


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "1_915"
    run = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 7
    OUT.mkdir(exist_ok=True)
    a = args_for(exp)
    res = analyse(a)
    p = res["models"]["both"]["params"]
    fig_ellipse(res, p)
    fig_distributions(p, a.eps)
    fig_votes(res, p, run)
    fig_models(res, run)
    print(f"saved figures in {OUT}")


if __name__ == "__main__":
    main()
