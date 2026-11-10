"""Predict a simulated link's distance from the closest parallel wall from its phase-fit residuals (and attenuation).

The simulated counterpart of roomsweeps_wall_distance_fit.py. Same links, target, split, models and frequency band
(--freq-min … --freq-max, default 775–955 MHz) as gridsim_wall_distance.py (which this imports), but the phase features
are replaced by a linear phase fit of the same unwrapped θ̂, computed here exactly as roomsweeps_wall_distance_fit.py
does (not the 800–960 MHz delta_* columns of grid_link_dataset.py):

- Δ per frequency: unwrapped θ̂ minus its least-squares line over frequency (rad)
- the sine fit to Δ: amplitude A (rad) and period (MHz)

Features: Δ, A, period and α̂ per frequency (unless --no-attn); never the link direction or length. The transformer's
frequency tokens are (Δ, α̂) per frequency; A and period go into its read-out token.

    python gridsim_wall_distance_fit.py                       # links 69–115 cm long (≈ the real 92 cm links)
    python gridsim_wall_distance_fit.py --length-cm 46,92,138 # links 23–161 cm long in one dataset
    python gridsim_wall_distance_fit.py --exc 0               # one exciter per simulation
    python gridsim_wall_distance_fit.py --no-attn             # phase-fit features only
    python gridsim_wall_distance_fit.py --kfold 5             # 5-fold cross-validation over simulations
    python gridsim_wall_distance_fit.py --no-transformer      # SVR and MLP only
    python gridsim_wall_distance_fit.py --max-wall-cm 200     # only links within 200 cm of their closest parallel wall
"""
import argparse
from pathlib import Path

import numpy as np

from gridsim_wall_distance import (common_args, evaluate, freq_cols, link_features, lengths_label, load_links,
                                   output_tag, save_outputs, transformer_models)
from roomsweeps_wall_distance import ROOT

OUT_DIR = Path(__file__).resolve().parent / "gridsim_wall_distance_fit_out"


def phase_fit_rows(theta, freqs, n_periods=400, max_cond=100, chunk=4096):
    """roomsweeps_wall_distance_fit.phase_fit for every row of theta (links × freqs) at once: Δ, the line's standard
    error, and the sine fit (A, P) over the same trial periods, skipping the same ill-conditioned ones."""
    line = np.column_stack([freqs, np.ones_like(freqs)])
    delta = theta - (line @ np.linalg.lstsq(line, theta.T, rcond=None)[0]).T
    std_err = np.sqrt(np.sum(delta ** 2, axis=1) / (len(freqs) - 2))
    periods, Ms = [], []
    for P in np.linspace(2 * np.min(np.diff(freqs)), 2 * np.ptp(freqs), n_periods):
        w = 2 * np.pi * freqs / P
        M = np.column_stack([np.sin(w), np.cos(w), np.ones_like(freqs)])
        if np.linalg.cond(M) <= max_cond:
            periods.append(P)
            Ms.append(M)
    periods, Ms = np.array(periods), np.array(Ms)                     # (Q,), (Q, n, 3)
    pinv = np.linalg.pinv(Ms)                                         # (Q, 3, n)
    A, period = np.empty(len(theta)), np.empty(len(theta))
    for i in range(0, len(theta), chunk):
        d = delta[i:i + chunk]
        c = np.einsum("qkn,ln->lqk", pinv, d)                         # (L, Q, 3): sin, cos, const
        sse = np.sum((np.einsum("qnk,lqk->lqn", Ms, c) - d[:, None]) ** 2, axis=2)
        best = np.argmin(sse, axis=1)
        cb = c[np.arange(len(d)), best]
        A[i:i + chunk], period[i:i + chunk] = np.hypot(cb[:, 0], cb[:, 1]), periods[best]
    return delta, std_err, A, period


def build_dataset(links, freq_range, use_attn=True):
    """Δ per frequency, sine A, sine period of θ̂ unwrapped over freq_range, α̂ per frequency (if use_attn). The meta
    columns std_err, sine_A, sine_period_mhz are replaced by this fit."""
    th_cols, freqs = freq_cols(links, "theta", freq_range)
    delta, std_err, A, P = phase_fit_rows(np.unwrap(links[th_cols].to_numpy(), period=np.pi, axis=1), freqs)
    links = links.assign(std_err=std_err, sine_A=A, sine_period_mhz=P)
    extra = [delta, np.column_stack([A, P])]
    if use_attn:
        extra.append(links[freq_cols(links, "alpha", freq_range)[0]].to_numpy())
    return *link_features(links, extra), freqs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap)
    ap.add_argument("--attn", action=argparse.BooleanOptionalAction, default=True,
                    help="use α̂ as features alongside the phase fit (--no-attn: phase fit only)")
    args = ap.parse_args()

    lengths, links = load_links(args)
    X, y, meta, freqs = build_dataset(links, (args.freq_min, args.freq_max), args.attn)
    print(f"{len(X)} samples (one per link and exciter), {X.shape[1]} features: Δ + sine A, period{' + α̂' if args.attn else ''} at "
          f"{len(freqs)} freqs {freqs.min():g}–{freqs.max():g} MHz")

    # frequency tokens: Δ in columns 0…F-1, α̂ in F+2…2F+1 (after sine A, period)
    token_cols = np.arange(len(freqs))[None] + np.array([0, len(freqs) + 2][:2 if args.attn else 1])[:, None]
    te, preds = evaluate(X, y, meta, args.train_frac, args.seed, args.max_train, args.kfold,
                         transformer_models(args, token_cols))

    tag = output_tag(args) + ("" if args.attn else "_fit_only")
    save_outputs(OUT_DIR, tag, y, te, preds, meta,
                 f"simulated {lengths_label(lengths)} links, Δ + sine fit{' + α̂' if args.attn else ' only'}: "
                 "distance from closest parallel wall (test set)")


if __name__ == "__main__":
    main()
