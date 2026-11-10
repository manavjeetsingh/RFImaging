"""Post-train a transformer saved by `gridsim_wall_distance.py --save-model` on the real room-sweep links, next to a
residual CNN pretrained on the same simulated links.

Same data, target, splits and outputs as roomsweeps_wall_distance.py (whose functions it uses): every tag pair whose labels
are --length-cm apart along a grid direction, target = distance of the link's midpoint from the closest parallel wall,
one sample per link and experiment (median over runs), split 70 / 30 by label pair at random, or with --loo leave one
experiment out (the other experiments' copies of its label pairs dropped from training; --no-loo-dedup: kept). The features are built exactly as the saved model was trained on the simulation: the frequency band,
α̂ on/off, --zero-start and --unwrap are read from the model file (the real frequencies must be the same as the sim's),
and --length-cm defaults to the model's.

For each split it compares, on the held-out samples:

- mean of train, SVR, MLP              as in roomsweeps_wall_distance.py, trained on the real training samples only
- Transformer (pretrained)             the saved sim model as is (zero-shot)
- Transformer (post-trained)           a copy of it trained further on the real training samples (--ft-*; every
                                       weight, or only the regression head with --freeze-encoder; the sim input / target
                                       standardization is kept unless --renorm)
- Transformer (scratch)                the same architecture trained on the real training samples only (--no-scratch: off)
- ResCNN (pretrained)                  the residual CNN below, trained on the simulated links the transformer was trained
                                       on (the model file's sim CSV, link lengths, exciters and --max-wall-cm), zero-shot
- ResCNN (post-trained)                a copy of it trained further on the real training samples (--rescnn-ft-*), each
                                       step on a real batch plus an augmented sim batch so it keeps the sim prior
                                       (--no-rescnn-joint: real batches only); --freeze-encoder and --renorm as above
- ResCNN (scratch)                     the same CNN trained on the real training samples only (--no-scratch: off)
(--no-rescnn: no ResCNN rows)

The ResCNN reads one token per frequency, [Δ, centred log α̂]: Δ is θ̂ (unwrapped with period π) minus its straight-line
fit over the band, centred log α̂ is log α̂ minus its band mean; the line's slope and the mean log α̂ are two per-link
globals (phase only: Δ and the slope). The tokens go through a 1-D convolution and 4 residual blocks (batch norm, GELU,
dilated 3-tap convolutions, dilations 1, 2, 4, 1, width 64); the mean and the max over frequency of the last feature
map, with the globals, feed a 2-layer regression head. No direction, length or location input. Inputs are standardized
on the (augmented) sim rows, the target on the sim targets; trained with AdamW, a one-cycle learning rate and the Huber
loss. During sim pretraining every batch is perturbed to look like the real sweeps (--rescnn-aug, on by default):
K = 8 random extra paths multiplying the response, 1 + Σ a_k exp(−j(2πfτ_k + φ_k)) with rms amplitude 0.2–0.6 and a
6–15 ns exponential delay profile; the ≈ 2.78 ns hardware delay that makes the sim phase slope match the real one;
α̂ × 0.47 with a random per-link gain; a smooth phase-only wiggle up to 12°; 1.5° / 2 % noise. The pretrained CNN is
cached in roomsweeps_wall_distance_posttrain_out/ (--rescnn-retrain: train it again).

    python roomsweeps_wall_distance_posttrain.py --model ../gridsim_processing/gridsim_wall_distance_out/transformer_92cm.pt
    python roomsweeps_wall_distance_posttrain.py --model … --loo              # leave one experiment out
    python roomsweeps_wall_distance_posttrain.py --model … --freeze-encoder   # train the regression heads only
    python roomsweeps_wall_distance_posttrain.py --model … --renorm           # standardize on the real training rows
    python roomsweeps_wall_distance_posttrain.py --model … --ft-epochs 500 --ft-lr 3e-4
    python roomsweeps_wall_distance_posttrain.py --model … --no-rescnn        # transformers only
    python roomsweeps_wall_distance_posttrain.py --model … --rescnn-ft-epochs 100 --no-rescnn-joint
"""
import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gridsim_processing"))   # gridsim_wall_distance

from gridsim_wall_distance import DATA_PATH, TransformerRegressor, load_sim, select_links
from gridsim_wall_distance import build_dataset as build_sim_dataset
from roomsweeps_wall_distance import (DATA_DIR, N_COLS, N_ROWS, ROOT, TILE_CM, all_positions, build_dataset, evaluate,
                                      find_walls, fit_predict, grid_xy, load_bidir, parallel_links, plot_predictions,
                                      wall_distance_cm)

OUT_DIR = Path(__file__).resolve().parent / "roomsweeps_wall_distance_posttrain_out"


# ---------------------------------------------------------------- residual CNN

def unwrap_pi(th):
    """θ̂ (N, F) unwrapped along frequency with period π (leaves an already unwrapped θ̂ unchanged)."""
    d = torch.diff(th, dim=1)
    d = d - math.pi * torch.round(d / math.pi)
    return torch.cat([th[:, :1], th[:, :1] + torch.cumsum(d, 1)], 1)


def augment(th, al, f_mhz, gen, clutter=(0.2, 0.6), tau0_ns=(6.0, 15.0), k=8, tau_hw_ns=2.78, tau_hw_sd=0.15,
            gain=0.47, gain_sd=0.25, noise_deg=1.5, noise_rel=0.02, wiggle_deg=(0.0, 12.0)):
    """Randomly perturbed copies of simulated (θ̂, α̂) (N, F) tensors (α̂ may be None) that look like the real sweeps:
    the response times 1 + Σ_k a_k exp(−j(2πfτ_k + φ_k)), k extra paths with per-link rms amplitude in clutter and an
    exponential delay profile (time constant in tau0_ns); a hardware delay tau_hw_ns ± tau_hw_sd on the phase; α̂ times
    gain · exp(gain_sd · N(0, 1)); a smooth phase-only wiggle (random 2nd–4th order polynomial over the band, rms up to
    wiggle_deg); and noise_deg / noise_rel measurement noise. Real arithmetic only."""
    n, F = th.shape
    dev = th.device
    fz = torch.as_tensor(np.asarray(f_mhz, float) * 1e6, dtype=torch.float64, device=dev)
    U = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(*s, generator=gen, device=dev, dtype=torch.float64)
    N = lambda *s: torch.randn(*s, generator=gen, device=dev, dtype=torch.float64)
    t0 = U(*tau0_ns, n, 1) * 1e-9
    tau = U(0.0, 1.0, n, k) * 4 * t0 + 0.3e-9
    w = torch.exp(-tau / t0)
    w = w / w.square().sum(1, keepdim=True).sqrt()
    amp = U(*clutter, n, 1) * w * (-torch.log(U(1e-9, 1.0, n, k))).sqrt()
    ph = -(2 * math.pi * tau[:, :, None] * fz + U(0, 2 * math.pi, n, k)[:, :, None])
    re = 1 + (amp[:, :, None] * torch.cos(ph)).sum(1)
    im = (amp[:, :, None] * torch.sin(ph)).sum(1)
    hw = (tau_hw_ns + tau_hw_sd * N(n, 1)) * 1e-9
    x = (fz - fz.mean()) / (fz.max() - fz.min()) * 2
    wig = sum(N(n, 1) * x ** j for j in (2, 3, 4))
    wig = wig / (wig.std(1, keepdim=True) + 1e-9) * U(*wiggle_deg, n, 1) * math.pi / 180
    th2 = th.double() + torch.atan2(im, re) + 2 * math.pi * hw * (fz - fz[0]) + wig + math.radians(noise_deg) * N(n, F)
    if al is None:
        return th2.float(), None
    al2 = al.double() * torch.hypot(re, im) * gain * torch.exp(gain_sd * N(n, 1)) * (1 + noise_rel * N(n, F))
    return th2.float(), (al2.abs() + 1e-6).float()


class ResCNN(nn.Module):
    """1-D residual CNN over frequency tokens (N, F, n_channels); mean and max pooling over frequency plus the n_global
    per-link features feed the regression head."""

    def __init__(self, n_channels, n_global, width=64, dilations=(1, 2, 4, 1), dropout=0.1):
        super().__init__()
        self.inp = nn.Conv1d(n_channels, width, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.BatchNorm1d(width), nn.GELU(), nn.Conv1d(width, width, 3, padding=d, dilation=d),
            nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(width, width, 3, padding=1))
            for d in dilations])
        self.head = nn.Sequential(nn.Linear(2 * width + n_global, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, tokens, glob):
        x = self.inp(tokens.transpose(1, 2))
        for block in self.blocks:
            x = x + block(x)
        return self.head(torch.cat([x.mean(-1), x.amax(-1), glob], 1)).squeeze(-1)


class ResCNNRegressor:
    """sklearn-style wrapper around ResCNN on the feature rows of build_dataset (θ̂ in columns 0…F-1, α̂ in F…2F-1 if
    attn). pretrain fits it on simulated rows with the realism augmentation (aug: augment keyword arguments, or None
    for none); fine_tune trains a copy further on real rows, optionally mixed with augmented sim batches; fit trains it
    from scratch on real rows. save / load write and read the fitted network and its standardization."""

    NORM_KEYS = ("tok_mu", "tok_sd", "glob_mu", "glob_sd")

    def __init__(self, freqs, attn, aug=None, width=64, dropout=0.1):
        self.freqs, self.attn, self.aug = np.asarray(freqs, float), attn, aug
        self.arch = {"width": width, "dropout": dropout}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        f = torch.as_tensor(self.freqs, dtype=torch.float32, device=self.device)
        self.A = torch.stack([(f - f.mean()) / 100, torch.ones_like(f)], 1)       # straight line over the band
        self.P = torch.linalg.pinv(self.A)

    def tensors(self, X, y=None):
        """(θ̂, α̂ or None, y or None) float32 tensors of feature rows X."""
        F = len(self.freqs)
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)
        return T(X[:, :F]), T(X[:, F:2 * F]) if self.attn else None, None if y is None else T(y)

    def _raw(self, th, al):
        th = unwrap_pi(th)
        c = th @ self.P.T                                    # (N, 2): slope, level of the line fit
        tok, glob = [th - c @ self.A.T], [c[:, :1]]
        if al is not None:
            la = torch.log(al.clamp_min(1e-6))
            lm = la.mean(1, keepdim=True)
            tok.append(la - lm)
            glob.append(lm)
        return torch.stack(tok, -1), torch.cat(glob, 1)

    def _set_norm(self, th, al, y):
        tok, glob = self._raw(th, al)
        self.tok_mu, self.tok_sd = tok.mean((0, 1)), tok.std((0, 1)) + 1e-6
        self.glob_mu, self.glob_sd = glob.mean(0), glob.std(0) + 1e-6
        self.y_mu, self.y_sd = float(y.mean()), float(y.std())

    def _features(self, th, al):
        tok, glob = self._raw(th, al)
        return (tok - self.tok_mu) / self.tok_sd, (glob - self.glob_mu) / self.glob_sd

    def _new_model(self):
        n = 2 if self.attn else 1
        return ResCNN(n, n, **self.arch).to(self.device)

    def _loss(self, th, al, y):
        return nn.functional.huber_loss(self.model(*self._features(th, al)), (y - self.y_mu) / self.y_sd)

    def _train(self, real, sim, epochs, batch_size, lr, weight_decay, gen, frozen=(), sim_batch=256, label="ResCNN",
               log_every=0):
        """real / sim: (θ̂, α̂, y) tensors or None; an epoch is one pass over real (else over sim). Each step uses a real
        batch and / or an augmented sim batch of sim_batch rows (batch_size without real). frozen: modules kept fixed."""
        n = len(real[2]) if real is not None else len(sim[2])
        bs = batch_size if real is not None else sim_batch
        steps = -(-n // bs)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps, pct_start=0.05)
        for epoch in range(epochs):
            self.model.train()
            for m in frozen:
                m.eval()                                    # keep the frozen batch-norm statistics too
            perm, total = torch.randperm(n, device=self.device, generator=gen), 0.0
            for b in perm.split(bs):
                loss = 0.0
                if real is not None:
                    loss = loss + self._loss(real[0][b], None if real[1] is None else real[1][b], real[2][b])
                if sim is not None:
                    s = b if real is None else torch.randint(0, len(sim[2]), (sim_batch,), device=self.device,
                                                             generator=gen)
                    th, al = sim[0][s], None if sim[1] is None else sim[1][s]
                    if self.aug is not None:
                        th, al = augment(th, al, self.freqs, gen, **self.aug)
                    loss = loss + self._loss(th, al, sim[2][s])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * len(b)
            if log_every and (epoch + 1 == epochs or (epoch + 1) % log_every == 0):
                print(f"  {label} epoch {epoch + 1}/{epochs}: train loss {total / n:.4f}")

    def _gen(self, seed):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        return gen

    def pretrain(self, sim, epochs=30, batch_size=512, lr=1e-3, weight_decay=1e-2, seed=0, log_every=10):
        """Fit on simulated (θ̂, α̂, y) tensors, every batch augmented if self.aug; standardized on augmented rows."""
        torch.manual_seed(seed)
        gen = self._gen(seed)
        idx = torch.randperm(len(sim[2]), device=self.device, generator=gen)[:50000]
        th, al = sim[0][idx], None if sim[1] is None else sim[1][idx]
        if self.aug is not None:
            th, al = augment(th, al, self.freqs, gen, **self.aug)
        self._set_norm(th, al, sim[2])
        self.model = self._new_model()
        self._train(None, sim, epochs, None, lr, weight_decay, gen, sim_batch=batch_size, label="ResCNN sim pretraining",
                    log_every=log_every)
        return self

    def fit(self, X, y, epochs=60, batch_size=32, lr=1e-3, weight_decay=1e-2, seed=0):
        """Train from scratch on real rows (standardized on them)."""
        torch.manual_seed(seed)
        real = self.tensors(X, y)
        self._set_norm(*real)
        self.model = self._new_model()
        self._train(real, None, epochs, batch_size, lr, weight_decay, self._gen(seed))
        return self

    def fine_tune(self, X, y, sim=None, epochs=60, batch_size=32, lr=2e-4, weight_decay=1e-2, renorm=False,
                  freeze_encoder=False, seed=0):
        """A copy trained further on real rows (X, y), each step also on an augmented batch of the sim tensors if
        given; self is left unchanged. renorm: standardize on these rows; freeze_encoder: train the head only."""
        ft = copy.deepcopy(self)
        torch.manual_seed(seed)
        real = ft.tensors(X, y)
        if renorm:
            ft._set_norm(*real)
        frozen = (ft.model.inp, ft.model.blocks) if freeze_encoder else ()
        for m in frozen:
            for p in m.parameters():
                p.requires_grad_(False)
        ft._train(real, sim, epochs, batch_size, lr, weight_decay, ft._gen(seed), frozen)
        return ft

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        th, al, _ = self.tensors(X)
        out = [self.model(*self._features(th[i:i + 4096], None if al is None else al[i:i + 4096]))
               for i in range(0, len(th), 4096)]
        return torch.cat(out).cpu().numpy() * self.y_sd + self.y_mu

    def save(self, path, **info):
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                    "arch": self.arch, "freqs": self.freqs.tolist(), "attn": self.attn, "aug": self.aug,
                    "norm": {k: getattr(self, k).detach().cpu() for k in self.NORM_KEYS},
                    "y": torch.tensor([self.y_mu, self.y_sd], dtype=torch.float64), "info": info}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        reg = cls(ck["freqs"], ck["attn"], ck["aug"], **ck["arch"])
        for k in cls.NORM_KEYS:
            setattr(reg, k, ck["norm"][k].to(reg.device))
        reg.y_mu, reg.y_sd = ck["y"].tolist()
        reg.model = reg._new_model()
        reg.model.load_state_dict(ck["state_dict"])
        return reg, ck["info"]


def pretrained_rescnn(args, info):
    """(ResCNN pretrained on the transformer's simulated links, those links as (θ̂, α̂, y) tensors), cached."""
    lengths = [float(x) for x in info["length_cm"].split(",")]
    exc = [int(x) for x in info["exc"].split(",")] if info.get("exc") else None
    data = Path(info.get("data") or DATA_PATH)
    print(f"\nResCNN: loading the simulated links the transformer was trained on ({data.name}) …")
    links = select_links(load_sim(data), lengths, exc, info["max_wall_cm"])
    X, y, _, freqs = build_sim_dataset(links, (info["freq_min"], info["freq_max"]), info["attn"], info["zero_start"],
                                       info["unwrap"])
    if not np.array_equal(freqs, np.asarray(info["freqs"], float)):
        raise SystemExit(f"sim frequencies {list(freqs)} differ from the model's {info['freqs']}")
    aug = {} if args.rescnn_aug else None                       # {}: augment's defaults
    reg = ResCNNRegressor(freqs, info["attn"], aug)
    sim = reg.tensors(X, y)
    path = OUT_DIR / (f"rescnn_{args.model.stem}_ep{args.rescnn_epochs}" + ("" if args.rescnn_aug else "_noaug")
                      + f"_seed{args.seed}.pt")
    if path.exists() and not args.rescnn_retrain:
        reg, _ = ResCNNRegressor.load(path)
        print(f"  loaded the pretrained ResCNN from {path.relative_to(ROOT)}")
    else:
        t0 = time.time()
        print(f"  pretraining on {len(y)} sim rows{' with the realism augmentation' if aug is not None else ''} …")
        reg.pretrain(sim, epochs=args.rescnn_epochs, seed=args.seed)
        OUT_DIR.mkdir(exist_ok=True)
        reg.save(path, transformer=str(args.model), epochs=args.rescnn_epochs, seed=args.seed, sim_rows=len(y))
        print(f"  {time.time() - t0:.0f} s, wrote {path.relative_to(ROOT)}")
    return reg, sim


# ---------------------------------------------------------------- evaluation

def make_fit_fn(pretrained, args, rescnn=None, sim=None):
    """fit_fn for roomsweeps_wall_distance.evaluate: its baselines plus the three transformers (and three ResCNNs)."""
    def fit_fn(X, y, tr, te, seed):
        preds, train_mae = fit_predict(X, y, tr, te, seed)
        tfs = {"Transformer (pretrained)": pretrained,
               "Transformer (post-trained)": pretrained.fine_tune(
                   X[tr], y[tr], epochs=args.ft_epochs, batch_size=args.ft_batch_size, lr=args.ft_lr,
                   weight_decay=args.ft_weight_decay, renorm=args.renorm, freeze_encoder=args.freeze_encoder, seed=seed)}
        if args.scratch:
            tfs["Transformer (scratch)"] = TransformerRegressor(
                pretrained.token_cols, epochs=args.ft_epochs, batch_size=args.ft_batch_size, lr=args.scratch_lr,
                weight_decay=args.ft_weight_decay, seed=seed, log_every=0, **pretrained.arch).fit(X[tr], y[tr])
        if rescnn is not None:
            tfs["ResCNN (pretrained)"] = rescnn
            tfs["ResCNN (post-trained)"] = rescnn.fine_tune(
                X[tr], y[tr], sim if args.rescnn_joint else None, epochs=args.rescnn_ft_epochs,
                batch_size=args.ft_batch_size, lr=args.rescnn_ft_lr, weight_decay=args.ft_weight_decay,
                renorm=args.renorm, freeze_encoder=args.freeze_encoder, seed=seed)
            if args.scratch:
                tfs["ResCNN (scratch)"] = ResCNNRegressor(rescnn.freqs, rescnn.attn, **rescnn.arch).fit(
                    X[tr], y[tr], epochs=args.rescnn_ft_epochs, batch_size=args.ft_batch_size,
                    lr=args.rescnn_scratch_lr, weight_decay=args.ft_weight_decay, seed=seed)
        for name, model in tfs.items():
            preds[name] = model.predict(X[te])
            train_mae[name] = mean_absolute_error(y[tr], model.predict(X[tr]))
        return preds, train_mae
    return fit_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help=".pt file written by gridsim_wall_distance.py --save-model")
    ap.add_argument("--length-cm", default=None,
                    help="link length(s) in cm, comma-separated, each a whole number of tiles (default: the model's)")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-wall-cm", type=float, default=None,
                    help="ignore links whose midpoint is more than this many cm from the closest parallel wall")
    ap.add_argument("--dedup", action="store_true",
                    help="keep each label pair once, from the first experiment that measured it")
    ap.add_argument("--loo", action="store_true",
                    help="leave one experiment out: train on the other experiments, test on it, for every experiment")
    ap.add_argument("--loo-dedup", action=argparse.BooleanOptionalAction, default=True,
                    help="with --loo, drop the training samples of every label pair the held-out experiment measured "
                         "(--no-loo-dedup: keep them)")
    ap.add_argument("--ft-epochs", type=int, default=200, help="post-training (and scratch) epochs")
    ap.add_argument("--ft-batch-size", type=int, default=32)
    ap.add_argument("--ft-lr", type=float, default=1e-4, help="peak post-training learning rate")
    ap.add_argument("--ft-weight-decay", type=float, default=1e-2)
    ap.add_argument("--freeze-encoder", action="store_true", help="post-train the regression heads only")
    ap.add_argument("--renorm", action="store_true",
                    help="re-estimate the input / target standardization on the real training samples")
    ap.add_argument("--scratch", action=argparse.BooleanOptionalAction, default=True,
                    help="also train the transformer (and the ResCNN) from scratch on the real training samples")
    ap.add_argument("--scratch-lr", type=float, default=1e-3)
    ap.add_argument("--rescnn", action=argparse.BooleanOptionalAction, default=True,
                    help="also pretrain the residual CNN on the transformer's sim links and post-train it")
    ap.add_argument("--rescnn-epochs", type=int, default=30, help="ResCNN sim pretraining epochs")
    ap.add_argument("--rescnn-aug", action=argparse.BooleanOptionalAction, default=True,
                    help="perturb the sim batches to look like the real sweeps (clutter, hardware delay, gain, noise)")
    ap.add_argument("--rescnn-retrain", action="store_true", help="pretrain the ResCNN again even if it is cached")
    ap.add_argument("--rescnn-ft-epochs", type=int, default=60, help="ResCNN post-training (and scratch) epochs")
    ap.add_argument("--rescnn-ft-lr", type=float, default=2e-4, help="peak ResCNN post-training learning rate")
    ap.add_argument("--rescnn-scratch-lr", type=float, default=1e-3)
    ap.add_argument("--rescnn-joint", action=argparse.BooleanOptionalAction, default=True,
                    help="post-train the ResCNN on real batches mixed with augmented sim batches "
                         "(--no-rescnn-joint: real batches only)")
    args = ap.parse_args()

    pretrained, info = TransformerRegressor.load(args.model)
    pretrained.log_every = 0
    attn, zero_start, unwrap = info["attn"], info["zero_start"], info["unwrap"]
    freq_range = (info["freq_min"], info["freq_max"])
    sim_max = "" if info["max_wall_cm"] is None else f" within {info['max_wall_cm']:g} cm of a wall"
    print(f"model {args.model}: trained on sim links of {info['length_cm']} cm{sim_max}, "
          f"θ̂{'' if unwrap else ' (wrapped)'}{' (start at 0)' if zero_start else ''}{' + α̂' if attn else ''} at "
          f"{len(info['freqs'])} freqs {freq_range[0]:g}–{freq_range[1]:g} MHz")

    lengths = [float(x) for x in (args.length_cm or info["length_cm"]).split(",")]
    n_tiles = sorted({int(round(l / TILE_CM)) for l in lengths})
    for l in lengths:
        assert round(l / TILE_CM) > 0 and abs(round(l / TILE_CM) * TILE_CM - l) < 1, \
            f"{l:g} cm is not a whole number of {TILE_CM:g} cm tiles"
    lengths_cm = [f"{n * TILE_CM:.0f}" for n in n_tiles]

    positions = all_positions()
    exps = [e for e in positions if any((DATA_DIR / e).glob("*.csv"))]
    print(f"experiments with a CSV: {', '.join(exps)}")
    walls = find_walls()
    print("walls (distance of the nearest grid edge from the wall):")
    for name, w in walls.items():
        edge = grid_xy(0, 0) if "col 0" in name or "row 1" in name else grid_xy(N_COLS - 1, N_ROWS - 1)
        print(f"  {name:<30} {wall_distance_cm(edge, w):6.1f} cm")

    links = parallel_links(positions, exps, n_tiles, walls, args.dedup, args.max_wall_cm)
    if links.empty:
        raise SystemExit("no links match --length-cm / --max-wall-cm")
    print(f"\n{len(links)} links ({links.groupby(['label_a', 'label_b']).ngroups} label pairs) of "
          f"{'/'.join(map(str, n_tiles))} tiles ({'/'.join(lengths_cm)} cm)")
    print(links.groupby(["n_tiles", "direction", "wall"]).agg(links=("exp", "size"), min_cm=("wall_cm", "min"),
                                                              max_cm=("wall_cm", "max")).round(1).to_string())

    print("\nloading θ̂, α̂ …")
    bidir = {e: load_bidir(e) for e in exps}
    X, y, meta, freqs = build_dataset(links, bidir, freq_range, attn, zero_start, unwrap)
    if not np.array_equal(np.asarray(freqs, float), np.asarray(info["freqs"], float)):
        raise SystemExit(f"real frequencies {list(freqs)} differ from the model's {info['freqs']}")
    assert X.shape[1] == pretrained.token_cols.size + len(pretrained.link_mu), "feature layout differs from the model's"
    print(f"{len(X)} samples, {X.shape[1]} features; the sim target was {pretrained.y_mu:.0f} ± {pretrained.y_sd:.0f} cm, "
          f"the real one is {y.mean():.0f} ± {y.std():.0f} cm")

    rescnn, sim = pretrained_rescnn(args, info) if args.rescnn else (None, None)
    if rescnn is not None:
        zs = rescnn.predict(X)
        print(f"  zero-shot ResCNN on all {len(y)} real samples: MAE {mean_absolute_error(y, zs):.1f} cm, "
              f"correlation with the truth {np.corrcoef(zs, y)[0, 1]:.2f}\n")

    te, preds = evaluate(X, y, meta, args.train_frac, args.seed, args.loo, make_fit_fn(pretrained, args, rescnn, sim),
                         loo_dedup=args.loo_dedup)

    OUT_DIR.mkdir(exist_ok=True)
    tag = (args.model.stem + "_real" + "-".join(lengths_cm) + "cm" + ("_dedup" if args.dedup else "")
           + ("_loo" + ("-dedup" if args.loo_dedup else "") if args.loo else "") + (f"_max{args.max_wall_cm:g}cm" if args.max_wall_cm is not None else "")
           + f"_ft{args.ft_epochs}ep_lr{args.ft_lr:g}" + ("_head" if args.freeze_encoder else "")
           + ("_renorm" if args.renorm else "")
           + (f"_rescnn{args.rescnn_ft_epochs}ep_lr{args.rescnn_ft_lr:g}" + ("" if args.rescnn_joint else "_realonly")
              + ("" if args.rescnn_aug else "_noaug") if args.rescnn else ""))
    out = meta.iloc[te].reset_index(drop=True).assign(**{f"pred_{k}": v for k, v in preds.items()})
    out.to_csv(OUT_DIR / f"test_predictions_{tag}.csv", index=False)
    plot_predictions(y[te], preds, meta.iloc[te], OUT_DIR / f"pred_vs_true_{tag}.png",
                     f"{', '.join(lengths_cm)} cm real links, sim-pretrained transformer ({args.model.stem})"
                     f"{' and ResCNN' if args.rescnn else ''}: distance from closest parallel wall (test set)")
    print(f"\nwrote {(OUT_DIR / f'test_predictions_{tag}.csv').relative_to(ROOT)}, "
          f"{(OUT_DIR / f'pred_vs_true_{tag}.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
