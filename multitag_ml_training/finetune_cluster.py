"""
CLI for fine-tuning a TagPairTransformer checkpoint (from train_cluster.py or
an earlier finetune_cluster.py run) on a new cluster_features CSV.

What is inherited from the pretrained run
-----------------------------------------
  architecture   model_config from the checkpoint (only --dropout may change)
  feature flags  no_alpha / include_avgv / no_positions / pos_frame / run_mode,
                 read from <model>_config.json (or inferred from the checkpoint's
                 feature_slices); the new dataset must give the same feat_dim
                 and number of pair slots (same K, same frequency count)
  scalers        the checkpoint's feature scalers are applied as-is, so the
                 pretrained weights see inputs on the scale they were trained
                 on (--refit_scalers fits new ones on the new train split)
  loss settings  l2_lambda / huber_delta / dist_clip, unless given

Train / val split on the new dataset works exactly as in train_cluster.py
(--split random|group, --test_size, --val_experiments, --experiments).
Before training, the pretrained model is scored on the new val set as a
baseline.

Freezing
--------
  --freeze_layers N   freeze the input projection, CLS token, position
                      embedding and the first N encoder layers; the remaining
                      layers, final norm and the two heads are trained.
  --freeze_layers -1  freeze the whole encoder (heads + final norm only).

Examples
--------
python multitag_ml_training/finetune_cluster.py \\
    --model multitag_ml_training/Saved/ETF_cluster_K10_45slots_alpha_poslocal_d128_l6_ep300_s42_1012153000.pth \\
    --dataset multitag/cluster_features_K10_P2.csv --epochs 100 --lr 1e-4

python multitag_ml_training/finetune_cluster.py \\
    --model <pretrained>.pth --dataset <new>.csv \\
    --val_experiments 12 --freeze_layers 4

Saved: multitag_ml_training/Saved/<pretrained run_stem>_ft_ep{e}_s{s}_{MMDDHHMMSS}.pth
       + matching _config.json and _split.csv (read by ETF_eval_cluster.ipynb)
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_cluster import (
    _HERE,
    TagPairTransformer,
    Trainer,
    ValTensors,
    _param_groups,
    _seed_everything,
    combined_loss,
    load_for_args,
    preprocess,
    save_split,
    scalers_from_checkpoint,
    split_indices,
)

# Feature flags that must match the pretrained run (they define feat_dim).
FEATURE_FLAGS = ["no_alpha", "include_avgv", "no_positions", "pos_frame", "run_mode"]
LOSS_DEFAULTS = dict(l2_lambda=1e-4, huber_delta=0.1, dist_clip=1000.0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a TagPairTransformer checkpoint on a cluster_features CSV"
    )

    # Inputs
    p.add_argument("--model", required=True,
                   help="Pretrained .pth (its _config.json next to it is used if present)")
    p.add_argument("--dataset", required=True,
                   help="cluster_features_K*_P*.csv to fine-tune on")

    # Data (feature flags are inherited from the pretrained run)
    p.add_argument("--pos_frame", choices=["local", "world"], default=None,
                   help="Only needed if the pretrained run has no _config.json")
    p.add_argument("--run_mode", choices=["each", "mean"], default=None,
                   help="Default: as in the pretrained run")
    p.add_argument("--experiments", type=int, nargs="+", default=None,
                   help="Use only these experiments (default: all)")
    p.add_argument("--val_experiments", type=int, nargs="+", default=None,
                   help="Hold these experiments out as the val set; --test_size is then ignored")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="Fraction of samples (or sub-clusters with --split group) for val")
    p.add_argument("--split", choices=["random", "group"], default="random",
                   help="random: samples drawn at random; group: whole sub-clusters on one side")
    p.add_argument("--refit_scalers", action="store_true",
                   help="Fit new feature scalers on the new train split instead of "
                        "reusing the pretrained ones")

    # Model
    p.add_argument("--dropout", type=float, default=None,
                   help="Override the pretrained dropout")
    p.add_argument("--freeze_layers", type=int, default=0,
                   help="Freeze the embedding + first N encoder layers (-1 = whole encoder)")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="Linear LR warmup (0 = ReduceLROnPlateau)")
    p.add_argument("--weight_decay", type=float, default=0.01,
                   help="AdamW weight decay on weight matrices")
    p.add_argument("--l2_lambda", type=float, default=None,
                   help="Default: as in the pretrained run")
    p.add_argument("--huber_delta", type=float, default=None,
                   help="Default: as in the pretrained run")
    p.add_argument("--dist_clip", type=float, default=None,
                   help="Default: as in the pretrained run")
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--save_dir", default=os.path.join(_HERE, "Saved"))
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0,
                   help="CUDA device index (ignored if CUDA unavailable)")

    return p.parse_args()


# ── Pretrained run ────────────────────────────────────────────────────────────

def adapt_sim_checkpoint(ckpt: dict) -> bool:
    """
    Convert a T2TSionnaSim train_realistic.py checkpoint in place to the
    train_cluster.py layout: build feature_slices from include_attns /
    include_positions and rename the "attn" scaler to "alpha" (same feature).
    Sim positions are centroid-relative in the world frame. Returns True if
    the checkpoint was a sim one.
    """
    if "feature_slices" in ckpt:
        return False
    n_freq = ckpt["n_freq"]
    slices, cur = {"theta": [0, n_freq]}, n_freq
    if ckpt.get("include_attns", False):
        slices["alpha"] = [cur, cur + n_freq]; cur += n_freq
    if ckpt.get("include_positions", False):
        slices["pos"] = [cur, cur + 6]; cur += 6
    if cur != ckpt["model_config"]["feat_dim"]:
        raise ValueError(f"cannot infer the feature layout of the checkpoint: "
                         f"slices {slices} give feat_dim={cur}, model has "
                         f"{ckpt['model_config']['feat_dim']}")
    ckpt["feature_slices"] = slices
    ckpt["scalers"] = {("x_alpha" if k == "x_attn" else k): v
                       for k, v in ckpt["scalers"].items()}
    print(f"[finetune] sim checkpoint (train_realistic) — feature layout {slices}")
    return True


def parent_args(model_path: str, ckpt: dict, is_sim: bool = False) -> dict:
    """Args of the pretrained run: from its _config.json, else inferred from the checkpoint."""
    cfg_path = model_path.replace(".pth", "_config.json")
    if os.path.isfile(cfg_path) and not is_sim:
        with open(cfg_path) as f:
            return json.load(f)["args"]
    pos_frame = "world" if is_sim else "local"
    print(f"[finetune] no train_cluster config — inferring feature flags from the "
          f"checkpoint (pos_frame / run_mode default to {pos_frame} / each)")
    slices = ckpt["feature_slices"]
    return dict(no_alpha="alpha" not in slices, include_avgv="avgv" in slices,
                no_positions="pos" not in slices, pos_frame=pos_frame, run_mode="each")


def freeze(model: TagPairTransformer, n_layers: int) -> None:
    if n_layers == 0:
        return
    layers = model.transformer.layers
    n = len(layers) if n_layers < 0 else n_layers
    if n > len(layers):
        raise ValueError(f"--freeze_layers {n_layers}: model has only {len(layers)} layers")
    frozen = [model.input_proj, *layers[:n]]
    for m in frozen:
        for prm in m.parameters():
            prm.requires_grad = False
    model.cls_token.requires_grad = False
    model.pos_embed.requires_grad = False
    print(f"[finetune] froze embedding + {n}/{len(layers)} encoder layers")


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else f"cuda:{args.gpu}" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}  seed: {args.seed}")

    # ── Pretrained checkpoint ─────────────────────────────────────────────────
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    is_sim = adapt_sim_checkpoint(ckpt)
    parent = parent_args(args.model, ckpt, is_sim)
    parent_stem = ckpt.get("run_stem", os.path.splitext(os.path.basename(args.model))[0])
    print(f"Pretrained : {args.model}")

    # Feature flags come from the pretrained run; CLI only fills in / overrides
    # pos_frame and run_mode.
    for k in FEATURE_FLAGS:
        if getattr(args, k, None) is None:
            setattr(args, k, parent.get(k, {"pos_frame": "local", "run_mode": "each"}.get(k, False)))
    for k, d in LOSS_DEFAULTS.items():
        if getattr(args, k) is None:
            setattr(args, k, parent.get(k, d))
    print("feature flags:", {k: getattr(args, k) for k in FEATURE_FLAGS})

    # ── Load ──────────────────────────────────────────────────────────────────
    (X_raw, Y_dists, Y_angles, centroids, poses,
     n_pairs, n_freq, meta_df, feature_slices) = load_for_args(args)
    feat_dim = X_raw.shape[2]
    n_tags = int(round((1 + np.sqrt(1 + 8 * n_pairs)) / 2))

    model_config = dict(ckpt["model_config"])
    if (n_pairs, feat_dim) != (model_config["n_slots"], model_config["feat_dim"]):
        raise ValueError(
            f"dataset gives n_slots={n_pairs}, feat_dim={feat_dim}; the model expects "
            f"n_slots={model_config['n_slots']}, feat_dim={model_config['feat_dim']} "
            f"— different K, frequency count or feature flags"
        )
    ck_slices = {k: list(v) for k, v in ckpt["feature_slices"].items()}
    if ck_slices != {k: [v.start, v.stop] for k, v in feature_slices.items()}:
        raise ValueError(f"feature layout differs from the pretrained run: {ck_slices}")

    # ── Split + preprocess ────────────────────────────────────────────────────
    train_idx, val_idx = split_indices(meta_df, args.test_size, args.seed,
                                       args.val_experiments, args.split)
    fixed_scalers = None if args.refit_scalers else scalers_from_checkpoint(ckpt)
    split, scalers = preprocess(X_raw, Y_dists, Y_angles, feature_slices,
                                train_idx, val_idx, centroids, poses,
                                scalers=fixed_scalers)
    print("scalers          :", "refit on new train split" if args.refit_scalers else "pretrained")
    print("train experiments:", sorted(meta_df.loc[train_idx, "experiment"].unique().tolist()))
    print("val experiments  :", sorted(meta_df.loc[val_idx, "experiment"].unique().tolist()))

    # ── Tensors + DataLoader ──────────────────────────────────────────────────
    def to_t(arr) -> torch.Tensor:
        return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)

    X_tr, X_va   = to_t(split["X_train"]),   to_t(split["X_val"])
    Yd_tr, Yd_va = to_t(split["Y_d_train"]), to_t(split["Y_d_val"])
    Ya_tr, Ya_va = to_t(split["Y_a_train"]), to_t(split["Y_a_val"])

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(X_tr, Yd_tr, Ya_tr),
        batch_size=args.batch_size, shuffle=True, generator=g,
    )
    val_tensors = ValTensors(X=X_va, Y_d=Yd_va, Y_a=Ya_va)

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.dropout is not None:
        model_config["dropout"] = args.dropout
    model = TagPairTransformer(**model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    freeze(model, args.freeze_layers)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} "
          f"({n_train:,} trainable)")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(_param_groups(model, args.weight_decay), lr=args.lr)
    if args.warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: min(1.0, (ep + 1) / args.warmup_epochs),
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

    # ── Loss ──────────────────────────────────────────────────────────────────
    def loss_fn(pred_d, pred_a, true_d, true_a, m):
        return combined_loss(pred_d, pred_a, true_d, true_a, m,
                             l2_lambda=args.l2_lambda, huber_delta=args.huber_delta,
                             dist_clip=args.dist_clip)

    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        loss_fn=loss_fn, device=device, use_bitmap=False, log_every=args.log_every,
    )

    # ── Baseline: pretrained model on the new val set ─────────────────────────
    baseline = trainer.evaluate(val_tensors) | {"val_loss": trainer._val_loss(val_tensors)}
    print(
        f"\nPretrained on new val: loss {baseline['val_loss']:.4f}  "
        f"MAE dist: {baseline['mae_dist_m']:.3f} m  "
        f"MAE angle: {baseline['mae_angle_deg']:.2f} deg\n"
    )

    # ── Output paths ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%m%d%H%M%S")
    run_stem = (
        f"{parent_stem}_ft"
        + (f"_frz{args.freeze_layers}" if args.freeze_layers else "")
        + (f"_valexp{'-'.join(map(str, args.val_experiments))}"
           if args.val_experiments else "")
        + f"_ep{args.epochs}_s{args.seed}_{ts}"
    )
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, f"{run_stem}.pth")
    config_path     = os.path.join(args.save_dir, f"{run_stem}_config.json")
    split_path      = os.path.join(args.save_dir, f"{run_stem}_split.csv")
    save_split(split_path, meta_df, train_idx, val_idx)
    print(f"Split      → {split_path}")

    extra_meta = dict(
        n_pairs=n_pairs,
        n_freq=n_freq,
        n_tags=n_tags,
        feat_dim=feat_dim,
        feature_slices={k: [v.start, v.stop] for k, v in feature_slices.items()},
        seed=args.seed,
        run_stem=run_stem,
        split_file=os.path.basename(split_path),
        train_idx=train_idx,
        val_idx=val_idx,
        finetuned_from=os.path.abspath(args.model),
        fixed_scalers=not args.refit_scalers,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    history = trainer.fit(
        train_loader, val_tensors,
        num_epochs=args.epochs,
        save_best_path=checkpoint_path,
        model_config=model_config,
        scaler=scalers,
        extra_meta=extra_meta,
    )

    # ── Final eval (on best checkpoint) ───────────────────────────────────────
    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    metrics    = trainer.evaluate(val_tensors)
    best_epoch = history.best_val_epoch()

    args_out = vars(args) | {"dataset": os.path.abspath(args.dataset),
                             "model": os.path.abspath(args.model)}
    with open(config_path, "w") as f:
        json.dump({
            "run_stem":        run_stem,
            "args":            args_out,
            "model_config":    model_config,
            "feature_slices":  extra_meta["feature_slices"],
            "split_file":      os.path.basename(split_path),
            "n_freq":          n_freq,
            "n_train":         int(len(train_idx)),
            "n_val":           int(len(val_idx)),
            "finetuned_from":  extra_meta["finetuned_from"],
            "fixed_scalers":   extra_meta["fixed_scalers"],
            "pretrained_metrics": baseline,
            "best_epoch":      best_epoch,
            "best_val_loss":   float(min(history.val_losses)),
            "final_metrics":   metrics,
            "train_losses":    history.train_losses,
            "val_losses":      history.val_losses,
        }, f, indent=2)

    print(
        f"\nBest val epoch: {best_epoch}  "
        f"MAE dist: {metrics['mae_dist_m']:.3f} m (pretrained {baseline['mae_dist_m']:.3f})  "
        f"MAE angle: {metrics['mae_angle_deg']:.2f} deg (pretrained {baseline['mae_angle_deg']:.2f})"
    )
    print(f"Checkpoint → {checkpoint_path}")
    print(f"Config     → {config_path}")


if __name__ == "__main__":
    main()
