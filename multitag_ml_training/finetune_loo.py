"""
Leave-one-experiment-out (LOO) fine-tuning.

For every experiment e in the dataset (or in --folds), fine-tune the pretrained
checkpoint with finetune_cluster.py on all *other* experiments and validate on e
(--val_experiments e). Each fold is a separate finetune_cluster.py process, so
a fold is exactly what a single manual fine-tune run would produce; folds are
spread over the given GPUs.

Every argument this script does not know is forwarded to finetune_cluster.py
unchanged (--epochs, --lr, --freeze_layers, --dropout, --refit_scalers,
--dist_clip, --batch_size, …). --val_experiments, --save_dir and --gpu are set
per fold and may not be passed.

Output: multitag_ml_training/Saved/finetune_<run_id>/
    loo_manifest.json                  run settings + one entry per fold (status,
                                       checkpoint, pretrained / fine-tuned metrics)
    loo_summary.csv                    one row per held-out experiment
    <stem>_valexp<e>_….pth             fold checkpoints, + _config.json / _split.csv
    logs/fold_exp<e>.log               stdout/stderr of each fold
read by ETF_eval_loo.ipynb.

Examples
--------
python multitag_ml_training/finetune_loo.py \\
    --model /home/manaj/T2TSionnaSim/MLScripts/Saved/ETF_realistic_45slots_attn_d128_l6_ep300_s42_1014155745.pth \\
    --dataset multitag/cluster_features_K10_P1.csv \\
    --gpus 0 1 --epochs 30 --freeze_layers 4 --dist_clip 300

# a few folds only / resume an interrupted run (finished folds are skipped)
python multitag_ml_training/finetune_loo.py --model … --dataset … --folds 3 12
python multitag_ml_training/finetune_loo.py --model … --dataset … --run_id 1014231500 --epochs 30 …
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
FINETUNE = os.path.join(_HERE, "finetune_cluster.py")
RESERVED = {"--val_experiments", "--save_dir", "--gpu"}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Leave-one-experiment-out fine-tuning (wraps finetune_cluster.py). "
                    "Unknown arguments are forwarded to finetune_cluster.py.",
    )
    p.add_argument("--model", required=True, help="Pretrained .pth")
    p.add_argument("--dataset", required=True, help="cluster_features_K*_P*.csv")
    p.add_argument("--folds", type=int, nargs="+", default=None,
                   help="Experiments to hold out, one fold each (default: every experiment "
                        "in the dataset, or in --experiments if forwarded)")
    p.add_argument("--run_id", default=None,
                   help="Output folder Saved/finetune_<run_id> (default: MMDDHHMMSS). "
                        "An existing run_id resumes: folds with a _config.json are skipped")
    p.add_argument("--saved_root", default=os.path.join(_HERE, "Saved"))
    p.add_argument("--gpus", type=int, nargs="+", default=[0],
                   help="CUDA devices to spread the folds over")
    p.add_argument("--jobs_per_gpu", type=int, default=1)
    p.add_argument("--dry_run", action="store_true", help="Print the fold commands and exit")
    args, extra = p.parse_known_args()
    bad = RESERVED & {a.split("=")[0] for a in extra}
    if bad:
        p.error(f"{sorted(bad)} are set per fold by finetune_loo.py")
    return args, extra


def _forwarded(extra: list[str], name: str) -> list[int] | None:
    """Values of a forwarded `--name v1 v2 …` option (e.g. --experiments), if present."""
    if name not in extra:
        return None
    vals = []
    for tok in extra[extra.index(name) + 1:]:
        if tok.startswith("--"):
            break
        vals.append(int(tok))
    return vals


def dataset_experiments(dataset: str) -> list[int]:
    """Experiments in the dataset, from the parsed cache if present (fast), else the CSV."""
    cache = os.path.splitext(dataset)[0] + ".parsed.npz"
    if os.path.isfile(cache):
        return sorted(np.unique(np.load(cache)["keys"][:, 0]).astype(int).tolist())
    exps = set()
    for ch in pd.read_csv(dataset, usecols=["experiment"], chunksize=500_000):
        exps |= set(ch["experiment"].unique().tolist())
    return sorted(int(e) for e in exps)


def fold_config(run_dir: str, exp: int) -> str | None:
    # "_valexp1_" never matches "_valexp11_": the experiment is followed by "_ep".
    hits = sorted(glob.glob(os.path.join(run_dir, f"*_valexp{exp}_ep*_config.json")),
                  key=os.path.getmtime)
    return hits[-1] if hits else None


def fold_entry(run_dir: str, exp: int, status: str, seconds: float | None = None) -> dict:
    entry = dict(val_experiment=exp, status=status, seconds=seconds)
    cfg_path = fold_config(run_dir, exp)
    if cfg_path:
        with open(cfg_path) as f:
            cfg = json.load(f)
        entry |= dict(
            run_stem=cfg["run_stem"],
            checkpoint=cfg["run_stem"] + ".pth",
            config=os.path.basename(cfg_path),
            n_train=cfg["n_train"], n_val=cfg["n_val"],
            best_epoch=cfg["best_epoch"],
            pretrained_metrics=cfg.get("pretrained_metrics"),
            final_metrics=cfg["final_metrics"],
        )
    return entry


def write_summary(run_dir: str, manifest: dict) -> None:
    with open(os.path.join(run_dir, "loo_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    rows = []
    for e in manifest["folds"]:
        fold = manifest["results"].get(str(e), {})
        pre, fin = fold.get("pretrained_metrics") or {}, fold.get("final_metrics") or {}
        rows.append(dict(
            val_experiment=e, status=fold.get("status", "pending"),
            n_train=fold.get("n_train"), n_val=fold.get("n_val"), best_epoch=fold.get("best_epoch"),
            pretrained_mae_dist_m=pre.get("mae_dist_m"), mae_dist_m=fin.get("mae_dist_m"),
            pretrained_mae_angle_deg=pre.get("mae_angle_deg"), mae_angle_deg=fin.get("mae_angle_deg"),
            checkpoint=fold.get("checkpoint"),
        ))
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "loo_summary.csv"), index=False)


def main() -> None:
    args, extra = parse_args()
    args.model, args.dataset = os.path.abspath(args.model), os.path.abspath(args.dataset)
    folds =args.folds or _forwarded(extra, "--experiments") or dataset_experiments(args.dataset)
    if len(folds) < 2 and not args.folds:
        sys.exit(f"need at least 2 experiments for leave-one-out, found {folds}")

    run_id = args.run_id or datetime.now().strftime("%m%d%H%M%S")
    run_dir = os.path.join(args.saved_root, f"finetune_{run_id}")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    manifest_path = os.path.join(run_dir, "loo_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            old = json.load(f)
        if old["forwarded_args"] != extra or os.path.abspath(args.model) != old["model"]:
            sys.exit(f"{run_dir} was started with different settings:\n"
                     f"  model {old['model']}\n  args  {old['forwarded_args']}\n"
                     f"use a new --run_id, or the same arguments to resume")
        results = old.get("results", {})
    else:
        results = {}

    manifest = dict(
        run_id=run_id,
        created=datetime.now().isoformat(timespec="seconds"),
        model=os.path.abspath(args.model),
        dataset=os.path.abspath(args.dataset),
        folds=folds,
        forwarded_args=extra,
        gpus=args.gpus,
        results=results,
    )

    todo = []
    for e in folds:
        if fold_config(run_dir, e):
            manifest["results"][str(e)] = fold_entry(run_dir, e, "done",
                                                     results.get(str(e), {}).get("seconds"))
            print(f"[loo] exp {e}: already done — skipped")
        else:
            todo.append(e)

    def cmd(e: int) -> list[str]:
        return [sys.executable, "-u", FINETUNE,
                "--model", args.model, "--dataset", args.dataset,
                "--val_experiments", str(e), "--save_dir", run_dir, "--gpu", "0", *extra]

    print(f"[loo] run dir : {run_dir}")
    print(f"[loo] folds   : {folds}  ({len(todo)} to run)")
    print(f"[loo] forward : {' '.join(extra) or '(finetune_cluster defaults)'}")
    if args.dry_run:
        for e in todo:
            print(" ".join(cmd(e)))
        return
    write_summary(run_dir, manifest)

    # One worker thread per GPU slot, pulling folds from a shared queue.
    queue, lock = deque(todo), threading.Lock()
    failed = []

    def worker(gpu: int) -> None:
        while True:
            with lock:
                if not queue:
                    return
                e = queue.popleft()
            log_path = os.path.join(log_dir, f"fold_exp{e}.log")
            env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)}
            print(f"[loo] exp {e}: start on GPU {gpu}  (log {os.path.relpath(log_path, _HERE)})")
            t0 = time.time()
            with open(log_path, "w") as log:
                log.write(" ".join(cmd(e)) + "\n\n")
                log.flush()
                rc = subprocess.run(cmd(e), stdout=log, stderr=subprocess.STDOUT,
                                    env=env, cwd=_HERE).returncode
            dt = time.time() - t0
            with lock:
                ok = rc == 0 and fold_config(run_dir, e) is not None
                entry = fold_entry(run_dir, e, "done" if ok else f"failed (rc={rc})", dt)
                manifest["results"][str(e)] = entry
                write_summary(run_dir, manifest)
                if ok:
                    m, p = entry["final_metrics"], entry.get("pretrained_metrics") or {}
                    print(f"[loo] exp {e}: done in {dt / 60:.1f} min — "
                          f"MAE {m['mae_dist_m']:.3f} m / {m['mae_angle_deg']:.1f}° "
                          f"(pretrained {p.get('mae_dist_m', float('nan')):.3f} m)")
                else:
                    failed.append(e)
                    print(f"[loo] exp {e}: FAILED (rc={rc}) — see {log_path}")

    threads = [threading.Thread(target=worker, args=(g,))
               for g in args.gpus for _ in range(args.jobs_per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    write_summary(run_dir, manifest)
    print(f"\n[loo] summary → {os.path.join(run_dir, 'loo_summary.csv')}")
    print(pd.read_csv(os.path.join(run_dir, "loo_summary.csv")).to_string(index=False))
    if failed:
        sys.exit(f"[loo] failed folds: {failed} — rerun with --run_id {run_id} to retry them")


if __name__ == "__main__":
    main()
