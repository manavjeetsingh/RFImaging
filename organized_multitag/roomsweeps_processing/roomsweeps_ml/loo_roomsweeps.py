"""
Leave-one-out training of the room-sweep transformer (wraps train_roomsweeps.py).

Room-sweep counterpart of multitag_ml_training/finetune_loo.py. Every fold is a separate train_roomsweeps.py
process (so a fold is exactly what a single manual run would produce), trained from scratch; folds are spread
over the given GPUs.

Folds (--fold_by)
-----------------
  experiment (default)  one fold per experiment e: --val_experiments e. Training samples that share a link
                        location (floor-label pair) with e are dropped (train_roomsweeps's purge; forward
                        --no_purge to keep them, which leaks).
  columns               the floor's grid columns 0-16 cut into consecutive blocks of --block_size (default 3:
                        0-2, 3-5, …, 15-16), one fold per block: --val_columns <block>. Val = sub-clusters with
                        every tag in the block, train = sub-clusters with no tag in it. The location holdout.
  --folds restricts to some folds: experiment numbers, or block indices (0 = first block) with --fold_by columns.

Every argument this script does not know is forwarded to train_roomsweeps.py unchanged (--epochs,
--include_positions, --dist_clip, --batch_size, …). --val_experiments, --val_columns, --save_dir and --gpu are
set per fold.

Output: Saved/loo_<run_id>/
    loo_manifest.json         settings + one entry per fold (status, checkpoint, metrics)
    loo_summary.csv           one row per fold
    ETF_rs_…_valexp<e>_….pth  fold checkpoints (or _valcol<c-c-c>_), + _config.json / _split.csv
    logs/fold_<name>.log      stdout/stderr of each fold
read by roomsweeps_eval_loo.ipynb.

GPUs: --gpus all (default) uses every GPU nvidia-smi lists, running --jobs_per_gpu (default 2) folds at once on
each; a new fold starts whenever one finishes.

Examples
--------
  python loo_roomsweeps.py --dataset data/roomsweeps_K6_P1_span150cm.npz --gpus all --jobs_per_gpu 2 --epochs 300
  python loo_roomsweeps.py --dataset data/roomsweeps_K6_P1_span150cm.npz --fold_by columns --epochs 300
  python loo_roomsweeps.py --dataset … --gpus 1 --jobs_per_gpu 3 …   # one GPU only
  python loo_roomsweeps.py --dataset … --folds 3 8           # a few folds only
  python loo_roomsweeps.py --dataset … --run_id 1120150000 … # resume: finished folds are skipped
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
TRAIN = os.path.join(_HERE, "train_roomsweeps.py")
RESERVED = {"--val_experiments", "--val_columns", "--save_dir", "--gpu", "--split", "--test_size"}
N_COLS = 17


def parse_args():
    p = argparse.ArgumentParser(description="Leave-one-out training (wraps train_roomsweeps.py). "
                                            "Unknown arguments are forwarded to train_roomsweeps.py.",
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help=".npz from roomsweeps_cluster_features.py")
    p.add_argument("--fold_by", choices=["experiment", "columns"], default="experiment")
    p.add_argument("--block_size", type=int, default=3, help="grid columns per fold with --fold_by columns")
    p.add_argument("--folds", type=int, nargs="+", default=None,
                   help="only these folds: experiments, or block indices with --fold_by columns")
    p.add_argument("--run_id", default=None,
                   help="output folder Saved/loo_<run_id> (default MMDDHHMMSS); an existing one resumes")
    p.add_argument("--saved_root", default=os.path.join(_HERE, "Saved"))
    p.add_argument("--gpus", nargs="+", default=["all"],
                   help="CUDA devices to spread the folds over, or 'all' (default) for every visible GPU")
    p.add_argument("--jobs_per_gpu", type=int, default=2, help="folds trained at once on each GPU")
    p.add_argument("--dry_run", action="store_true", help="print the fold commands and exit")
    args, extra = p.parse_known_args()
    bad = RESERVED & {a.split("=")[0] for a in extra}
    if bad:
        p.error(f"{sorted(bad)} are set per fold by loo_roomsweeps.py")
    args.gpus = resolve_gpus(args.gpus)
    return args, extra


def resolve_gpus(spec: list[str]) -> list[int]:
    """GPU indices from --gpus: 'all' → every GPU nvidia-smi lists (CUDA_VISIBLE_DEVICES if set), else the ints."""
    if spec != ["all"]:
        return [int(g) for g in spec]
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return [int(g) for g in vis.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        gpus = list(range(sum(line.startswith("GPU ") for line in out.splitlines())))
    except (OSError, subprocess.CalledProcessError):
        gpus = []
    if not gpus:
        sys.exit("--gpus all: no GPU found (nvidia-smi); pass --gpus explicitly")
    return gpus


def _forwarded(extra, name):
    """Values of a forwarded `--name v1 v2 …` option, if present."""
    if name not in extra:
        return None
    vals = []
    for tok in extra[extra.index(name) + 1:]:
        if tok.startswith("--"):
            break
        vals.append(int(tok))
    return vals


def make_folds(args, extra) -> list[dict]:
    """[{name, flag, values}] — one per fold, in order."""
    if args.fold_by == "experiment":
        exps = _forwarded(extra, "--experiments") or \
            sorted(np.unique(np.load(args.dataset)["keys"][:, 0]).astype(int).tolist())
        if args.folds:
            unknown = set(args.folds) - set(exps)
            if unknown:
                sys.exit(f"--folds not in the dataset: {sorted(unknown)}")
            exps = [e for e in exps if e in args.folds]
        return [dict(name=f"exp{e}", tag=f"_valexp{e}_", flag="--val_experiments", values=[e]) for e in exps]
    blocks = [list(range(c, min(c + args.block_size, N_COLS))) for c in range(0, N_COLS, args.block_size)]
    idx = args.folds if args.folds is not None else range(len(blocks))
    out = []
    for i in idx:
        if not 0 <= i < len(blocks):
            sys.exit(f"--folds: block index {i} out of range 0-{len(blocks) - 1}")
        b = "-".join(map(str, blocks[i]))
        out.append(dict(name=f"col{b}", tag=f"_valcol{b}_", flag="--val_columns", values=blocks[i]))
    return out


def fold_config(run_dir, fold):
    # "_valexp1_" never matches "_valexp11_": the fold tag ends with "_".
    hits = sorted(glob.glob(os.path.join(run_dir, f"*{fold['tag']}*_config.json")), key=os.path.getmtime)
    return hits[-1] if hits else None


def fold_entry(run_dir, fold, status, seconds=None):
    entry = dict(fold=fold["name"], flag=fold["flag"], values=fold["values"], status=status, seconds=seconds)
    cfg_path = fold_config(run_dir, fold)
    if cfg_path:
        with open(cfg_path) as f:
            cfg = json.load(f)
        entry |= dict(run_stem=cfg["run_stem"], checkpoint=cfg["run_stem"] + ".pth",
                      config=os.path.basename(cfg_path), n_train=cfg["n_train"], n_val=cfg["n_val"],
                      best_epoch=cfg["best_epoch"], final_metrics=cfg["final_metrics"])
    return entry


def write_summary(run_dir, manifest):
    with open(os.path.join(run_dir, "loo_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    rows = []
    for fd in manifest["folds"]:
        r = manifest["results"].get(fd["name"], {})
        m = r.get("final_metrics") or {}
        rows.append(dict(fold=fd["name"], held_out=" ".join(map(str, fd["values"])),
                         status=r.get("status", "pending"), n_train=r.get("n_train"), n_val=r.get("n_val"),
                         best_epoch=r.get("best_epoch"), mae_dist_cm=m.get("mae_dist_cm"),
                         mae_angle_deg=m.get("mae_angle_deg"), minutes=(r.get("seconds") or np.nan) / 60,
                         checkpoint=r.get("checkpoint")))
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "loo_summary.csv"), index=False)


def main():
    args, extra = parse_args()
    args.dataset = os.path.abspath(args.dataset)
    requested = make_folds(args, extra)
    if not requested:
        sys.exit("no folds")
    folds = list(requested)

    run_id = args.run_id or datetime.now().strftime("%m%d%H%M%S")
    run_dir = os.path.join(args.saved_root, f"loo_{run_id}")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    manifest_path = os.path.join(run_dir, "loo_manifest.json")
    results = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            old = json.load(f)
        if (old["forwarded_args"], old["dataset"], old["fold_by"]) != (extra, args.dataset, args.fold_by):
            sys.exit(f"{run_dir} was started with different settings:\n  dataset {old['dataset']}\n"
                     f"  fold_by {old['fold_by']}\n  args    {old['forwarded_args']}\n"
                     f"use a new --run_id, or the same arguments to resume")
        results = old.get("results", {})
        names = {f["name"] for f in requested}
        folds = [f for f in old["folds"] if f["name"] not in names] + requested   # keep earlier calls' folds
        folds.sort(key=lambda f: f["values"][0])

    manifest = dict(run_id=run_id, created=datetime.now().isoformat(timespec="seconds"),
                    dataset=args.dataset, fold_by=args.fold_by, folds=folds, forwarded_args=extra,
                    gpus=args.gpus, results=results)

    for fd in folds:
        if fold_config(run_dir, fd):
            manifest["results"][fd["name"]] = fold_entry(run_dir, fd, "done",
                                                         results.get(fd["name"], {}).get("seconds"))
    todo = [fd for fd in requested if manifest["results"].get(fd["name"], {}).get("status") != "done"]
    for fd in requested:
        if fd not in todo:
            print(f"[loo] {fd['name']}: already done — skipped")

    def cmd(fd):
        return [sys.executable, "-u", TRAIN, "--dataset", args.dataset, fd["flag"], *map(str, fd["values"]),
                "--save_dir", run_dir, "--gpu", "0", *extra]

    print(f"[loo] run dir : {run_dir}")
    print(f"[loo] folds   : {[f['name'] for f in folds]}  ({len(todo)} to run)")
    print(f"[loo] forward : {' '.join(extra) or '(train_roomsweeps defaults)'}")
    if args.dry_run:
        for fd in todo:
            print(" ".join(cmd(fd)))
        return
    write_summary(run_dir, manifest)

    queue, lock, failed = deque(todo), threading.Lock(), []

    def worker(gpu):
        while True:
            with lock:
                if not queue:
                    return
                fd = queue.popleft()
            log_path = os.path.join(log_dir, f"fold_{fd['name']}.log")
            env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)}
            print(f"[loo] {fd['name']}: start on GPU {gpu}  (log {os.path.relpath(log_path, _HERE)})", flush=True)
            t0 = time.time()
            with open(log_path, "w") as log:
                log.write(" ".join(cmd(fd)) + "\n\n")
                log.flush()
                rc = subprocess.run(cmd(fd), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=_HERE).returncode
            dt = time.time() - t0
            with lock:
                ok = rc == 0 and fold_config(run_dir, fd) is not None
                entry = fold_entry(run_dir, fd, "done" if ok else f"failed (rc={rc})", dt)
                manifest["results"][fd["name"]] = entry
                write_summary(run_dir, manifest)
                if ok:
                    m = entry["final_metrics"]
                    print(f"[loo] {fd['name']}: done in {dt / 60:.1f} min — MAE {m['mae_dist_cm']:.1f} cm / "
                          f"{m['mae_angle_deg']:.1f}°  (n_train {entry['n_train']}, n_val {entry['n_val']})",
                          flush=True)
                else:
                    failed.append(fd["name"])
                    print(f"[loo] {fd['name']}: FAILED (rc={rc}) — see {log_path}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus for _ in range(args.jobs_per_gpu)]
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
