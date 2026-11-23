# Room-sweep transformer

The room-sweep version of the multi-tag pipeline (`multitag/cluster_features.py` → `multitag_ml_training/train_cluster.py` →
`ETF_eval_master.ipynb`). It uses the same `TagPairTransformer`, loss and trainer (imported from `~/T2TSionnaSim`, override
with `T2T_ROOT`). Targets are the same too: the distance [cm] and incidence angle [deg] of the wall hit by four horizontal rays
(N, S, E, W) from the sub-cluster centroid.

| Step | File | Output |
|---|---|---|
| 1. Generate sub-clusters | `roomsweeps_cluster_features.py` | `data/roomsweeps_K{K}_P{P}[...].npz` |
| 2. Train | `train_roomsweeps.py` | `Saved/ETF_rs_*.pth` + `_config.json` + `_split.csv` |
| 3. Evaluate / visualize | `roomsweeps_eval.ipynb` | tables, plots, `_perexp_eval.csv`, `_val_predictions.csv` |
| 2b. Leave-one-out training | `loo_roomsweeps.py` | `Saved/loo_<run_id>/` (one model per fold + manifest + summary) |
| 3b. LOO evaluation | `roomsweeps_eval_loo.ipynb` | per-fold tables and plots, `loo_perfold_eval.csv`, `loo_val_predictions.csv` |

Use the T2TSionnaSim venv: `PY=~/T2TSionnaSim/.venv/bin/python`. Run everything from this folder.

## 1. Generate

```bash
$PY roomsweeps_cluster_features.py 6 1 --max-span-cm 150        # 16 660 sub-clusters, 15 links each (~140 MB)
$PY roomsweeps_cluster_features.py 4 3 --max-span-cm 104        # smaller clusters, 3 orderings each
$PY roomsweeps_cluster_features.py 10 1                          # 45 links, like the multitag K10 runs (~1.2 GB)
$PY roomsweeps_cluster_features.py 6 1 --max-span-cm 150 --frame room   # N/S/E/W fixed to the room's walls
```

- `K P`: tags per sub-cluster and random tag orderings per combination. Tag 0 of the ordering sets the local frame,
  so with `--frame local`, P > 1 also augments the targets.
- `--max-span-cm`: keep only compact sub-clusters (largest nominal tag-to-tag distance).
- `--max-combos-per-exp`: randomly subsample combinations per experiment.
- `--experiments`, `--freq-min/--freq-max`: default to all 17 experiments and the full 775–995 MHz band.

Walls are the four room walls from `roomsweeps_wall_distance.find_walls()`, and rays are intersected with them
analytically, so furniture never counts as a wall. θ̂ and α̂ come from the cached bidirectional tables in
`organized_multitag/.cache`.

## 2. Train

```bash
D=data/roomsweeps_K6_P1_span150cm.npz
$PY train_roomsweeps.py --dataset $D --val_columns 7 8 9                        # location holdout (honest)
$PY train_roomsweeps.py --dataset $D --val_experiments 8                        # experiment holdout, link locations purged
$PY train_roomsweeps.py --dataset $D --val_columns 7 8 9 --include_positions    # + tag positions (local frame)
$PY train_roomsweeps.py --dataset $D --split group                              # random sub-cluster split (optimistic)
```

Features per link slot are θ̂ (unwrapped over frequency) and α̂ per frequency. **Tag positions are opt-in**
(`--include_positions`, `--pos_frame local|room`): they encode link direction and length. The run name says
`_pos<frame>` or `_nopos`.

Splits:

- `--val_columns c…`: val is the sub-clusters with every tag in those grid columns (0–16), train is the ones with no
  tag there, and the rest is unused.
- `--val_experiments e…`: train samples that share a floor-label pair with val are dropped (`--no_purge` keeps them,
  which leaks).
- Default: a random split of the samples.

Every floor label always holds the same tag, so random and experiment splits re-measure the same links. The notebook
reports a 1-NN fingerprint baseline to expose this.

Other options match `train_cluster.py` (`--epochs 300 --batch_size 32 --lr 2e-3 --dist_clip 300 --gpu 0 ...`).
`--dist_clip` clamps the GT in the loss. In this room about half of the rays are longer than 3 m, so a larger clip
(e.g. 1000) trains on the far walls too. The notebook scores only rays with GT ≤ the clip by default
(`EVAL_MAX_GT_CM`).

## 3. Evaluate

Open `roomsweeps_eval.ipynb` and run all cells. It uses the newest checkpoint in `Saved/` (or set `CHECKPOINT_PATH`).
It shows:

- per-experiment metrics against the constant and 1-NN baselines
- per-ray MAE
- predicted against true values
- error CDFs and error against range
- a floor map of error by centroid
- predicted wall hits drawn in the room
- one sub-cluster with its rays
- the train/val gap

## Leave-one-out

`loo_roomsweeps.py` trains one model per fold from scratch. Each fold is a separate `train_roomsweeps.py` process,
so a fold is exactly what a single manual run would produce.

```bash
# leave one experiment out: 17 folds, every GPU, 2 folds at a time per GPU, at most 300 epochs each
$PY loo_roomsweeps.py --dataset data/roomsweeps_K6_P1_span150cm.npz --gpus all --jobs_per_gpu 2 --epochs 300

# leave one block of 3 grid columns out (location holdout, 6 folds)
$PY loo_roomsweeps.py --dataset data/roomsweeps_K6_P1_span150cm.npz --fold_by columns --gpus all --jobs_per_gpu 2 --epochs 300
```

- `--epochs N` is the maximum number of epochs per fold. Each fold keeps its best-val-loss checkpoint.
- `--gpus all` (default) uses every GPU `nvidia-smi` lists, or `--gpus 0 1` picks GPUs. `--jobs_per_gpu` (default 2)
  sets how many folds share one GPU. A new fold starts whenever one finishes.
- Any other `train_roomsweeps.py` flag is passed through to every fold (`--include_positions`, `--dist_clip 1000`,
  `--batch_size 128`, `--no_purge`, …).
- Experiment folds drop training samples that share a link location with the held-out experiment.
  Column folds (`--block_size`, default 3) hold out the sub-clusters lying entirely in the block and train only on
  sub-clusters with no tag in it.
- `--folds 3 8` runs only some folds (block indices with `--fold_by columns`). `--dry_run` prints the commands.
- Resume: rerun with `--run_id <id>` and the same arguments, and finished folds are skipped. Logs are in
  `Saved/loo_<id>/logs/`.

Then open `roomsweeps_eval_loo.ipynb`, which uses the newest `Saved/loo_*` (or set `LOO_DIR`). Each fold's model
is scored only on the data it held out, against that fold's constant and 1-NN baselines. Every sample is held out
exactly once, so the pooled table and the floor error map cover the whole dataset.
