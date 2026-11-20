# Multi-tag wall distance: commands

Run these in a terminal, starting from the repository root (`RFImaging`).

```zsh
cd organized_multitag/roomsweeps_processing
```

## Run all cluster sizes (n = 4 to 8)

Runs one size after another at the 150 cm span limit. `caffeinate -i` keeps the Mac awake. Press `Ctrl+C` to stop everything.

```zsh
caffeinate -i zsh -c 'for n in 4 5 6 7 8; do PYTHONUNBUFFERED=1 ../../.venv/bin/python roomsweeps_wall_distance_multitag.py --n-tags $n --max-span-cm 150 2>&1 | tee roomsweeps_wall_distance_multitag_out/run_log_n${n}_span150cm.txt; done'
```

## Run one cluster size

Each command is independent, so you can run only the sizes you need.

| n | Links per cluster | Clusters | Estimated time |
|---|---|---|---|
| 4 | 6 | 11,662 | 2 hours |
| 5 | 10 | 16,898 | 4.5 hours |
| 6 | 15 | 16,660 | 6.5 hours |
| 7 | 21 | 11,084 | 6 hours |
| 8 | 28 | 4,777 | 3.5 hours |

All five take about 22 hours in total. Only the n = 4 time is based on measured folds. The others are scaled from it.

```zsh
caffeinate -i zsh -c 'PYTHONUNBUFFERED=1 ../../.venv/bin/python roomsweeps_wall_distance_multitag.py --n-tags 4 --max-span-cm 150 2>&1 | tee roomsweeps_wall_distance_multitag_out/run_log_n4_span150cm.txt'
```

```zsh
caffeinate -i zsh -c 'PYTHONUNBUFFERED=1 ../../.venv/bin/python roomsweeps_wall_distance_multitag.py --n-tags 5 --max-span-cm 150 2>&1 | tee roomsweeps_wall_distance_multitag_out/run_log_n5_span150cm.txt'
```

```zsh
caffeinate -i zsh -c 'PYTHONUNBUFFERED=1 ../../.venv/bin/python roomsweeps_wall_distance_multitag.py --n-tags 6 --max-span-cm 150 2>&1 | tee roomsweeps_wall_distance_multitag_out/run_log_n6_span150cm.txt'
```

```zsh
caffeinate -i zsh -c 'PYTHONUNBUFFERED=1 ../../.venv/bin/python roomsweeps_wall_distance_multitag.py --n-tags 7 --max-span-cm 150 2>&1 | tee roomsweeps_wall_distance_multitag_out/run_log_n7_span150cm.txt'
```

```zsh
caffeinate -i zsh -c 'PYTHONUNBUFFERED=1 ../../.venv/bin/python roomsweeps_wall_distance_multitag.py --n-tags 8 --max-span-cm 150 2>&1 | tee roomsweeps_wall_distance_multitag_out/run_log_n8_span150cm.txt'
```

## Output

Each run writes these files to `roomsweeps_wall_distance_multitag_out/`, where `<n>` is the cluster size:

| File | Contents |
|---|---|
| `run_log_n<n>_span150cm.txt` | Console output, including each fold's results |
| `summary_n<n>_span150cm.csv` | Pooled results for each model |
| `folds_n<n>_span150cm.csv` | Results and data sizes for each held-out experiment |
| `predictions_n<n>_span150cm.csv` | Held-out predictions for each cluster |
| `pred_vs_true_n<n>_span150cm.png` | Predicted against true distance |
| `errors_n<n>_span150cm.png` | Error distribution and MAE by true distance |

The summary, folds, predictions and plot files are written only when a run finishes all 17 folds.

## Existing files that will be replaced

- `*_n8_span150cm.*` came from a 2-fold, 1-epoch test, so don't use them. A full n = 8 run replaces them.
- `run_log_n4_span150cm.txt` holds the stopped n = 4 run (6 of 17 folds). A new n = 4 run replaces it, so copy it first if you want to keep it.

## Useful options

| Option | Effect |
|---|---|
| `--max-span-cm 104` | Changes the cluster span limit |
| `--max-clusters-per-exp 300` | Randomly keeps at most this many clusters per experiment, which shortens the run |
| `--models mlp,transformer` | Trains only the listed models (`svr`, `mlp`, `rescnn`, `transformer`) |
| `--no-attn` | Uses phase only |
| `--no-purge` | Keeps training clusters that share a label pair with the held-out experiment, which makes results look better than they are |
