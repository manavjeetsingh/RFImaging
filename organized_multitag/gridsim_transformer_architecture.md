# Frequency-token transformer for wall distance

Architecture of the small transformer that `gridsim_wall_distance.py` and `gridsim_wall_distance_fit.py` fit
alongside the SVR and MLP. It predicts `wall_cm`, the distance of a link's midpoint from the closest parallel wall, from
the same feature matrix `X` the other models get. The code is in `gridsim_wall_distance.py`, section "transformer":
`FreqTransformer` (the network), `TransformerRegressor` (scaling, training, sklearn-style `fit` / `predict`) and
`transformer_models` (built from the command line). `--no-transformer` turns it off.

**The link direction and length are never inputs**, to this model or to any other. They are only used to select links
(`--n-tiles`, `--length-cm`) and to break down the results by direction.

## 1. Idea

The SVR and MLP see one flat vector per link, so to them "θ̂ at 800 MHz" and "θ̂ at 805 MHz" are unrelated columns.
The transformer reads the frequency sweep as a **sequence**. Each frequency is one token, and attention can compare
any two frequencies directly, which matters for the ripple period and the slope of the phase. A learned **read-out
token** (like `[CLS]`) goes at the front, and the regression head reads its output. In the fit script, the sine fit
(A, period), which has one value per link and not per frequency, is projected and added to that read-out token.

## 2. Inputs

`X` is split by column into **frequency tokens** and **per-link features**. Each script's `main` passes `token_cols`, a
`(channels, F)` array of column indices into `X`. Every column not in `token_cols` is a per-link feature.

| Script | F (default band) | Token channels per frequency | Per-link features |
|---|---|---|---|
| `gridsim_wall_distance.py` | 18 (775–955 MHz) | θ̂ unwrapped, α̂ | none |
| `gridsim_wall_distance_fit.py` | 18 (775–955 MHz) | Δ (phase-fit residual), α̂ | sine A, sine period |

With `--no-attn` there is a single channel per token (θ̂ or Δ).

Column layout of `X` that `token_cols` points into:

```
gridsim_wall_distance.py       [ θ̂ f1…fF | α̂ f1…fF ]
                                 └ ch 0 ┘   └ ch 1 ┘

gridsim_wall_distance_fit.py   [ Δ f1…fF | A period | α̂ f1…fF ]
                                 └ ch 0 ┘  └ per-link ┘ └ ch 1 ┘
```

### Scaling (fitted on the training rows only)

- **Token channels:** one mean and std per channel, computed over all frequencies together. Scaling each frequency
  separately, as `StandardScaler` does for the SVR and MLP, would remove the frequency trend, e.g. the slope of the
  unwrapped phase. With one scale per channel, the curve's shape across frequency is kept.
- **Per-link features:** one mean and std per column.
- **Target:** standardized. Predictions are mapped back to cm.

## 3. Network (`FreqTransformer`)

Defaults: `dim = 64`, `heads = 4`, `layers = 3`, feed-forward width `2·dim = 128`, dropout 0.1.

```
  learned read-out vector (64)            frequency tokens (N, F, C)
            │                                       │
            + Linear G → 64 (sine A, period;   Linear C → 64
            │   fit script only, G = 2)             │
            │                            + learned position embedding (F, 64)
            │                                       │
            └──────────────── concat ───────────────┘
                                │
                 sequence (N, 1 + F, 64)      token 0 = read-out token
                                │
               ┌────────────────┴────────────────┐
               │  Transformer encoder layer ×3   │   pre-norm (norm_first=True):
               │    x += MHA(LN(x))   4 heads    │   self-attention over all 1 + F tokens,
               │    x += FFN(LN(x))   64→128→64  │   no mask
               └────────────────┬────────────────┘
                                │
                     output of token 0 (N, 64)
                                │
                   LayerNorm → Linear 64 → 1
                                │
                 standardized wall distance (N,)
```

- **Position embedding:** learned, one vector per frequency index (not per MHz value). The model is therefore tied to
  the frequency grid it was trained on: same band, same number of frequencies.
- **Read-out token:** a learned vector with no position embedding. When the script has per-link features, a linear
  projection of them is added to it. With none, as in the phase script, it is the same for every link and carries
  information only through attention.
- **Head:** reads only token 0. The frequency tokens reach the head only through attention.

### Size

| Part | Phase script (C = 2, F = 18, G = 0) | Fit script (C = 2, F = 18, G = 2) |
|---|---|---|
| token embedding `Linear(2, 64)` | 192 | 192 |
| position embedding `F × 64` | 1 152 | 1 152 |
| read-out vector | 64 | 64 |
| per-link projection `Linear(G, 64)` | — | 192 |
| 3 encoder layers (33 472 each: attention 16 640, FFN 16 576, 2 LayerNorms 256) | 100 416 | 100 416 |
| head `LayerNorm(64)` + `Linear(64, 1)` | 193 | 193 |
| **total** | **102 017** | **102 209** |

## 4. Training (`TransformerRegressor.fit`)

| Setting | Value |
|---|---|
| loss | Huber (δ = 1) on the standardized target |
| optimizer | AdamW, lr 1e-3, weight decay 1e-2 |
| schedule | one-cycle: 5 % warm-up, then cosine decay, stepped every batch |
| batch size | 256 |
| epochs | 100 (`--tf-epochs`), fixed, no early stopping |
| seed | `--seed` (`torch.manual_seed`) |
| device | CUDA if available, else CPU; all training data is held on the device |

The transformer is fitted on the same rows as the SVR and MLP: the training simulations of the split, subsampled to
`--max-train` (default 20 000). With `--kfold K` a fresh model is trained per fold. It is reported in the same tables,
prediction CSVs (`pred_Transformer`) and plots as the other models.

Command-line knobs (shared by both scripts):

```
--transformer / --no-transformer   fit it or not (default: fit)
--tf-epochs 100                    training epochs
--tf-dim 64                        width (must be divisible by the 4 heads)
--tf-layers 3                      encoder layers
```

## 5. Results (775–955 MHz, default 70 / 30 split by simulation, 20 000 training rows)

Test MAE in cm, without direction or length as inputs. `--length-cm L` keeps links L ± 23 cm long (any tile count).

| Links | Features | mean of train | SVR | MLP | Transformer |
|---|---|---|---|---|---|
| 69–115 cm (`--length-cm 92`) | θ̂ + α̂ | 109.8 | **74.5** | 79.3 | 76.5 |
| | Δ + sine fit + α̂ | 109.8 | 79.1 | 86.9 | **77.5** |
| 23–161 cm (`--length-cm 46,92,138`) | θ̂ + α̂ | 109.0 | 91.7 | 88.0 | **84.3** |
| | Δ + sine fit + α̂ | 109.0 | 90.9 | 97.7 | **90.0** |
| 69–115 cm, ≤ 200 cm from the wall | θ̂ + α̂ | 42.9 | 18.1 | 17.5 | **15.0** |
| | Δ + sine fit + α̂ | 42.9 | 19.0 | 20.7 | **16.6** |

The transformer is the best model in most configurations, most clearly when links far from the wall are left out
(`--max-wall-cm`).
