# Person location from links: algorithm

Script: `person_location_video.py`. Experiments: `walking915_1`, `walking915_2`. Empty room: `baseline915`.

## Setup

- 8 tags → 28 links (every pair T1–T2 … T7–T8).
- One frequency: 915 MHz.
- One **run** = every link measured once (~0.4 s), then a ~2 s gap.
- 70 runs per experiment (~145 s).

## Step 1: One value per link per run

- Each link has two MPPs: Rx = T1 (Tx = T2), and Rx = T2 (Tx = T1).
- Phase: θ̂ = (Θ₁ + Θ₂) / 2, mod 180°.
- Attenuation: α̂ = ½ (β₁ / V₂ + β₂ / V₁).
- Time of the value = stop time of the later of the two MPPs.
- Time 0 = start of the first MPP.

## Step 2: Score each link (how much it jumped)

- For each link, take a **rolling median** over 11 runs (centred).
- **Residual** = this run − rolling median.
  - Phase residual: compared on the doubled angle, so 179° and 1° count as 2° apart.
  - Attenuation residual: log α̂ − rolling median of log α̂.
- **Noise σ** for each link: the same residuals in `baseline915`, robust std (1.4826 × MAD).
- **Link score** = √( (phase residual / σ_phase)² + (log-α residual / σ_logα)² ).
- Result: a **frame** for each run, a vector of 28 link scores.

## Step 3: Candidate clusters

- **Link centre** = midpoint between its two tags.
- Try every group of 6 links (28 choose 6 = 376,740).
- Keep a group only if all 6 centres are within **0.4 m** of the group's own centre.
- 8,087 groups are kept. Every link appears in at least one.
- Done once. The tags do not move.

## Step 4: Pick the best cluster in each run

- **Cluster score** = geometric mean of (1 + link score) over its 6 links, minus 1.
- Why the geometric mean:
  - High only when **all 6** links jumped.
  - One huge link with 5 quiet links scores **low**.
  - The old RMS let one strong link win the whole cluster.
- Pick the cluster with the **highest** score.

## Step 5: Location

- Location = **plain average** of the 6 link centres of the chosen cluster.
- No weights.
- One location per run (~every 2 s).

## Step 6: Confidence check (is someone there?)

### What "different from before" means

- A run is **not** compared with the run just before it.
- It is compared with the link's **rolling median** over 11 runs (~22 s): 5 runs before, itself, and 5 runs after.
- The rolling median is the link's "normal" value at that time.
- A person crossing a link changes it for 1–2 runs. The median ignores this, so the residual is big.
- So the **link score** (Step 2) = how unusual this run is compared with the runs around it.
  - It is in units of the empty room's noise: a score of 5 = 5× the normal jitter of that link.

### How the threshold is set

- Run the same steps 2–4 on `baseline915` (nobody in the room).
- That gives one best-cluster score per run: 70 values, from noise and glitches only.
- **Threshold** = their 99th percentile = **8.76**.
- Meaning: an empty room beats 8.76 in only ~1 run in 100.

### Decision for each run (group of 56 MPPs)

- Best-cluster score > 8.76 → **someone is there** (solid star).
- Otherwise → **not sure** (faded star).
- Each run is decided on its own. There is no memory and no smoothing between runs (the tracker in Step 6b adds that).

### Numbers (best-cluster score)

| | p10 | median | p90 |
|---|---|---|---|
| `baseline915` (empty) | – | 3.9 | 6.4 |
| `walking915_1` | 12.1 | 20.7 | 51.3 |
| `walking915_2` | 12.3 | 20.6 | 51.4 |

- Even the quiet walking runs (p10 ≈ 12) are above 8.76.
- So the check passes almost always (70/70 and 68/70).
- Reason: while someone walks, many links move a bit, so some 6-link cluster is always raised.
- So the check says **"someone is in the room"**, not **"someone is at this spot"**.

### Limits of this check

- **Standing still:** after ~5 runs (~10 s), the person becomes part of the rolling median and the score drops.
  - So it detects **motion**, not presence.
- **Not real time:** it needs 5 runs (~10 s) after the current one.
- **Not a jump test:** the notebook's motion detection opens an event when the score **rises by more than a delta from the previous run**. This script does not do that.

## Step 6b: Tracking (linking the runs)

Steps 4–5 decide every run on its own: run k+1 does not use run k. The tracker (`--tracker`, code in
`tracking.py`) carries a belief about the position from run to run.

### grid (default): Bayes filter on a floor grid

- **State:** the position, on a 5 cm grid over the layout. Only points within `--reach` (0.3 m) of some link
  centre are allowed: elsewhere no link says anything.
- **Belief:** a probability for every grid point. Start: uniform.
- **Predict** (between runs, dt ≈ 2 s): blur the belief with a Gaussian of std `--walk-speed` × dt
  (0.45 m/s = 1 mph, the walking speed in these experiments → ~0.9 m). The person can have walked anywhere within about that distance.
- **Update** (each run): multiply by the likelihood, then normalise.
  - Soft cluster score at point p: G(p) = geometric mean of (1 + link score) over the **6 links whose centres are
    nearest to p**, weighted by exp(−d² / 2ρ²) (`--rho` 0.3 m), minus 1.
  - The same idea as the cluster score (6 links, geometric mean), but for every point of the floor, with no hard
    choice of one cluster.
  - Likelihood = (1 + G(p))^β (`--beta` 4). Larger β trusts each run more, smaller β trusts the motion model more.
  - Runs below the threshold: no update, prediction only.
- **Output:** posterior mean (the star) and its covariance (dashed 1-σ ellipse); the posterior itself is the
  green shading in the video.
- `--smooth`: forward–backward smoothing, so each run uses the runs after it too (the rolling median in Step 2
  already does).

Why not (only) a Kalman filter:

- The per-run measurement is an argmax over clusters. When two areas score about the same, it flips between them.
  A Kalman filter assumes one Gaussian blob and averages the two. The grid keeps both until the next runs decide.
- The grid uses all 28 link scores every run, not only the 6 chosen ones.
- Per-run locations can only be cluster centroids, and these bunch in the middle of the cross ("pulled to the
  middle"). The grid can put the person anywhere near a link centre.

### kalman: constant-velocity Kalman filter

- State x, y, vx, vy. Initial velocity std `--walk-speed`. White-noise acceleration `--accel-std` (0.3 m/s²: the
  velocity can change by about the walking speed between runs, as the person turns often in this small area).
- Measurement = the per-run location of Step 5, std `--meas-std` (0.3 m) × √(threshold / cluster score), so
  strong runs count more.
- A measurement more than 3.5 σ (Mahalanobis) from the prediction is ignored; runs below the threshold are
  prediction only. `--smooth` = RTS smoother.
- Inherits the per-run location's bias to the middle. It only smooths it.

### Limit for both: one update every ~2 s

- At 1 mph the person moves ~0.9 m between runs, half the size of the cross. So the motion model can only
  say "close to where they were". It cannot predict the next position.
- Faster runs (fewer links per run, or one update per link at its own `ready` time) would help most.

## Step 7: Video

- **Left:** camera frame.
  - `vid_start_time.txt` = where time 0 is in the `.mov`.
  - The frame changes at every new MPP and holds between MPPs.
- **Right:** tag layout, rotated so T1 is straight above T4.
  - Link colour and width = link score. A link changes when its value is ready.
  - The chosen cluster is outlined in orange.
  - Star = location, with a trail of the last 4 locations. With a tracker: the star is the track, the grey circle
    is the per-run location, the dashed ellipse is the 1-σ uncertainty, the green shading is the grid posterior.
  - Both change when the run's last link is ready.
- **Bottom:** best-cluster score over time, the threshold line, and a cursor.
- Plays in real time (`--speed` changes it).

## Outputs

- `<exp>_person_location.mp4`: the video (grid tracker); `_kalman.mp4` / `_raw.mp4` with `--tracker kalman` /
  `none`, and the CSV likewise.
- `<exp>_person_location.csv`: one row per run with:
  - time, x, y (rotated layout, metres): the per-run location
  - track x, track y, track sd (m), when a tracker is on
  - cluster score, above threshold or not
  - share of the strongest link (1/6 = even)
  - the 6 chosen links
  - all 28 link scores

## Options

| Option | Default | Meaning |
|---|---|---|
| `--radius` | 0.4 | Max distance (m) of a link centre from its cluster's centre |
| `--agg` | gmean | Cluster score: gmean, median, trim (drops the top link), mean, rms |
| `--base-pct` | 99 | Percentile of `baseline915` used as the threshold |
| `--speed` | 1 | Video speed (1 = real time) |
| `--vid-start` | from file | Overrides `vid_start_time.txt` |
| `--rotate` | auto | Layout rotation; auto puts T1 above T4 |
| `--tracker` | grid | grid, kalman, or none (per-run location only) |
| `--smooth` | off | Use all runs (forward–backward / RTS) |
| `--beta` | 4 | grid: likelihood exponent |
| `--walk-speed` | 0.45 | walking speed (m/s, 1 mph): grid motion std, kalman initial velocity std |
| `--rho` | 0.3 | grid: weight width of the 6 nearest link centres (m) |
| `--reach` | 0.3 | grid: max distance from a link centre (m) |
| `--meas-std` | 0.3 | kalman: location std at the threshold score (m) |
| `--accel-std` | 0.3 | kalman: acceleration std (m/s²) |
| `--no-video` | off | Write only the CSV |

## Known limits

- **Pulled to the middle:** long links (e.g. T1–T3, T5–T7) all have their centres near the middle of the cross.
- **Threshold is too low:** almost every walking run passes (see Step 6).
- **Offline only:** the rolling median uses runs after the current one.
- **Slow:** one location every ~2 s (see Step 6b).
