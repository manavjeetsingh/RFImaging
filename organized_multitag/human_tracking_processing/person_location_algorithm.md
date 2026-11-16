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
- Each run is decided on its own. There is no memory and no smoothing between runs.

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

## Step 7: Video

- **Left:** camera frame.
  - `vid_start_time.txt` = where time 0 is in the `.mov`.
  - The frame changes at every new MPP and holds between MPPs.
- **Right:** tag layout, rotated so T1 is straight above T4.
  - Link colour and width = link score. A link changes when its value is ready.
  - The chosen cluster is outlined in orange.
  - Star = location, with a trail of the last 4 locations.
  - Both change when the run's last link is ready.
- **Bottom:** best-cluster score over time, the threshold line, and a cursor.
- Plays in real time (`--speed` changes it).

## Outputs

- `<exp>_person_location.mp4`: the video.
- `<exp>_person_location.csv`: one row per run with:
  - time, x, y (rotated layout, metres)
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
| `--no-video` | off | Write only the CSV |

## Known limits

- **Pulled to the middle:** long links (e.g. T1–T3, T5–T7) all have their centres near the middle of the cross.
- **Threshold is too low:** almost every walking run passes (see Step 6).
- **Offline only:** the rolling median uses runs after the current one.
- **Slow:** one location every ~2 s.
