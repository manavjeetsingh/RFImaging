# Room sweep experiment: convention, setup and room

Applies to everything in `roomsweeps_data/`, `roomsweeps_processing.ipynb`, `variance_processing.ipynb`,
`roomsweeps_wall_distance.py` and `roomsweeps_wall_distance_fit.py`. Written so that the experiment can be simulated
and the simulated data can go through the same pipeline. Numbers marked *(scan)* were measured from the 3-D room scan;
items under "Not recorded" are unknown and have to be assumed.

## 1. Floor grid and labels

- The floor has square tiles of **1 ft 6 in = 45.72 cm** (code uses `TILE_CM = 46`). Measured in the scan, the
  spacing is 45.3 ± 1.6 cm along Direction 0 and 45.1 ± 1.7 cm along Direction 1 *(scan)*.
- Tile **intersections** are labelled **1–136**: 17 columns × 8 rows, 8 labels per column, every column numbered
  the same way (not serpentine). Same as `location_logic.md`.

```
col = (n - 1) // 8      # 0 … 16
row = (n - 1) %  8      # 0 … 7      (row 0 = "row 1" in the sketch below)
```

```
            col 0   col 1  col 2 | col 3 … col 13 | col 14 col 15 col 16
 row 1        1      9      17   |  25  …  105    |  113    121    129    ┐
 row 2        2                  |                |                       │ Edge10
 row 3        3                  |                |                       │
 row 4        4                  |                |                132    ┘
 row 5        5                  |                |                       ┐
 row 6        6                  |                |                       │ Center1
 row 7        7                  |                |                       │
 row 8        8      16      24  |  32  …  112    |  120    128    136    ┘
             └──── Edge02 ─────┘ └── Center0 ───┘ └───── Edge01 ──────┘
```

- `under-N` (from the metal-wall experiments) is the unlabelled 9th intersection one row past N in N's column. It
  doesn't occur in the room sweeps.

### Directions

| Direction | Moves along | Index that changes | Example |
|-----------|-------------|--------------------|---------|
| **Direction 0** | across columns | column (`(n-1)//8`) | 1 → 9 → 17 → … → 129 |
| **Direction 1** | down a column | row (`(n-1)%8`) | 1 → 2 → … → 8 |

### Regions

Region names are `Edge<direction><end>` or `Center<direction>`.

| Region | Direction | Columns / rows | Labels |
|--------|-----------|----------------|--------|
| **Edge02** | 0 | columns 0–2 | 1–24 |
| **Center0** | 0 | columns 3–13 | 25–112 |
| **Edge01** | 0 | columns 14–16 | 113–136 |
| **Edge10** | 1 | rows 1–4 | n with (n-1)%8 in 0–3 (1–4, 9–12, …, 129–132) |
| **Center1** | 1 | rows 5–8 | n with (n-1)%8 in 4–7 (5–8, 13–16, …, 133–136) |

Every label is in exactly one direction-0 region and one direction-1 region, e.g. label 50 → (Center0, Edge10).
`label_regions(n)` in the notebooks returns this pair. Read from the whiteboard sketch: Center0 runs from 32 (col 3)
to 112 (col 13); Direction 1 has only one edge (Edge10), no "Edge11".

## 2. Room geometry *(scan)*

Source: `roomscans/merged.ply` (5 merged scans `mapping_2214_{1..5}.ply`, room OCS 2214), metres, z up, floor ≈ 0.
Label centres: `roomscans/merged_markers.csv` (`index, x, y, z, scene`; marker z = 0.015–0.076 m, i.e. on the floor).

Grid corners in the scan frame:

| Label | grid (col, row) | x (m) | y (m) |
|-------|-----------------|-------|-------|
| 1 | (0, 0) | −1.204 | −3.455 |
| 8 | (0, 7) | 1.235 | −1.475 |
| 129 | (16, 0) | −5.937 | 2.031 |
| 136 | (16, 7) | −3.478 | 4.072 |

- Direction 0 (1 → 129): 7.245 m, heading 130.8° in the scan xy-plane.
- Direction 1 (1 → 8): 3.142 m, heading 39.1°. The two directions are 91.7° apart in the scan (nominally 90°; the
  difference is scan distortion).

### Walls

Found by `find_walls()` in `roomsweeps_wall_distance.py`: scan points 1.0–2.0 m above the floor beyond each edge
of the grid, median offset around the densest 10 cm bin. "Normal" points from the wall into the room.

| Wall | Parallel to | Distance from nearest grid line | Point on wall (x, y) | Normal |
|------|-------------|--------------------------------|----------------------|--------|
| Edge02 wall, beyond column 0 | Direction 1 | 64.3 cm from column 0 | (−0.799, −3.954) | (−0.630, 0.776) |
| Edge01 wall, beyond column 16 | Direction 1 | 58.8 cm from column 16 | (−6.313, 2.483) | (0.639, −0.769) |
| Row-1 wall, beyond row 1 | Direction 0 | 35.8 cm from row 1 | (−1.476, −3.689) | (0.757, 0.653) |
| Row-8 wall, beyond row 8 | Direction 0 | 296.6 cm from row 8 | (3.495, 0.445) | (−0.762, −0.647) |

So the room is a rectangle of about:

- **8.48 m** along Direction 0: 0.643 + 7.245 (columns 0 → 16) + 0.588.
- **6.47 m** along Direction 1: 0.358 + 3.142 (rows 1 → 8) + 2.966.
- **~3.5 m** high: 99.5th percentile of scan z = 3.51 m.

The scan's own extent in the grid frame agrees: −0.79 … 7.87 m along Direction 0 and −0.47 … 6.19 m along
Direction 1, measured from label 1. The scan also has furniture; z 0.3–2.0 m is drawn as obstacles in the notebooks.
The walls were detected automatically, so a long piece of furniture along a wall could have been picked instead of
the wall; check against the scan if it matters.

## 3. Tags and sweep layout

- **15 tags**, three groups of five: **T1–T5**, **T6–T10**, **T15–T19** (T11–T14 aren't used).
- Each experiment puts the three groups in **three adjacent grid columns**, one group per column, tags on
  **rows 1–5** of that column (grid rows 0–4). Within a group the lowest tag number is on row 1.
- **Leapfrog sweep:** experiment *n* (1–15) uses columns *n−1*, *n*, *n+1*. From one experiment to the next, the
  group in the lowest column jumps to the column after the highest, so the group order rotates by one each time.
  Neighbouring experiments therefore share two columns, and every column except 0 and 16 is measured in 2–3
  experiments.
- Positions are in `roomsweeps_data/<exp>/positions.txt` for all 15 experiments (`Tag, Position`, one tag per line,
  position = floor label). `infer_positions(n)` reproduces every file exactly.

| Exp | Column n−1 | Column n | Column n+1 |
|-----|-------------|---------|-------------|
| 1 | T1–T5 @ 1–5 | T6–T10 @ 9–13 | T15–T19 @ 17–21 |
| 2 | T6–T10 @ 9–13 | T15–T19 @ 17–21 | T1–T5 @ 25–29 |
| 3 | T15–T19 @ 17–21 | T1–T5 @ 25–29 | T6–T10 @ 33–37 |
| 4 | T1–T5 @ 25–29 | T6–T10 @ 33–37 | T15–T19 @ 41–45 |
| 5 | T6–T10 @ 33–37 | T15–T19 @ 41–45 | T1–T5 @ 49–53 |
| 6 | T15–T19 @ 41–45 | T1–T5 @ 49–53 | T6–T10 @ 57–61 |
| 7 | T1–T5 @ 49–53 | T6–T10 @ 57–61 | T15–T19 @ 65–69 |
| 8 | T6–T10 @ 57–61 | T15–T19 @ 65–69 | T1–T5 @ 73–77 |
| 9 | T15–T19 @ 65–69 | T1–T5 @ 73–77 | T6–T10 @ 81–85 |
| 10 | T1–T5 @ 73–77 | T6–T10 @ 81–85 | T15–T19 @ 89–93 |
| 11 | T6–T10 @ 81–85 | T15–T19 @ 89–93 | T1–T5 @ 97–101 |
| 12 | T15–T19 @ 89–93 | T1–T5 @ 97–101 | T6–T10 @ 105–109 |
| 13 | T1–T5 @ 97–101 | T6–T10 @ 105–109 | T15–T19 @ 113–117 |
| 14 | T6–T10 @ 105–109 | T15–T19 @ 113–117 | T1–T5 @ 121–125 |
| 15 | T15–T19 @ 113–117 | T1–T5 @ 121–125 | T6–T10 @ 129–133 |

"T1–T5 @ 1–5" means T1@1, T2@2, …, T5@5. Rows 6–8 of the grid (labels with (n-1)%8 = 5–7) are never occupied,
so tags are only in Edge10 (rows 1–4) and the first row of Center1 (row 5).

## 4. Measurement protocol

One CSV per experiment: `roomsweeps_data/<exp>/mpp_testing_multifeq_775-995MHz_3runs_<timestamp>.csv` (~500 MB,
most of it the `Channel_*_voltages` columns). The pipeline reads the **first `*.csv` directly in the experiment
folder**, so put nothing else with a `.csv` extension there (derived files go in subfolders, e.g. `fits/`).

- **Every ordered pair** of the 15 tags is measured: Rx = A, Tx = B and Rx = B, Tx = A, so 15 × 14 = 210 ordered
  pairs (105 unordered).
- **22 frequencies:** 775, 785, 795, 805, then 825–995 MHz in 10 MHz steps. **815 MHz is missing.**
- **3 runs** (`Run Exp Num` 0, 1, 2), 1 repetition each (`MPP Repetition` = 1).
- 210 × 22 × 3 = **13 860 rows** per experiment, one row per (Rx, Tx, frequency, run). One experiment takes about 10 min.
- Wireless connection, sample rate ≈ 40 kHz, dwell ≈ 120 samples, Esync SNR 10–37 dB (median 24 dB).

### CSV columns

| Column | Meaning / typical values |
|--------|--------------------------|
| `Rx`, `Tx` | `Tag<k>` |
| `Frequency (MHz)` | integer, see above |
| `Run Exp Num` | 0, 1, 2 |
| `MPP Repetition` | 1 |
| `Unidirectional Phase (deg)` | Θ̂, in (−180, 180] |
| `Unidirectional V` | V̂, 3–176 (median 48) |
| `Unidirectional beta` | β̂, 0.004–7.5 (median 0.73) |
| `MPP Start Time (s)`, `MPP Stop Time (s)`, `Time Taken (s)` | Unix time; 4.5–44 s per row |
| `Connection`, `Collection Type`, `Sample Rate (Hz)`, `Dwell (samples)` | `wireless`, `multiple`, ≈40 kHz, ≈120 |
| `Esync Rho`, `Esync SNR (dB)` | sync quality |
| `Channel_{1,3,4,6,7,8}_median` | per-channel medians |
| `Channel_*_voltages` (7 columns) | raw traces, **skipped by the pipeline** |

A simulation only needs `Rx, Tx, Frequency (MHz), Run Exp Num, Unidirectional Phase (deg), Unidirectional V,
Unidirectional beta` for the pipeline to run.

### Bidirectional phase and attenuation (what the analysis uses)

For a tag pair 1 < 2 (by tag number), row 1 has Rx = T1 / Tx = T2 and row 2 has Rx = T2 / Tx = T1, matched per
frequency and run:

$$\hat\alpha_{12} = \tfrac12\left(\frac{\hat\beta_1}{\hat V_2} + \frac{\hat\beta_2}{\hat V_1}\right) \qquad
\hat\theta_{12} = \left(\frac{\hat\Theta_1 + \hat\Theta_2}{2}\right) \bmod \pi$$

with Θ̂ converted to radians. θ̂ is only defined mod π.

Measured spread *(all 15 experiments)*:

- θ̂ varies little between the 3 runs: circular sd median 1.8°, p75 4.7°, p95 25°.
- α̂ is 0.004–0.042 (5th–95th percentile, median 0.015), with run-to-run coefficient of variation median 0.04,
  p95 0.34.

## 5. Analysis targets (what simulated data is used for)

- **Links** (`roomsweeps_wall_distance*.py`): tag pairs whose labels are exactly *k* tiles apart along one grid
  direction (k = 1–4, i.e. 46/92/138/184 cm). The target is the perpendicular distance of the link's midpoint from
  the closer of the two walls parallel to it: Edge02/Edge01 walls for Direction 1 links, Row-1/Row-8 walls for
  Direction 0 links. Range ≈ 36–425 cm.
- Per experiment, this gives 9 + 5 = 14 links of 2 tiles (and 148 / 126 / 34 / 17 distinct label pairs of 1–4
  tiles over all experiments).
- **Phase fit** (`variance_processing.ipynb`): θ̂ per link, circular mean over runs, unwrapped along frequency
  (period π), least-squares line over 800–960 MHz (15 frequencies). The residuals Δ and a sine fit to them are
  saved to `roomsweeps_data/<exp>/fits/phase_fit.csv`.

## 6. Not recorded (assume and state it when simulating)

- Tag height above the floor, antenna type, polarisation and orientation (positions are floor labels only).
- Transmit power and the physical meaning/units of `Unidirectional V` and `beta`.
- Wall, floor and ceiling materials, and the furniture positions (only visible in the scan).
- Whether people were in the room during the sweeps.
