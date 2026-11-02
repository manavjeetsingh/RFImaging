# Location logic (metal_wall experiments)

## Floor grid

- The floor is tiled with square tiles of **1 ft 6 in = 45.72 cm** (we use **46 cm**, `TILE_CM`).
- Tile **intersections** are labelled with integers **1 … 136**.
- Labels run **8 per column**: column 0 holds 1–8, column 1 holds 9–16, …, column 16 holds 129–136
  (17 columns × 8 rows = 136).
- Every column is numbered in the **same direction** (not serpentine): label 1, 9, 17, … are all on the
  same row.

## Label → grid index

For a numeric label `n` (1-based):

```
col = (n - 1) // 8      # 0 … 16
row = (n - 1) %  8      # 0 … 7
```

### `under-N` labels

Only 8 rows per column are labelled. `under-N` means the **next (unlabelled) intersection in the same
column, one row past N**:

```
col = (N - 1) // 8
row = (N - 1) %  8 + 1
```

In practice `N` is always the last label of a column (72, 80, 88 …), so `under-N` lands on row 8 —
the 9th, unlabelled row of that column. Example: `under-72` → col 8, row 8.

## Grid index → metric coordinates

```
x_cm = col * TILE_CM     # across columns
y_cm = row * TILE_CM     # along a column (direction of increasing label)
```

Origin `(0, 0)` is intersection **1**. `x` points from column 1–8 toward column 9–16; `y` points from
label 1 toward label 8. Z (height) isn't recorded in `positions.txt`.

## positions.txt format

One file per experiment directory (`metal_wall_data/<exp>/positions.txt`):

```
Tag, Position
1, 69
5, under-72
screen, 31
```

- First column: tag id (matches `TagK` in the CSV `Rx`/`Tx` columns), or `screen` for the metal
  screen/wall.
- Second column: a floor label (`69`) or an `under-N` label.

## Current layout (experiments 1 and 2)

The 15 tags form a 3 × 5 block that is the same in both experiments:

| column | rows 4,5,6,7,8 (labels)        | tags        |
|--------|--------------------------------|-------------|
| 8      | 69, 70, 71, 72, under-72       | 1, 2, 3, 4, 5 |
| 9      | 77, 78, 79, 80, under-80       | 15, 16, 17, 18, 19 |
| 10     | 85, 86, 87, 88, under-88       | 6, 7, 8, 9, 10 |

Only the screen moves: label **31** (col 3) in experiment 1, label **55** (col 6) in experiment 2.
That puts it 5 and 2 columns (230 cm and 92 cm) from the nearest tag column.

### Screen extent

The screen is a wall, not a point. The `screen` label gives its **column** (and its centre: row 6 is
the middle of tag rows 4–8). It stretches parallel to the tag columns across **all 5 tag rows**:

```
x      = screen_col * TILE_CM
y from = min tag row * TILE_CM  (row 4, 184 cm)
y to   = max tag row * TILE_CM  (row 8, 368 cm)
```

So it faces every tag in a row directly (`screen_segment()` in the notebook).

## Assumptions to confirm

1. Every column runs in the same direction (no serpentine numbering).
2. `under-N` is +1 row past N in N's column.
3. Tile pitch is 46 cm in both directions.
