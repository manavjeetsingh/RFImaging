"""Rewrite every experiment CSV under a data folder with the improved v1 processing:
robust MPP extraction (mpp_extraction_robust) + 70th-percentile channel aggregation (phase_calc_improved).

For each sub-directory of DATA_DIR holding an experiment CSV <name>.csv, writes <name>_improvedv1.csv next to it.
Every column that does not depend on MPP extraction / phase calculation is copied through byte-for-byte (the CSV is
read as text). Recomputed per row, from 'Voltages (mV)':

    Channel_<k>_voltages        edge-trimmed samples of channel k in the robust-fitted sweep (despiked)
    Channel_<k>_median          median of those samples (as before; no longer what the phase uses)
    Channel_<k>_p70             NEW: 70th percentile of those samples -- the value the phase calc uses
    Unidirectional Phase (deg)  \
    Unidirectional V             } cal_theta_et_al least squares on the Channel_<k>_p70 values
    Unidirectional beta         /
    Spikes Replaced             NEW: number of trace samples the despiker replaced

A row whose extraction or phase calc fails keeps its non-MPP columns, gets empty recomputed columns and is counted.
Attenuation (alpha) is not stored in these CSVs; downstream code computes it from Unidirectional beta / V.

    python reprocess_improvedv1.py ../roomsweeps_data
    python reprocess_improvedv1.py ../roomsweeps_data --jobs 4 --overwrite
"""
import argparse
import ast
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import mpp_extraction_robust as mppr
import phase_calc_improved as pci
import phase_calc_original as pco
from mpp_extraction import DEFAULT_CHANNELS
from phase_pipeline import fast_parse

SUFFIX = "_improvedv1"
PERCENTILE = 70.0
CHUNK_ROWS = 2000

VOLT_COL, FREQ_COL = "Voltages (mV)", "Frequency (MHz)"
PHASE_COLS = ["Unidirectional Phase (deg)", "Unidirectional V", "Unidirectional beta"]
SPIKE_COL = "Spikes Replaced"


def experiment_csvs(data_dir):
    """The one experiment CSV of each sub-directory (ignoring previous outputs)."""
    out = []
    for sub in sorted((p for p in Path(data_dir).iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        csvs = [c for c in sub.glob("*.csv") if not c.stem.endswith(SUFFIX)]
        if len(csvs) != 1:
            print(f"  skip {sub.name}: {len(csvs)} experiment CSVs ({[c.name for c in csvs]})")
            continue
        out.append(csvs[0])
    return out


def channels_from_header(cols):
    ch = sorted(int(c.split("_")[1]) for c in cols if c.startswith("Channel_") and c.endswith("_median"))
    return ch or DEFAULT_CHANNELS


def output_columns(cols, channels):
    """Input column order, with Channel_<k>_p70 after each Channel_<k>_median and Spikes Replaced at the end."""
    out = []
    for c in cols:
        out.append(c)
        if c.startswith("Channel_") and c.endswith("_median"):
            out.append(c.replace("_median", "_p70"))
    missing = [f"Channel_{k}_{s}" for k in channels for s in ("voltages", "median", "p70") if f"Channel_{k}_{s}" not in out]
    return out + missing + ([SPIKE_COL] if SPIKE_COL not in out else [])


def grid_kwargs(row):
    dwell = row.get("Dwell (samples)", "")
    if dwell not in ("", None) and np.isfinite(float(dwell)):
        return {"dwell_samples": float(dwell)}
    return {"transport": row["Connection"]}


_LEADING_FLOAT = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_trace(text):
    """fast_parse, falling back to the leading float of each token for transfer-corrupted samples
    (e.g. '627.069504:447395' in roomsweeps 6 -> 627.069504; the despiker handles anything still off)."""
    try:
        return fast_parse(text)
    except ValueError:
        vals = []
        for tok in text.strip("[] \n").split(","):
            m = _LEADING_FLOAT.match(tok.strip())
            if m is None:
                raise
            vals.append(float(m.group()))
        return np.array(vals)


def process_row(row, channels, cfg, tag_mac):
    """{column: text} of the recomputed columns for one row (all text so the CSV round-trips like the input)."""
    v = parse_trace(row[VOLT_COL])
    medians, samples, _, spikes = mppr.segment_capture_robust(v, channels, **grid_kwargs(row))
    theta, V, beta, p70 = pci.cal_theta_et_al_percentile(samples, tag_mac[row["Rx"]], tag_mac[row["Tx"]], cfg,
                                                         float(row[FREQ_COL]) * 1e6, q=PERCENTILE)
    out = {PHASE_COLS[0]: repr(theta), PHASE_COLS[1]: repr(V), PHASE_COLS[2]: repr(beta),
           SPIKE_COL: str(int(spikes.sum()))}
    for k in channels:
        out[f"Channel_{k}_voltages"] = str(samples[k].tolist())
        out[f"Channel_{k}_median"] = repr(medians[k])
        out[f"Channel_{k}_p70"] = repr(p70[k])
    return out


def process_csv(csv, overwrite=False):
    csv = Path(csv)
    dst = csv.with_name(f"{csv.stem}{SUFFIX}.csv")
    if dst.exists() and not overwrite:
        return f"{csv.parent.name}: {dst.name} exists, skipped (--overwrite to redo)"
    t0 = time.time()
    cfg, tag_mac = pco.load_hw_config(), pco.load_tag_mac_mapping()
    cols = list(pd.read_csv(csv, nrows=0).columns)
    channels = channels_from_header(cols)
    out_cols = output_columns(cols, channels)
    recomputed = PHASE_COLS + [SPIKE_COL] + [f"Channel_{k}_{s}" for k in channels for s in ("voltages", "median", "p70")]

    tmp = dst.with_name(dst.name + ".tmp")
    n = failed = 0
    first_errors = []
    reader = pd.read_csv(csv, dtype=str, keep_default_na=False, chunksize=CHUNK_ROWS)
    with open(tmp, "w", newline="") as fh:
        for i, chunk in enumerate(reader):
            rows = []
            for r in chunk.to_dict("records"):
                try:
                    r.update(process_row(r, channels, cfg, tag_mac))
                except Exception as e:
                    failed += 1
                    if len(first_errors) < 3:
                        first_errors.append(f"{r.get('Rx')} <- {r.get('Tx')} run {r.get('Run Exp Num')}: "
                                            f"{type(e).__name__}: {e}")
                    r.update({c: "" for c in recomputed})
                rows.append(r)
            pd.DataFrame(rows, columns=out_cols).to_csv(fh, index=False, header=(i == 0))
            n += len(rows)
    os.replace(tmp, dst)
    msg = f"{csv.parent.name}: {n} rows -> {dst.name} in {time.time() - t0:.0f}s, {failed} failed"
    return msg + "".join(f"\n      {e}" for e in first_errors)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path, help="folder whose sub-directories each hold one experiment CSV")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="experiments processed in parallel")
    ap.add_argument("--overwrite", action="store_true", help=f"redo experiments that already have a {SUFFIX} CSV")
    args = ap.parse_args()

    csvs = experiment_csvs(args.data_dir)
    print(f"{len(csvs)} experiments in {args.data_dir}, {args.jobs} jobs")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(process_csv, c, args.overwrite): c for c in csvs}
        for f in as_completed(futs):
            try:
                print("  " + f.result(), flush=True)
            except Exception as e:
                print(f"  {futs[f].parent.name}: FAILED {type(e).__name__}: {e}", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
