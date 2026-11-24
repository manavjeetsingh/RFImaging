"""Plot every MPP trace of a data CSV with the windows found by a chosen MPP-extraction algorithm, into one PDF.

For each unordered tag pair (A, B) and run, a row holds both directions side by side (A <- B left, B <- A right),
ROWS_PER_PAGE runs per page. Each panel: ADC samples, the fitted channel boundaries (vertical lines), the
per-channel medians (red, labelled), the area outside the fitted sweep shaded, samples the algorithm replaced
(red ▼ at the top edge), and the unidirectional phase from the chosen phase calc in the title.

The algorithms are the ones registered in phase_pipeline.py (MPP_EXTRACTORS, PHASE_CALCS); an extractor must
return "windows" to be plottable.

    python plot_traces.py --data ../multitag_organized_exciter_comparison_data/bladerf --mpp robust
    python plot_traces.py --data a.csv b_folder --mpp original --runs 0-4 --pairs Tag4-Tag8,Tag1-Tag16

Output: out/traces_<mpp>__<phase>/<dataset folder>_<csv stem>.pdf
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages               # noqa: E402

import phase_pipeline as pp                                        # noqa: E402

ROWS_PER_PAGE = 3


def parse_pairs(s):
    if not s:
        return None
    return {tuple(sorted(p.split("-"), key=pp.tag_num)) for p in s.split(",")}


def run_mpp(row, mpp_fn, phase_fn, channels, cfg, tag_mac):
    v = np.asarray(row[pp.VOLT_COL], dtype=float)
    try:
        res = mpp_fn(v, pp.row_meta(row), channels)
        phase = phase_fn(res, tag_mac[row["Rx"]], tag_mac[row["Tx"]], cfg,
                         float(row[pp.FREQ_COL]) * 1e6)
        return v, res, phase, None
    except Exception as e:                                         # still plot the raw trace
        return v, None, np.nan, f"{type(e).__name__}: {e}"


def draw(ax, row, v, res, phase, err, label):
    rate = row.get("Sample Rate (Hz)")
    dt = 1.0 / rate if rate is not None and np.isfinite(rate) and rate > 0 else 1.0
    t = np.arange(len(v)) * dt
    ax.plot(t, v, ".", markersize=1.5, color="#2a78d6", label="ADC samples", rasterized=True)

    lo, hi = np.percentile(v, [0.5, 99.5])                        # a single spike would flatten the trace
    margin = max((hi - lo) * 0.25, 0.05)
    ax.set_ylim(lo - margin, hi + margin)

    if res is not None and "windows" in res:
        wins = res["windows"]
        edges = sorted({b for w in wins.values() for b in w})
        ax.axvspan(t[0], edges[0] * dt, color="#e1e0d9", alpha=0.6, label="outside fitted sweep")
        ax.axvspan(edges[-1] * dt, t[-1], color="#e1e0d9", alpha=0.6)
        for e in edges:
            ax.axvline(e * dt, color="#4a3aa8", linewidth=1.0)
        for ch, (a, b) in wins.items():
            m = res["medians"][ch]
            ax.hlines(m, a * dt, b * dt, color="#e34948", linewidth=2, zorder=3)
            ax.annotate(f"ch{ch}", ((a + b) / 2 * dt, m), textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=6, color="#52514e")
    spikes = None if res is None else res.get("spikes")
    if spikes is not None and np.any(spikes):
        ax.plot(t[spikes], np.full(int(np.sum(spikes)), hi + margin * 0.8), "v", color="#e34948",
                markersize=5, label=f"replaced ({int(np.sum(spikes))})", zorder=4)

    head = (f"[{label}] {row['Rx']} <- {row['Tx']} | {row[pp.FREQ_COL]:g} MHz | run {row[pp.RUN_COL]} | "
            f"rep {row[pp.REP_COL]} | ")
    ax.set_title(head + (f"phase {phase:.2f} deg" if err is None else "extraction FAILED"),
                 fontsize=8, color="#000000" if err is None else "#e34948")
    if err is not None:
        ax.text(0.5, 0.5, err, transform=ax.transAxes, ha="center", fontsize=7, color="#e34948", wrap=True)
    ax.set_xlabel("Time [s]" if dt != 1.0 else "Sample", fontsize=7)
    ax.set_ylabel("ADC out [mV]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")


def plot_csv(csv, mpp_name, phase_name, runs, pairs, out_dir, cfg, tag_mac):
    mpp_fn, phase_fn = pp.MPP_EXTRACTORS[mpp_name], pp.PHASE_CALCS[phase_name]
    df = pp.load_traces(csv)
    channels = pp.channels_of(csv)
    if runs is not None:
        df = df[df[pp.RUN_COL].isin(runs)]
    df = df.assign(A=[min(r, t, key=pp.tag_num) for r, t in zip(df["Rx"], df["Tx"])],
                   B=[max(r, t, key=pp.tag_num) for r, t in zip(df["Rx"], df["Tx"])])
    if pairs is not None:
        df = df[[(a, b) in pairs for a, b in zip(df["A"], df["B"])]]
    df = df.assign(_a=df["A"].map(pp.tag_num), _b=df["B"].map(pp.tag_num))

    # one figure row per (pair, freq, run, rep): left A <- B, right B <- A
    lines = []
    for (_, _, freq, run, rep), g in df.groupby(["_a", "_b", pp.FREQ_COL, pp.RUN_COL, pp.REP_COL], sort=True):
        a, b = g["A"].iloc[0], g["B"].iloc[0]
        lines.append([g[(g["Rx"] == a) & (g["Tx"] == b)], g[(g["Rx"] == b) & (g["Tx"] == a)]])

    label = f"{Path(csv).parent.name if Path(csv).is_file() else Path(csv).name} / {mpp_name}"
    pdf_path = out_dir / f"{pp.data_csv(csv).parent.name}_{pp.data_csv(csv).stem}.pdf"
    t0, n_fail = time.time(), 0
    print(f"  {pdf_path.name}: {len(df)} MPPs -> {len(lines)} rows, {-(-len(lines) // ROWS_PER_PAGE)} pages")
    with PdfPages(pdf_path) as pdf:
        for p0 in range(0, len(lines), ROWS_PER_PAGE):
            page = lines[p0:p0 + ROWS_PER_PAGE]
            fig, axes = plt.subplots(ROWS_PER_PAGE, 2, figsize=(16, 3.6 * ROWS_PER_PAGE), squeeze=False)
            for i, line in enumerate(page):
                for j, sub in enumerate(line):
                    ax = axes[i, j]
                    if sub.empty:
                        ax.axis("off")
                        ax.text(0.5, 0.5, "direction missing", ha="center", transform=ax.transAxes)
                        continue
                    row = sub.iloc[0]
                    v, res, phase, err = run_mpp(row, mpp_fn, phase_fn, channels, cfg, tag_mac)
                    n_fail += err is not None
                    draw(ax, row, v, res, phase, err, label)
            for ax in axes[len(page):].flat:
                ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            done = p0 // ROWS_PER_PAGE + 1
            if done % 100 == 0:
                print(f"    page {done}  ({time.time() - t0:.0f}s)")
    print(f"    wrote {pdf_path} in {time.time() - t0:.0f}s ({n_fail} failed extractions)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, type=Path, help="data CSV(s) or dataset folder(s)")
    ap.add_argument("--mpp", default="original", help=f"MPP extraction algorithm: {list(pp.MPP_EXTRACTORS)}")
    ap.add_argument("--phase", default="original", help=f"phase calc (for the titles): {list(pp.PHASE_CALCS)}")
    ap.add_argument("--runs", help="runs to plot, e.g. 0-4 or 0,2,4 (default: all)")
    ap.add_argument("--pairs", help="only these unordered pairs, e.g. Tag4-Tag8,Tag1-Tag16 (default: all)")
    ap.add_argument("--out", type=Path, help="output dir (default: out/traces_<mpp>__<phase>)")
    args = ap.parse_args()
    if args.mpp not in pp.MPP_EXTRACTORS:
        ap.error(f"--mpp {args.mpp!r} unknown, choose from {list(pp.MPP_EXTRACTORS)}")
    if args.phase not in pp.PHASE_CALCS:
        ap.error(f"--phase {args.phase!r} unknown, choose from {list(pp.PHASE_CALCS)}")

    out = args.out or pp.OUT_ROOT / f"traces_{args.mpp}__{args.phase}"
    out.mkdir(parents=True, exist_ok=True)
    cfg, tag_mac = pp.pco.load_hw_config(), pp.pco.load_tag_mac_mapping()
    for data in args.data:
        plot_csv(data, args.mpp, args.phase, pp.parse_runs(args.runs), parse_pairs(args.pairs), out, cfg, tag_mac)


if __name__ == "__main__":
    main()
