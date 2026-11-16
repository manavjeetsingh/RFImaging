"""Video of how much every link differs from the empty-room baseline, over time.

For one single-frequency (915 MHz) walking experiment, every bidirectional measurement of a link (T1, T2) gives
θ̂ and α̂ (as in human_tracking_processing.ipynb). Each measurement is compared with the same link in the baseline
experiment, as in human_detection_processing.ipynb ("Distance from the baseline, every link"):

    d = || [ Δθ̂ / s_θ ,  Δα̂ / s_α ] ||

- the baseline value of a link is the median over its baseline runs (θ̂ with the mod-π median)
- s_θ, s_α are the spreads (standard deviation) of the baseline values over all links, so θ̂ and α̂ count equally
- Δθ̂ is taken the short way round mod π (at most π/2)

With one frequency, the vector of a link has 2 entries (θ̂, α̂) instead of 2 × n_freqs.

The video runs in real time (or at --speed): a link changes in the video at the moment its measurement completes,
i.e. at the stop time of the later of its two MPPs, measured from the start of the first MPP of the experiment.
Until then, the link keeps its previous value. A link that has just been updated is outlined briefly.

Ground truth: the camera recording (*.mov in the experiment folder) is shown next to the layout. vid_start_time.txt
in the experiment folder gives the time in the recording at which the experiment starts (the first MPP), as
seconds, m:ss or h:mm:ss (fractions allowed, e.g. 0:05 or 0:05.4); --vid-start overrides it. The camera panel
shows the recording at the stop time of the latest MPP, so it changes with every new MPP and holds between them.

    python link_distance_video.py walking915_1
    python link_distance_video.py walking915_2 --baseline baseline915 --fps 30 --speed 2
    python link_distance_video.py walking915_1 --vid-start 0:07
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # organized_multitag
DATA_DIR = ROOT / "human_tracking_data"
MARKERS_PATH = ROOT / "roomscans" / "merged_markers.csv"

# one colour per arm of the tag cross: tags 1-4 down column 8, tags 5-8 across row 5 (as in the notebooks)
TAG_GROUPS = {"T1–T4 (column 8)": range(1, 5), "T5–T8 (row 5)": range(5, 9)}
GROUP_COLORS = ["#2a78d6", "#eb6834"]
HIGHLIGHT = "#f2a900"
FLASH_S = 0.4                                        # how long a link stays outlined after an update (s)


# ---------------------------------------------------------------- data

def tag_num(name):
    return int(str(name).replace("Tag", ""))


def load_csv(exp_dir):
    f = next(Path(exp_dir).glob("*.csv"))
    cols = pd.read_csv(f, nrows=0).columns
    return pd.read_csv(f, usecols=[c for c in cols if "voltages" not in c.lower()])


def bidirectional(df):
    # one row per (link, freq, run): THETA, alpha, and the times of its two MPPs (s from the first MPP)
    keep = ["Rx", "Tx", "Frequency (MHz)", "Run Exp Num", "MPP Start Time (s)", "MPP Stop Time (s)",
            "Unidirectional Phase (deg)", "Unidirectional V", "Unidirectional beta"]
    d = df[keep].copy()
    t0 = d["MPP Start Time (s)"].min()
    d["start"] = d["MPP Start Time (s)"] - t0
    d["stop"] = d["MPP Stop Time (s)"] - t0
    fwd = d[d["Rx"].map(tag_num) < d["Tx"].map(tag_num)]
    rev = d[d["Rx"].map(tag_num) > d["Tx"].map(tag_num)]
    m = fwd.merge(rev, left_on=["Rx", "Tx", "Frequency (MHz)", "Run Exp Num"],
                  right_on=["Tx", "Rx", "Frequency (MHz)", "Run Exp Num"],
                  suffixes=("_a", "_b"), validate="one_to_one")
    th_a = np.deg2rad(m["Unidirectional Phase (deg)_a"])
    th_b = np.deg2rad(m["Unidirectional Phase (deg)_b"])
    m["THETA"] = ((th_a + th_b) / 2) % np.pi
    m["alpha"] = 0.5 * (m["Unidirectional beta_a"] / m["Unidirectional V_b"]
                        + m["Unidirectional beta_b"] / m["Unidirectional V_a"])
    m["ready"] = np.maximum(m["stop_a"], m["stop_b"])          # the bidirectional value exists from here on
    m = m.rename(columns={"Rx_a": "T1", "Tx_a": "T2"})
    return m[["T1", "T2", "Frequency (MHz)", "Run Exp Num", "start_a", "start_b", "ready", "THETA", "alpha"]]


def median_mod_pi(x):
    x = np.asarray(x, float)
    c = (np.angle(np.mean(np.exp(2j * x))) / 2) % np.pi
    return float((c + np.median((x - c + np.pi / 2) % np.pi - np.pi / 2)) % np.pi)


def link_distances(exp_m, base_m):
    # distance of every measurement in exp_m from its link's baseline, scaled as in human_detection_processing
    freqs = exp_m["Frequency (MHz)"].unique()
    assert len(freqs) == 1, f"expected one frequency, got {sorted(freqs)}"
    base_m = base_m[base_m["Frequency (MHz)"] == freqs[0]]
    assert len(base_m), f"baseline has no {freqs[0]} MHz measurements"
    base = base_m.groupby(["T1", "T2"]).agg(theta0=("THETA", median_mod_pi), alpha0=("alpha", "median"))
    s_theta, s_alpha = base["theta0"].std(ddof=0), base["alpha0"].std(ddof=0)
    m = exp_m.join(base, on=["T1", "T2"], how="inner")
    d_theta = (m["THETA"] - m["theta0"] + np.pi / 2) % np.pi - np.pi / 2
    m["dist"] = np.hypot(d_theta / s_theta, (m["alpha"] - m["alpha0"]) / s_alpha)
    return m.sort_values("ready").reset_index(drop=True), s_theta, s_alpha


def tag_positions(exp_dir):
    # tag number (str) -> x, y of its floor label
    markers = pd.read_csv(MARKERS_PATH).set_index("index")
    pos = pd.read_csv(next(Path(exp_dir).glob("positions.txt*")), skipinitialspace=True, dtype=str).dropna()
    tags = pos.iloc[:, 0].str.strip()
    labels = pos.iloc[:, 1].str.strip().astype(int)
    return pd.DataFrame({"x": markers.loc[labels, "x"].to_numpy(), "y": markers.loc[labels, "y"].to_numpy(),
                         "label": labels.to_numpy()}, index=tags), markers


def parse_clock(text):
    # "5", "5.4", "0:05", "1:02:03.5" -> seconds
    secs = 0.0
    for part in text.strip().split(":"):
        secs = secs * 60 + float(part)
    return secs


def video_start(exp_dir, override=None):
    # time in the camera recording at which the experiment starts, or None if it is not known
    if override is not None:
        return parse_clock(override)
    f = Path(exp_dir) / "vid_start_time.txt"
    return parse_clock(f.read_text()) if f.exists() else None


class CameraReader:
    # decodes the recording once, front to back, from the experiment start; frame(t) returns the frame at
    # experiment time t (s from the first MPP). t must not decrease between calls.
    def __init__(self, path, start, width=810):
        info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                               "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(path)],
                              capture_output=True, text=True, check=True).stdout.strip().split(",")
        w, h = int(info[0]), int(info[1])
        num, den = info[2].split("/")
        self.fps = float(num) / float(den)
        self.w, self.h = width, int(round(h * width / w / 2)) * 2
        self.proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(path),
             "-vf", f"scale={self.w}:{self.h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE)
        self.index, self.current = -1, np.zeros((self.h, self.w, 3), np.uint8)

    def frame(self, t):
        target = max(int(round(t * self.fps)), 0)
        n = self.w * self.h * 3
        while self.index < target:
            buf = self.proc.stdout.read(n)
            if len(buf) < n:                                       # end of the recording: keep the last frame
                break
            self.current = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
            self.index += 1
        return self.current

    def close(self):
        self.proc.stdout.close()
        self.proc.kill()
        self.proc.wait()


# ---------------------------------------------------------------- drawing

def tag_color(tag):
    for color, tags in zip(GROUP_COLORS, TAG_GROUPS.values()):
        if int(tag) in tags:
            return color
    return "#333333"


def link_curvature(ref, t1, t2):
    # links between tags on the same arm overlap the shorter ones, so bend them by how many tags they skip
    a = ref.loc[str(tag_num(t1)), ["x", "y"]].to_numpy(float)
    b = ref.loc[str(tag_num(t2)), ["x", "y"]].to_numpy(float)
    skipped = 0
    for tag, r in ref.iterrows():
        if tag in (str(tag_num(t1)), str(tag_num(t2))):
            continue
        p = r[["x", "y"]].to_numpy(float)
        u = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
        if 0 < u < 1 and np.linalg.norm(a + u * (b - a) - p) < 0.08:
            skipped += 1
    return 0.0 if skipped == 0 else 0.22 * skipped


def rotate_layout(ref, markers, degrees):
    # rotate the tags and floor labels about the centre of the tags, counter-clockwise by `degrees`
    c = ref[["x", "y"]].to_numpy(float).mean(axis=0)
    a = np.deg2rad(degrees)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    out = []
    for df in (ref, markers):
        df = df.copy()
        df[["x", "y"]] = (df[["x", "y"]].to_numpy(float) - c) @ R.T + c
        out.append(df)
    return out


def upright_angle(ref, top="1", bottom="4"):
    # rotation (degrees, counter-clockwise) that puts tag `top` straight above tag `bottom`
    v = ref.loc[top, ["x", "y"]].to_numpy(float) - ref.loc[bottom, ["x", "y"]].to_numpy(float)
    return 90.0 - np.degrees(np.arctan2(v[1], v[0]))


def make_video(exp, baseline, out, fps, speed, vmax, dpi, vid_start=None, rotate=None):
    exp_df = load_csv(DATA_DIR / exp)
    exp_m = bidirectional(exp_df)
    base_m = bidirectional(load_csv(DATA_DIR / baseline))
    m, s_theta, s_alpha = link_distances(exp_m, base_m)
    ref, markers = tag_positions(DATA_DIR / exp)
    rotate = upright_angle(ref) if rotate is None else rotate
    print(f"  layout rotated by {rotate:.1f}° counter-clockwise")
    ref, markers = rotate_layout(ref, markers, rotate)
    freq = int(m["Frequency (MHz)"].iloc[0])

    pairs = sorted({(a, b) for a, b in zip(m["T1"], m["T2"])}, key=lambda p: (tag_num(p[0]), tag_num(p[1])))
    events = {p: g[["ready", "dist"]].to_numpy() for p, g in m.groupby(["T1", "T2"])}   # sorted by ready time
    vmax = vmax or float(np.percentile(m["dist"], 99))
    norm = Normalize(0, vmax, clip=True)
    cmap = plt.get_cmap("Blues")
    t_end = float(m["ready"].max()) + 1.0
    print(f"{exp} vs {baseline} at {freq} MHz: {len(m)} link measurements, {len(pairs)} links, {t_end:.1f} s; "
          f"baseline spreads s_θ = {s_theta:.3f} rad, s_α = {s_alpha:.4g}; colour scale 0–{vmax:.2f}")

    # every unidirectional MPP, in time order: the camera panel follows these
    t0 = exp_df["MPP Start Time (s)"].min()
    mpps = pd.DataFrame({"stop": exp_df["MPP Stop Time (s)"] - t0, "Rx": exp_df["Rx"], "Tx": exp_df["Tx"]})
    mpps = mpps.sort_values("stop").reset_index(drop=True)
    mpp_stop = mpps["stop"].to_numpy()

    start = video_start(DATA_DIR / exp, vid_start)
    movs = sorted((DATA_DIR / exp).glob("*.mov"))
    camera = CameraReader(movs[0], start) if start is not None and movs else None
    if camera is None:
        print(f"  no camera panel: {'no .mov file' if not movs else 'no vid_start_time.txt (use --vid-start)'}")
    else:
        print(f"  camera: {movs[0].name}, experiment starts at {start:.2f} s in the recording")

    if camera:
        fig = plt.figure(figsize=(15.5, 8.4))
        cam_ax = fig.add_axes([0.01, 0.31, 0.50, 0.60])
        ax = fig.add_axes([0.52, 0.31, 0.38, 0.60])
        cax = fig.add_axes([0.915, 0.34, 0.012, 0.54])
        tax = fig.add_axes([0.06, 0.06, 0.90, 0.17])
        cam_ax.set_xticks([])
        cam_ax.set_yticks([])
        cam_img = cam_ax.imshow(camera.frame(0.0))
        cam_title = cam_ax.set_title("", fontsize=11)
    else:
        fig = plt.figure(figsize=(7.2, 8.6))
        ax = fig.add_axes([0.04, 0.30, 0.78, 0.62])
        cax = fig.add_axes([0.85, 0.33, 0.03, 0.56])
        tax = fig.add_axes([0.11, 0.06, 0.84, 0.17])

    # layout panel: floor labels, links, tags
    pad = 0.35
    ax.scatter(markers["x"], markers["y"], s=6, c="#9a9a9a", zorder=1)
    patches, flashes = {}, {}
    for a, b in pairs:
        pa = ref.loc[str(tag_num(a)), ["x", "y"]].to_numpy(float)
        pb = ref.loc[str(tag_num(b)), ["x", "y"]].to_numpy(float)
        cs = f"arc3,rad={link_curvature(ref, a, b)}"
        flashes[(a, b)] = ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="-", connectionstyle=cs, color=HIGHLIGHT,
                                                        lw=0, alpha=0.0, zorder=1.5, capstyle="round"))
        patches[(a, b)] = ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="-", connectionstyle=cs, color="#e6e6e6",
                                                        lw=0.8, ls="--", zorder=2, capstyle="round"))
    for tag, r in ref.iterrows():
        ax.scatter(r.x, r.y, s=260, color=tag_color(tag), edgecolors="white", linewidths=1.5, zorder=4)
        ax.annotate(tag, (r.x, r.y), ha="center", va="center", fontsize=9, color="white", fontweight="bold",
                    zorder=5)
    ax.set_xlim(ref["x"].min() - pad, ref["x"].max() + pad)
    ax.set_ylim(ref["y"].min() - pad, ref["y"].max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#dddddd")
    title = ax.set_title("", fontsize=12)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=f"distance from {baseline}")
    fig.suptitle(f"{exp}: distance of each link from {baseline}, [θ̂ | α̂] at {freq} MHz", fontsize=11, y=0.975)

    # timeline: every link's distance (faint) and the mean over links, with a cursor at the current time
    grid = np.arange(0, t_end, 0.05)
    held = []
    for p in pairs:
        ev = events[p]
        i = np.searchsorted(ev[:, 0], grid, side="right") - 1
        held.append(np.where(i >= 0, ev[np.maximum(i, 0), 1], np.nan))
        tax.step(ev[:, 0], ev[:, 1], where="post", color=cmap(0.45), lw=0.5, alpha=0.35)
    held = np.array(held)
    n_held = np.isfinite(held).sum(axis=0)
    mean = np.where(n_held > 0, np.nansum(held, axis=0) / np.maximum(n_held, 1), np.nan)
    tax.plot(grid, mean, color="#08306b", lw=1.5, label="mean over links")
    cursor = tax.axvline(0, color=HIGHLIGHT, lw=1.5)
    tax.set_xlim(0, t_end)
    tax.set_ylim(0, max(vmax, np.nanmax(m["dist"])) * 1.05)
    tax.set_xlabel("time from the first MPP (s)")
    tax.set_ylabel("distance")
    tax.legend(fontsize=8, loc="upper right", frameon=False)
    tax.grid(alpha=0.3)

    n_frames = int(np.ceil(t_end / speed * fps))
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(out), dpi=dpi):
        for k in range(n_frames):
            t = k / fps * speed                                    # experiment time shown in this frame
            n_done = 0
            for p in pairs:
                ev = events[p]
                i = np.searchsorted(ev[:, 0], t, side="right") - 1
                n_done += i + 1
                if i < 0:
                    continue                                       # not measured yet: stays grey and dashed
                d = norm(ev[i, 1])
                patch = patches[p]
                patch.set_color(cmap(d))
                patch.set_linestyle("-")
                patch.set_linewidth(0.8 + 5.5 * d)
                patch.set_zorder(2 + d)                            # the strongest links on top
                age = t - ev[i, 0]
                f = flashes[p]
                if age < FLASH_S:
                    f.set_alpha(0.9 * (1 - age / FLASH_S))
                    f.set_linewidth(0.8 + 5.5 * d + 5)
                else:
                    f.set_alpha(0.0)
            title.set_text(f"t = {t:6.2f} s    {n_done} / {len(m)} link measurements")
            if camera:
                j = np.searchsorted(mpp_stop, t, side="right") - 1    # the latest MPP finished by t
                if j >= 0:
                    tc = mpp_stop[j]
                    cam_img.set_data(camera.frame(tc))
                    cam_title.set_text(f"camera at the latest MPP: t = {tc:6.2f} s  "
                                       f"(T{tag_num(mpps['Rx'][j])} ← T{tag_num(mpps['Tx'][j])}, "
                                       f"{start + tc:.2f} s into the recording)")
                else:
                    cam_title.set_text("camera: waiting for the first MPP")
            cursor.set_xdata([t, t])
            writer.grab_frame()
            if k % (fps * 10) == 0:
                print(f"  frame {k} / {n_frames}  (t = {t:.1f} s)", flush=True)
    plt.close(fig)
    if camera:
        camera.close()
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", nargs="?", default="walking915_1", help="experiment folder in human_tracking_data")
    ap.add_argument("--baseline", default="baseline915", help="baseline experiment folder (default baseline915)")
    ap.add_argument("--out", type=Path, help="output .mp4 (default <exp>_link_distance.mp4 next to this script)")
    ap.add_argument("--fps", type=int, default=30, help="video frames per second (default 30)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time (default 1)")
    ap.add_argument("--vmax", type=float, help="top of the colour scale (default: 99th percentile of distances)")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--rotate", type=float, default=None,
                    help="rotate the layout counter-clockwise by this many degrees (default: the angle that puts "
                         "T1 straight above T4, so the T1-T4 arm is vertical)")
    ap.add_argument("--vid-start", help="time in the .mov at which the experiment starts, e.g. 0:05 "
                                        "(default: from vid_start_time.txt)")
    args = ap.parse_args()
    out = args.out or HERE / f"{args.exp}_link_distance.mp4"
    make_video(args.exp, args.baseline, out, args.fps, args.speed, args.vmax, args.dpi, args.vid_start, args.rotate)


if __name__ == "__main__":
    main()
