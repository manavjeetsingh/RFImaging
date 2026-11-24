"""Per-channel voltage extraction from an MPP capture.

Copied from BSTag-Firmware/data_collection:
  - ribbn_scripts/src/ribbn_scripts/processing/mpp_segment.py
        (fit_sweep_grid, channel_windows, segment_capture -- the algorithm)
  - sched_collect.py            (dwell_samples_from_slot, SAMPLE_US, MPP_DWELL_US)
  - measurePhasesMultiThreadedMultiTags.py   (getChannelVoltage)
  - post_processing.py          (parse_float_array, segment_trace -- for plotting)

How collection.py gets here:
  collection.py -> mtt.mainMultiWays ("individual")  -> getChannelVoltage -> segment_capture(transport=...)
               -> multiple_collection.mainMultiple ("multiple") -> _segment_round
                                       -> segment_capture(dwell_samples=dwell_samples_from_slot(slot))

The tag sweeps MPP_CHANNELS = {1,1,1,1,3,4,6,7,8}, holding MPP_DWELL_US (3ms)
on each entry, so ch1 is held for four dwells and the rest for one each. Only
the last six dwells are measured; the three leading ch1 dwells are padding.

The grid (start, dwell) is recovered from the trace itself, not from host
wall-clock timing (that jitters by ~3.6 ms, more than a whole dwell).
"""

import ast
import re

import numpy as np

# ---------------------------------------------------------------------------
# mpp_segment.py
# ---------------------------------------------------------------------------

# Samples per MPP_DWELL_US (3ms) dwell for "individual" collection, keyed by
# CONNECTION. A "multiple" (scheduled) round uses dwell_samples instead.
DWELL_SAMPLES = {"wireless": 45, "wired": 90}
DWELL_TOLERANCE = 0.03   # absorbs small drift in that loop rate

# Search margin (samples) when the dwell is measured (dwell_samples=).
MEASURED_DWELL_SLACK = 1
N_PAD = 4                # leading ch1 dwells in MPP_CHANNELS
EDGE_TRIM = 8            # samples either side of a switch, which catch the transition

DEFAULT_CHANNELS = [1, 3, 4, 6, 7, 8]


def _block_means(cumsum, starts, lo_off, hi_off):
    a = starts + lo_off + EDGE_TRIM
    b = starts + hi_off - EDGE_TRIM
    return (cumsum[b] - cumsum[a]) / (b - a)


def fit_sweep_grid(voltages, n_measured=6, *, transport=None, dwell_samples=None):
    """Find (start, dwell, score) of the measured dwells; start is the ch1 window.

    Give exactly one of `transport` (looks up DWELL_SAMPLES, +/-3% search) or
    `dwell_samples` (measured nominal, +/-MEASURED_DWELL_SLACK search).

    Scores a candidate grid by the contrast across the channel switches it
    implies, minus any step at the ch1 entry -- the padding dwell before the
    measured ch1 is the same channel, so that one boundary must be flat. That
    asymmetry is what pins the phase of the grid.
    """
    v = np.asarray(voltages, dtype=float)
    cs = np.concatenate([[0.0], np.cumsum(v)])

    if (transport is None) == (dwell_samples is None):
        raise ValueError("give exactly one of transport= or dwell_samples=; "
                         f"got transport={transport!r}, "
                         f"dwell_samples={dwell_samples!r}")

    if dwell_samples is not None:
        nominal = int(round(dwell_samples))
        slack = MEASURED_DWELL_SLACK
        if nominal < 2 * EDGE_TRIM + 1:
            raise ValueError(
                f"dwell_samples={dwell_samples} is too short to measure: "
                f"EDGE_TRIM drops {EDGE_TRIM} samples either side of every "
                f"switch, so a dwell needs more than {2 * EDGE_TRIM} samples.")
    else:
        try:
            nominal = DWELL_SAMPLES[transport]
        except KeyError:
            raise ValueError(f"unknown transport {transport!r}, expected one of "
                             f"{sorted(DWELL_SAMPLES)}") from None
        slack = None

    if slack is None:
        lo = int(nominal * (1 - DWELL_TOLERANCE))
        hi = int(nominal * (1 + DWELL_TOLERANCE)) + 1
    else:
        lo = max(2 * EDGE_TRIM + 1, nominal - slack)
        hi = nominal + slack + 1

    best = (-np.inf, None, None)
    for dwell in range(lo, hi):
        # one padding dwell must sit before the window, and the window must fit
        starts = np.arange(dwell, len(v) - n_measured * dwell + 1)
        if len(starts) == 0:
            continue
        means = [_block_means(cs, starts, i * dwell, (i + 1) * dwell) for i in range(n_measured)]
        prev = _block_means(cs, starts, -dwell, 0)
        score = sum(np.abs(means[i + 1] - means[i]) for i in range(n_measured - 1))
        score = score - np.abs(means[0] - prev)
        j = int(np.argmax(score))
        if score[j] > best[0]:
            best = (float(score[j]), int(starts[j]), dwell)
    if best[1] is None:
        raise ValueError(f"trace too short to contain a sweep ({len(v)} samples)")
    return best[1], best[2], best[0]


def channel_windows(start, dwell, channels):
    return {ch: (start + i * dwell, start + (i + 1) * dwell) for i, ch in enumerate(channels)}


def segment_capture(voltages, channels, *, transport=None, dwell_samples=None):
    """Return (medians, per-channel samples, (start, dwell, score))."""
    v = np.asarray(voltages, dtype=float)
    start, dwell, score = fit_sweep_grid(v, n_measured=len(channels),
                                         transport=transport,
                                         dwell_samples=dwell_samples)

    medians, per_channel = {}, {}
    for ch, (lo, hi) in channel_windows(start, dwell, channels).items():
        seg = v[lo + EDGE_TRIM:hi - EDGE_TRIM]
        medians[ch] = float(np.median(seg))
        per_channel[ch] = seg
    return medians, per_channel, (start, dwell, score)


# ---------------------------------------------------------------------------
# sched_collect.py  ("multiple" collection: dwell length from the slot's rate)
# ---------------------------------------------------------------------------

SAMPLE_US = 25          # paced listen-slot sample period
MPP_DWELL_US = 3000     # firmware dwell per MPP channel


def dwell_samples_from_slot(slot):
    """Nominal dwell length, in samples, for a listen slot's trace.

    `slot` is a dict with optional "rate_hz" (the CSV's "Sample Rate (Hz)").
    """
    rate_hz = slot.get("rate_hz")
    if not rate_hz:
        return MPP_DWELL_US / SAMPLE_US
    return rate_hz * MPP_DWELL_US / 1e6


# ---------------------------------------------------------------------------
# measurePhasesMultiThreadedMultiTags.py
# ---------------------------------------------------------------------------

def getChannelVoltage(voltage_readings, channels=DEFAULT_CHANNELS, plotting=False,
                      *, transport=None, dwell_samples=None):
    """Segment a capture into its per-channel dwells -> (medians, per-channel samples).

    Same as the live pipeline's getChannelVoltage, extended with dwell_samples=
    so it also covers the "multiple" collection path (multiple_collection._segment_round).
    """
    import matplotlib.pyplot as plt

    plot_voltage = np.asarray(voltage_readings)
    channel_medians, channel_voltages, (start, dwell, score) = segment_capture(
        plot_voltage, channels, transport=transport, dwell_samples=dwell_samples)

    if plotting:
        plt.figure(figsize=(20, 10))
        plt.plot(plot_voltage, '.')
        for i in range(len(channels) + 1):
            plt.axvline(x=start + i * dwell, color='b')
        for ch, (left, right) in channel_windows(start, dwell, channels).items():
            print(f"  ch{ch} [{left}, {right}): {len(channel_voltages[ch])} readings: {channel_medians[ch]}")
            plt.hlines(channel_medians[ch], left, right, color='r', zorder=3)
        plt.xlabel("Sample")
        plt.ylabel("ADC out [mV]")
        plt.show()

    return channel_medians, channel_voltages


# ---------------------------------------------------------------------------
# post_processing.py  (reading traces back from CSV + time-axis segmentation)
# ---------------------------------------------------------------------------

def parse_float_array(text: str) -> np.ndarray:
    """Parse a stored array cell, written either as a Python list or a numpy repr."""
    if "..." in text:
        raise ValueError("array was written as numpy's summarised repr and is missing samples")
    try:
        return np.asarray(ast.literal_eval(text), dtype=float)
    except (ValueError, SyntaxError):
        return np.asarray([float(v) for v in re.split(r"[\s,]+", text.strip("[] \n"))])


def segment_trace(voltages, elapsed, channels, *, transport=None, dwell_samples=None):
    """Same fit as segment_capture, returned on a time axis for plotting.

    Returns (time, ver_lines, segments, medians) where ver_lines are the
    len(channels)+1 boundary times and segments is [(ch, left_s, right_s, median)].
    `elapsed` (seconds for the whole trace) only scales the axis.
    """
    voltages = np.asarray(voltages, dtype=float)
    if dwell_samples is not None:
        medians, per_channel, (start, dwell, _) = segment_capture(
            voltages, channels, dwell_samples=dwell_samples)
    else:
        medians, per_channel, (start, dwell, _) = segment_capture(
            voltages, channels, transport=transport)

    dt = elapsed / len(voltages)
    time = np.arange(len(voltages)) * dt

    ver_lines = [(start + i * dwell) * dt for i in range(len(channels) + 1)]
    segments = [
        (ch, lo * dt, hi * dt, medians[ch])
        for ch, (lo, hi) in channel_windows(start, dwell, channels).items()
    ]
    return time, ver_lines, segments, medians
