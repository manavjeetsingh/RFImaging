"""Robust per-channel voltage extraction from an MPP capture.

Same sweep model and grid search as mpp_extraction (MPP_CHANNELS = {1,1,1,1,3,4,6,7,8}, one padding ch1 dwell
before the measured window, score = contrast across the 5 switches - step at the ch1 entry), with two changes
that stop a single corrupted sample from moving the grid:

1. despike(): samples far from a running median (|v - medfilt| > max(K_SIGMA * sigma, MIN_SPIKE_MV)) are
   replaced by the running median. The traces contain transfer-corrupted samples at ~10x / 100x / 0x their
   neighbours (a decimal point moved or lost), e.g. 1966.5 mV inside a flat 196.5 mV stretch.
2. fit_sweep_grid_robust(): blocks are scored by a TRIMMED MEAN (drop TRIM_FRAC of samples at each end)
   instead of the plain mean, so one leftover outlier in a ~100-sample block cannot shift it by tens of mV (the
   mean-based score rewards a spiky block twice as a fake switch and penalises it at the ch1 entry, which is
   what flipped the grid by one dwell). A plain median was tried and is worse: bladeRF traces alternate between
   two levels ~8-10 mV apart, and the median of such a block jumps between them, creating fake switches
   (moved the grid on ~20% of bladeRF traces). The trimmed mean still averages both levels.

Block statistics for every start are computed at once with a sliding window, so it stays vectorised.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from mpp_extraction import (DWELL_SAMPLES, DWELL_TOLERANCE, EDGE_TRIM, MEASURED_DWELL_SLACK,
                            channel_windows)

DESPIKE_WIN = 9          # running-median width (samples); odd, << dwell (~120) so steps are preserved
K_SIGMA = 8.0            # spike threshold in robust noise sigmas
MIN_SPIKE_MV = 20.0      # never call anything smaller than this a spike (bladeRF traces alternate ~10 mV)
TRIM_FRAC = 0.10         # fraction dropped at each end of a block for the grid score


def running_median(v, win=DESPIKE_WIN):
    half = win // 2
    padded = np.pad(v, half, mode="edge")
    return np.median(sliding_window_view(padded, win), axis=1)


def despike(voltages, win=DESPIKE_WIN, k_sigma=K_SIGMA, min_spike=MIN_SPIKE_MV):
    """Return (cleaned trace, boolean mask of replaced samples)."""
    v = np.asarray(voltages, dtype=float)
    mf = running_median(v, win)
    resid = v - mf
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    bad = np.abs(resid) > max(k_sigma * sigma, min_spike)
    out = v.copy()
    out[bad] = mf[bad]
    return out, bad


def _dwell_range(transport, dwell_samples):
    if (transport is None) == (dwell_samples is None):
        raise ValueError("give exactly one of transport= or dwell_samples=; "
                         f"got transport={transport!r}, dwell_samples={dwell_samples!r}")
    if dwell_samples is not None:
        nominal = int(round(dwell_samples))
        if nominal < 2 * EDGE_TRIM + 1:
            raise ValueError(f"dwell_samples={dwell_samples} is too short to measure")
        return max(2 * EDGE_TRIM + 1, nominal - MEASURED_DWELL_SLACK), nominal + MEASURED_DWELL_SLACK + 1
    try:
        nominal = DWELL_SAMPLES[transport]
    except KeyError:
        raise ValueError(f"unknown transport {transport!r}, expected one of {sorted(DWELL_SAMPLES)}") from None
    return int(nominal * (1 - DWELL_TOLERANCE)), int(nominal * (1 + DWELL_TOLERANCE)) + 1


def fit_sweep_grid_robust(voltages, n_measured=6, *, transport=None, dwell_samples=None,
                          trim_frac=TRIM_FRAC):
    """(start, dwell, score) like mpp_extraction.fit_sweep_grid, but with block trimmed means in the score."""
    v = np.asarray(voltages, dtype=float)
    lo, hi = _dwell_range(transport, dwell_samples)

    best = (-np.inf, None, None)
    for dwell in range(lo, hi):
        starts = np.arange(dwell, len(v) - n_measured * dwell + 1)
        if len(starts) == 0:
            continue
        w = dwell - 2 * EDGE_TRIM
        # stat[p] = trimmed mean of v[p : p + w]; block [s + off, s + off + dwell) -> stat[s + off + EDGE_TRIM]
        k = int(w * trim_frac)
        srt = np.sort(sliding_window_view(v, w), axis=1)
        med = srt[:, k:w - k].mean(axis=1)
        block = lambda off: med[starts + off + EDGE_TRIM]           # noqa: E731
        means = [block(i * dwell) for i in range(n_measured)]
        prev = block(-dwell)
        score = sum(np.abs(means[i + 1] - means[i]) for i in range(n_measured - 1))
        score = score - np.abs(means[0] - prev)
        j = int(np.argmax(score))
        if score[j] > best[0]:
            best = (float(score[j]), int(starts[j]), dwell)
    if best[1] is None:
        raise ValueError(f"trace too short to contain a sweep ({len(v)} samples)")
    return best[1], best[2], best[0]


def segment_capture_robust(voltages, channels, *, transport=None, dwell_samples=None):
    """Return (medians, per-channel samples, (start, dwell, score), spike mask) on the despiked trace."""
    v, bad = despike(voltages)
    start, dwell, score = fit_sweep_grid_robust(v, n_measured=len(channels),
                                                transport=transport, dwell_samples=dwell_samples)
    medians, per_channel = {}, {}
    for ch, (lo, hi) in channel_windows(start, dwell, channels).items():
        seg = v[lo + EDGE_TRIM:hi - EDGE_TRIM]
        medians[ch] = float(np.median(seg))
        per_channel[ch] = seg
    return medians, per_channel, (start, dwell, score), bad
