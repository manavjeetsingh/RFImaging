"""Improved unidirectional phase calculations, using the per-channel ADC samples instead of one median per channel.

Same model as phase_calc_original.cal_theta_et_al (Rx PV polynomial -> amplitude, Tx S11 per channel, least squares
amp_k = V + x1*a_k*cos(phi_k) + x2*a_k*sin(phi_k), theta = atan2(x2, x1)). The calibration and S11 basis do not
depend on the voltages, so the least-squares pseudo-inverse is built once per call and applied to many voltage
vectors at once.

1. cal_theta_percentile: aggregate each channel's samples with a percentile (default 70th) instead of the median.
2. cal_theta_sections:   split each channel's samples into N_SECTIONS contiguous sections, take the median of each
   section, solve for theta on N_COMBOS fixed combinations of one section per channel (out of N_SECTIONS^n_channels),
   and return the circular median of those thetas.

Inputs: samples = {channel: 1-D array of mV} (mpp extractor's "samples"), rx/tx MACs, cfg from
phase_calc_original.load_hw_config(), freq in Hz. All return theta in degrees, in (-180, 180].
"""

import numpy as np

N_SECTIONS = 10
N_COMBOS = 100
COMBO_SEED = 0
PERCENTILE = 70.0


# ---------------------------------------------------------------------------
# shared model pieces (vectorised versions of phase_calc_original.cal_theta_et_al)
# ---------------------------------------------------------------------------

def mv_to_amplitude(mv, rx_mac, cfg, freq):
    """mV (any shape) -> amplitude, via the Rx PV polynomial at `freq` (same formula as cal_theta_et_al)."""
    dbm = np.polyval(cfg['pv'][rx_mac][freq]['polynomial'], np.log(mv))
    uW = np.power(10, (dbm - 30) / 10) * 1e6
    return np.sqrt(uW * 50 * 2)


def lstsq_operator(channels, tx_mac, cfg, freq):
    """P (3 x n_channels) with [V, x1, x2] = P @ amp, for the Tx S11 basis of `channels` at `freq`."""
    h = []
    for ch in channels:
        s11 = cfg['s11'][tx_mac][ch]
        p = np.interp(freq, s11['freq'], np.unwrap(s11['phase']))
        a = np.interp(freq, s11['freq'], s11['amp'])
        h.append([1, a * np.cos(p), a * np.sin(p)])
    h = np.asarray(h)
    return np.linalg.inv(h.T @ h) @ h.T


def solve_from_mv(mv, channels, rx_mac, tx_mac, cfg, freq):
    """mv: (..., n_channels) mV in `channels` order -> (theta deg, V, beta), each shape (...)."""
    out = mv_to_amplitude(np.asarray(mv, float), rx_mac, cfg, freq) @ lstsq_operator(channels, tx_mac, cfg, freq).T
    return np.rad2deg(np.arctan2(out[..., 2], out[..., 1])), out[..., 0], np.hypot(out[..., 1], out[..., 2])


def theta_from_mv(mv, channels, rx_mac, tx_mac, cfg, freq):
    """mv: (..., n_channels) mV in `channels` order -> theta (deg), shape (...)."""
    return solve_from_mv(mv, channels, rx_mac, tx_mac, cfg, freq)[0]


def circular_median_deg(theta_deg):
    """Median of angles (deg): unwrap around the circular mean, take the median, wrap to (-180, 180]."""
    t = np.asarray(theta_deg, float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return np.nan
    ref = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(t)))))
    med = ref + np.median((t - ref + 180.0) % 360.0 - 180.0)
    return float(180.0 - (180.0 - med) % 360.0)


# ---------------------------------------------------------------------------
# 1. percentile aggregation
# ---------------------------------------------------------------------------

def cal_theta_et_al_percentile(samples, rx_mac, tx_mac, cfg, freq, q=PERCENTILE):
    """(theta_deg, V, beta, {channel: q-th percentile mV}) -- cal_theta_et_al on per-channel q-th percentiles."""
    channels = list(samples)
    agg = {ch: float(np.percentile(samples[ch], q)) for ch in channels}
    theta, V, beta = solve_from_mv(np.array([agg[ch] for ch in channels]), channels, rx_mac, tx_mac, cfg, freq)
    return float(theta), float(V), float(beta), agg


def cal_theta_percentile(samples, rx_mac, tx_mac, cfg, freq, q=PERCENTILE):
    """theta (deg) from the q-th percentile of each channel's samples."""
    return cal_theta_et_al_percentile(samples, rx_mac, tx_mac, cfg, freq, q)[0]


# ---------------------------------------------------------------------------
# 2. section medians + fixed combinations
# ---------------------------------------------------------------------------

def section_medians(seg, n_sections=N_SECTIONS):
    """Medians of n_sections contiguous, near-equal sections of seg (np.array_split: lengths differ by <= 1)."""
    seg = np.asarray(seg, float)
    if len(seg) < n_sections:
        raise ValueError(f"{len(seg)} samples cannot be split into {n_sections} sections")
    return np.array([np.median(s) for s in np.array_split(seg, n_sections)])


def fixed_combinations(n_channels, n_sections=N_SECTIONS, n_combos=N_COMBOS, seed=COMBO_SEED):
    """(n_combos, n_channels) section indices, fixed for a given (n_channels, n_sections, n_combos, seed).

    The first n_sections rows are the time-aligned diagonals (section i on every channel). The rest are distinct
    combinations drawn uniformly without replacement from the remaining n_sections**n_channels, with a fixed seed.
    """
    total = n_sections ** n_channels
    n_combos = min(n_combos, total)
    diag = [sum(i * n_sections ** k for k in range(n_channels)) for i in range(n_sections)][:n_combos]
    rng = np.random.default_rng(seed)
    pool = np.setdiff1d(np.arange(total), diag) if total <= 10 ** 7 else None
    if pool is not None:
        rest = rng.choice(pool, n_combos - len(diag), replace=False)
    else:
        rest, seen = [], set(diag)
        while len(rest) < n_combos - len(diag):
            c = int(rng.integers(total))
            if c not in seen:
                seen.add(c)
                rest.append(c)
    flat = np.concatenate([diag, rest]).astype(np.int64)
    # base-n_sections digits: column k = section index of channel k
    return (flat[:, None] // n_sections ** np.arange(n_channels)) % n_sections


_COMBO_CACHE = {}


def _combos(n_channels, n_sections, n_combos):
    key = (n_channels, n_sections, n_combos)
    if key not in _COMBO_CACHE:
        _COMBO_CACHE[key] = fixed_combinations(n_channels, n_sections, n_combos)
    return _COMBO_CACHE[key]


def section_thetas(samples, rx_mac, tx_mac, cfg, freq, n_sections=N_SECTIONS, n_combos=N_COMBOS):
    """The n_combos per-combination thetas (deg)."""
    channels = list(samples)
    meds = np.stack([section_medians(samples[ch], n_sections) for ch in channels])   # (n_channels, n_sections)
    combos = _combos(len(channels), n_sections, n_combos)                            # (n_combos, n_channels)
    mv = meds[np.arange(len(channels)), combos]                                      # (n_combos, n_channels)
    return theta_from_mv(mv, channels, rx_mac, tx_mac, cfg, freq)


def cal_theta_sections(samples, rx_mac, tx_mac, cfg, freq, n_sections=N_SECTIONS, n_combos=N_COMBOS):
    """Circular median (deg) of the thetas over fixed combinations of per-channel section medians."""
    return circular_median_deg(section_thetas(samples, rx_mac, tx_mac, cfg, freq, n_sections, n_combos))
