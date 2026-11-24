"""Original phase calculation + S11 / PV calibration loading.

Copied from BSTag-Firmware/data_collection:
  - collection.py                                    (build_s11_cfg, build_pv_cfg, load_hw_config)
  - ribbn_scripts/src/ribbn_scripts/processing/phase_cal.py   (cal_theta_et_al)
  - init.py / mac-tag-mapping.json                   (tag name <-> MAC)

Pipeline (per Rx <- Tx link, per MPP):
  1. channel medians (mV) from mpp_extraction.segment_capture
  2. Rx PV polynomial: dBm = polyval(poly, ln(mV)) -> uW -> amplitude sqrt(uW*50*2)
  3. Tx S11 per channel at `freq`: phase phi_k (unwrapped, interp), magnitude a_k
  4. Least squares over channels k:
        amp_k = V + x1*a_k*cos(phi_k) + x2*a_k*sin(phi_k)
     theta = atan2(x2, x1) (deg), beta = |(x1, x2)|
"""

import ast
import json
import math
import os

import numpy as np
import pandas as pd

FIRMWARE_DIR = "/Users/manavjeet/git/BSTag-Firmware/data_collection"
CALIB_DIR = os.path.join(FIRMWARE_DIR, "ribbn_scripts/src/ribbn_scripts/calibrations")
VNA_DATA_PATH = os.path.join(CALIB_DIR, "VNA_data_Sept2026/processed", "all_s11_poly.csv")
PV_DATA_PATH = os.path.join(CALIB_DIR, "PV_data_Sept2026_705_995_-7/processed", "all_pv_polynomials.csv")
MAC_TAG_FILE = os.path.join(FIRMWARE_DIR, "mac-tag-mapping.json")


# ---------------------------------------------------------------------------
# collection.py
# ---------------------------------------------------------------------------

def build_s11_cfg(vna_data):
    """Turn the raw all_s11_poly.csv rows into cfg['s11'][mac][channel] = {'freq' (Hz), 'phase' (rad), 'amp'}."""
    s11_cfg = {}
    for _, row in vna_data.iterrows():
        freq = np.array(ast.literal_eval(row['Frequencies']))
        phase = np.deg2rad(np.array(ast.literal_eval(row['Phases'])))
        amp = np.array(ast.literal_eval(row['Amplituds']))

        mac = row['Tag MAC'].replace('_', ':')

        tag_entry = s11_cfg.setdefault(mac, {})
        tag_entry[int(row['Channel'])] = {'freq': freq, 'phase': phase, 'amp': amp}

    return s11_cfg


def build_pv_cfg(pv_data):
    """Turn the all_pv_polynomials.csv rows into cfg['pv'][mac][freq (Hz)] = {'polynomial', 'inverse'}."""
    pv_cfg = {}
    for _, row in pv_data.iterrows():
        mac = row['Tag MAC'].replace('_', ':')

        tag_entry = pv_cfg.setdefault(mac, {})
        tag_entry[float(row['Frequency']) * 1e6] = {
            'polynomial': np.array(ast.literal_eval(row['Polynomial'])),
            'inverse': np.array(ast.literal_eval(row['Inverse'])),
        }

    return pv_cfg


def load_hw_config(vna_data_path=VNA_DATA_PATH, pv_data_path=PV_DATA_PATH):
    """cfg = {'s11': build_s11_cfg(...), 'pv': build_pv_cfg(...)} from the calibration CSVs."""
    vna_data = pd.read_csv(vna_data_path)
    pv_data = pd.read_csv(pv_data_path)

    cfg = {}
    cfg['s11'] = build_s11_cfg(vna_data)
    cfg['pv'] = build_pv_cfg(pv_data)
    return cfg


def load_tag_mac_mapping(mac_tag_file=MAC_TAG_FILE):
    """{'Tag1': '98:A3:...', ...} -- collection.py's tag_mac_mapping."""
    with open(mac_tag_file) as f:
        mac_tag_mapping = json.load(f)
    return {v: k for k, v in mac_tag_mapping.items()}


# ---------------------------------------------------------------------------
# phase_cal.py
# ---------------------------------------------------------------------------

def cal_theta_et_al(adcs, rxName, txName, cfg, freq):
    """adcs: {channel: median mV}; rxName/txName: MACs; freq in Hz.

    Returns (theta_deg, V, beta).
    """
    amp = []
    phi = []
    attn = []
    for channel in adcs.keys():
        dbm = np.polyval(cfg['pv'][rxName][freq]['polynomial'], np.log(adcs[channel]))
        uW = np.power(10, (dbm - 30) / 10) * 1e6
        amp.append(np.sqrt(uW * 50 * 2))

        # despite the _s11_poly file name, these hold the raw VNA sweep
        # (freq axis + per-point values), not polynomial coefficients
        s11 = cfg['s11'][txName][channel]
        phi.append(np.interp(freq, s11['freq'], np.unwrap(s11['phase'])))
        attn.append(np.interp(freq, s11['freq'], s11['amp']))

    h = []
    for a, p in zip(attn, phi):
        h.append([1, a * np.cos(p), a * np.sin(p)])
    out = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(h), h)), np.transpose(h)), amp)
    theta_deg = np.rad2deg(math.atan2(out[2], out[1]))
    V = out[0]
    beta = math.sqrt(out[1] * out[1] + out[2] * out[2])

    return theta_deg, V, beta


# ---------------------------------------------------------------------------
# post_processing.py  (load_combined_phases)
# ---------------------------------------------------------------------------

def combine_bidirectional_phase(phase_ab_deg, phase_ba_deg):
    """Link phase from the two unidirectional phases (A<-B and B<-A), in degrees.

    Same as post_processing.load_combined_phases: average the two directions,
    then mod 180 (mod pi) -- the per-tag offsets cancel in the sum, leaving the
    round-trip ambiguity of pi.
    """
    return ((phase_ab_deg + phase_ba_deg) / 2) % 180.0
