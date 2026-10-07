"""Python version of the final location selection in merging_finalLocselection.m.

Merges the per-tag estimates of the likelihood method and SDP. For every tag, if the two estimates are closer than
near_mult * lam (they agree, and the likelihood estimate is the finer one) or farther apart than far_mult * lam
(SDP is taken to have failed), the likelihood estimate is used; otherwise the SDP estimate is kept.
"""
import numpy as np


def merge_locations(lkhd_est, sdp_est, lam=0.16, near_mult=1.5, far_mult=5.0):
    """Merge two sets of estimates for the same tags.

    lkhd_est, sdp_est: (n_tags, dim) estimates, row i of both for the same tag.
    lam: length unit of the thresholds (0.16 m in merging_finalLocselection.m), in the same unit as the estimates.
    Returns (merged, use_lkhd, gap): merged (n_tags, dim) estimates, a boolean mask of the tags that took the
    likelihood estimate, and the distance between the two estimates of every tag.
    """
    lkhd_est, sdp_est = np.asarray(lkhd_est, float), np.asarray(sdp_est, float)
    if lkhd_est.shape != sdp_est.shape:
        raise ValueError(f"shape mismatch: {lkhd_est.shape} vs {sdp_est.shape}")
    gap = np.linalg.norm(lkhd_est - sdp_est, axis=1)
    use_lkhd = (gap < near_mult * lam) | (gap > far_mult * lam)
    merged = np.where(use_lkhd[:, None], lkhd_est, sdp_est)
    return merged, use_lkhd, gap


def loc_errors(true_pos, est):
    """Per-tag localization error: Euclidean distance between rows of true_pos and est (calc_locError)."""
    return np.linalg.norm(np.asarray(true_pos, float) - np.asarray(est, float), axis=1)


def rmsd(true_pos, est):
    """Root-mean-square localization error over all tags (calc_rmsd)."""
    return float(np.sqrt(np.mean(loc_errors(true_pos, est) ** 2)))
