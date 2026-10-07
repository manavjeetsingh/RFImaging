"""Python version of LikelyHoodLocalization.m (non-all_circles branch), without MATLAB.

Each tag is localized on an image: around every anchor it has a measured link to, circles are drawn with radius
wrapped + K * lam for the estimated number of whole steps K and for K +- 1 (K +- 2, K +- 3 with circles = 5, 7).
The image is blurred with a disk and the brightest point is the tag estimate. The confidence is the peak (1 after
normalization) over the brightest point farther than conf_radius from it.

After each round the top floor(n_tags / 10) tags by confidence become anchors at their estimated positions, as in
the MATLAB code. When that is 0 (fewer than 10 tags left) and promote_by is set, the most confident tag is tried as
an anchor and kept only if the other tags' localization improves; otherwise the remaining rounds would repeat the
current one, so the loop stops.

All lengths (positions, wrapped, lam, lim, conf_radius) are in the same unit, e.g. cm; scale is pixels per unit.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

RED_GRAY = 0.2989  # rgb2gray of the red circles drawn by insertShape


@dataclass
class Params:
    lam: float                  # distance per step of K
    lim: float                  # image covers [0, lim] x [0, lim]
    scale: float = 3.0          # pixels per unit length
    circles: int = 3            # circles per anchor: K, K +- 1 (5 or 7 add K +- 2, K +- 3)
    blur_px: int = 15           # radius of the disk blur, fspecial('disk', 15)
    conf_radius: float = 50.0   # black-out radius around the peak for the confidence
    rounds: int = 5             # rounds of iterative position correction
    promote_by: str | None = "conf"  # "conf", "error" (needs true_pos, oracle) or None (MATLAB rule only)


class _Image:
    # Pixel grid in MATLAB coordinates: pixel (row i, col j), 1-based, has its center at (x = j, y = i).
    def __init__(self, p: Params):
        self.p = p
        self.n = int(p.scale * p.lim)
        self.yy, self.xx = np.mgrid[1:self.n + 1, 1:self.n + 1].astype(float)
        r = p.blur_px
        y, x = np.mgrid[-r:r + 1, -r:r + 1]
        k = (x ** 2 + y ** 2 <= r ** 2).astype(float)
        self.kernel = k / k.sum()

    def rings(self, centers, radii, width):
        # One insertShape call: anti-aliased circle outlines; overlapping circles overwrite each other (max).
        img = np.zeros((self.n, self.n))
        s = self.p.scale
        for (cx, cy), r in zip(centers, radii):
            d = np.hypot(self.xx - cx * s, self.yy - cy * s)
            img = np.maximum(img, np.clip(width / 2 + 0.5 - np.abs(d - r * s), 0, 1))
        return RED_GRAY * img

    def blur(self, img):
        # imfilter(img, fspecial('disk', r), 'symmetric'), with a binary disk.
        r = self.p.blur_px
        pad = np.pad(img, r, mode="symmetric")
        return fftconvolve(pad, self.kernel, mode="same")[r:-r, r:-r]


def locate(image, wrapped, k, anchor_pos):
    """Localize one tag.

    wrapped, k: per-anchor wrapped distance and number of steps K for this tag (only measured links).
    anchor_pos: (n, 2) positions of those anchors.
    Returns a dict with the estimate, confidence and the raw / blurred images.
    """
    p = image.p
    raw = image.rings(anchor_pos, wrapped + k * p.lam, 2)
    for s in range(1, (p.circles - 1) // 2 + 1):
        raw = raw + image.rings(anchor_pos, wrapped + np.maximum(k - s, 0) * p.lam, 1)
        raw = raw + image.rings(anchor_pos, wrapped + (k + s) * p.lam, 1)
    cnv = image.blur(raw)
    if cnv.max() <= 0:  # no measured anchor: MATLAB falls back to (0, 0)
        return {"est": np.zeros(2), "conf": 0.0, "raw": raw, "blurred": cnv}
    cnv = cnv / cnv.max()
    yy, xx = np.nonzero(cnv == 1)
    x, y = xx.mean() + 1, yy.mean() + 1
    outside = np.hypot(image.xx - x, image.yy - y) > p.conf_radius * p.scale
    conf = 1 / cnv[outside].max() if outside.any() and cnv[outside].max() > 0 else np.inf
    return {"est": np.array([x, y]) / p.scale, "conf": conf, "raw": raw, "blurred": cnv}


def likelihood_localization(wrapped, K, measured, anchors, anchor_pos, tags, params: Params,
                            true_pos=None, labels=None, verbose=True):
    """Localize tags from wrapped distances and whole steps K to anchors.

    wrapped, K, measured: (N, N) symmetric matrices over all nodes (anchors and tags): wrapped distance, number of
        steps K, and whether the link was measured.
    anchors: node indices with known positions anchor_pos (len(anchors), 2); tags: node indices to localize.
    true_pos: (N, 2) true positions, only needed for promote_by="error".
    labels: names for the nodes in printed messages (default: their indices).
    Returns (estimates, rounds): estimates maps tag -> (x, y); rounds is a list with, per round, the anchors used,
    each tag's state (see locate), the promoted tags and the promotion trial (if any).
    """
    if params.promote_by == "error" and true_pos is None:
        raise ValueError('promote_by="error" needs true_pos')
    image = _Image(params)
    name = (lambda t: labels[t]) if labels is not None else (lambda t: t)

    def run_tag(t, anchor_ids, anchor_xy):
        use = np.array([measured[t, a] for a in anchor_ids], bool)
        ids = np.asarray(anchor_ids)[use]
        state = locate(image, wrapped[t, ids], K[t, ids], anchor_xy[use])
        state["anchors_used"] = ids
        return state

    def score(states):
        # Higher is better.
        if params.promote_by == "conf":
            return np.mean([s["conf"] for s in states.values()])
        return -np.mean([np.linalg.norm(s["est"] - true_pos[t]) for t, s in states.items()])

    anchor_ids, anchor_xy = list(anchors), np.asarray(anchor_pos, float)
    remaining, estimates, rounds = list(tags), {}, []
    for rnd in range(1, params.rounds + 1):
        states = {t: run_tag(t, anchor_ids, anchor_xy) for t in remaining}
        estimates.update({t: s["est"] for t, s in states.items()})
        ranked = sorted(remaining, key=lambda t: -states[t]["conf"])
        promoted = ranked[:len(remaining) // 10]
        info = {"round": rnd, "anchors": list(anchor_ids), "states": states, "trial": None}
        msg = f"round {rnd}: {len(remaining)} tags, {len(anchor_ids)} anchors"

        if not promoted and params.promote_by and len(remaining) > 1:
            cand, others = ranked[0], ranked[1:]
            trial_xy = np.vstack([anchor_xy, states[cand]["est"]])
            before = score({t: states[t] for t in others})
            after = score({t: run_tag(t, anchor_ids + [cand], trial_xy) for t in others})
            accept = after > before
            info["trial"] = {"tag": cand, "before": before, "after": after, "accepted": accept}
            msg += (f", try tag {name(cand)}: {params.promote_by} score of the other tags {before:.3f} -> {after:.3f}, "
                    + ("promoted" if accept else "not promoted"))
            promoted = [cand] if accept else []
        info["promoted"] = promoted
        rounds.append(info)
        if verbose:
            print(msg + (f", promoted: {[name(t) for t in promoted]}" if promoted else ""))

        if not promoted:
            if verbose and rnd < params.rounds:
                print(f"no promotion: rounds {rnd + 1}-{params.rounds} would repeat round {rnd}, stopping")
            break
        for t in promoted:
            anchor_ids.append(t)
            anchor_xy = np.vstack([anchor_xy, states[t]["est"]])
            remaining.remove(t)
    return estimates, rounds
