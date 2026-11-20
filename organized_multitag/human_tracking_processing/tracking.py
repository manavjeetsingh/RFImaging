"""Trackers that turn the per-run link scores of person_location_video.py into a track, where the estimate at run k
depends on the estimate at run k-1.

grid (default): a Bayes filter on a grid of floor positions (a discrete hidden Markov model).
    State: the person's position on a GRID_STEP grid over the tag layout. The belief is a probability per cell.
    Predict: the belief is blurred by a Gaussian with std speed * dt (dt = time between the runs, ~2 s): the
        person can have walked anywhere within roughly that distance.
    Update: the likelihood of cell p is (1 + G(p)) ** beta, where G(p) is the soft version of the cluster score:
        G(p) = exp(sum_i w_i(p) log(1 + s_i)) - 1,   w_i(p) ∝ exp(-|p - c_i|^2 / (2 rho^2)), sum_i w_i(p) = 1
        (s_i = link score, c_i = link centre). It is the geometric mean of (1 + link score) of the links whose
        centres are near p, i.e. the same quantity as the 6-link cluster score, but for every point of the floor
        and with no hard choice of 6 links. beta says how much one run is trusted against the motion model.
        Runs whose best cluster score is below the empty-room threshold give no update (prediction only).
    Output: the posterior mean and covariance per run (forward pass: uses runs up to k only); with smoothing,
        also the forward-backward posterior (uses all runs, like the centred rolling median already does).

kalman: a constant-velocity Kalman filter (state x, y, vx, vy) on the per-run cluster locations.
    The measurement noise shrinks as the cluster score grows (R = (meas_std^2 * threshold / score) I), a
    measurement more than GATE Mahalanobis units from the prediction is ignored, and runs below the threshold are
    prediction only. Optional Rauch-Tung-Striebel smoothing.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

GRID_STEP = 0.05          # m
GATE = 3.5                # Kalman measurement gate (Mahalanobis distance)


# ---------------------------------------------------------------- grid Bayes filter

def make_grid(ref, pad, step=GRID_STEP):
    xs = np.arange(ref["x"].min() - pad, ref["x"].max() + pad + step / 2, step)
    ys = np.arange(ref["y"].min() - pad, ref["y"].max() + pad + step / 2, step)
    return xs, ys


def support(centres, xs, ys, reach):
    # ny × nx mask of the grid points within `reach` of some link centre (elsewhere no link says anything)
    gx, gy = np.meshgrid(xs, ys)
    d = np.sqrt(((np.stack([gx, gy], -1)[:, :, None, :] - centres) ** 2).sum(-1)).min(-1)
    return d <= reach


def soft_cluster_score(scores, centres, xs, ys, rho, size):
    """-> runs × ny × nx: at every grid point, the geometric mean of (1 + link score), minus 1, over the `size`
    links with the nearest centres, weighted by exp(-d^2 / (2 rho^2)). Always `size` links, so (as in the cluster
    score) one strong link cannot carry a point on its own."""
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)                       # cells × 2
    d2 = ((pts[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)       # cells × links
    near = np.argsort(d2, axis=1)[:, :size]
    w = np.zeros_like(d2)
    rows = np.arange(len(pts))[:, None]
    w[rows, near] = np.exp(-(d2[rows, near] - d2[rows, near[:, :1]]) / (2 * rho ** 2))
    w /= w.sum(axis=1, keepdims=True)
    g = np.expm1(np.log1p(scores) @ w.T)                                  # runs × cells
    return g.reshape(len(scores), len(ys), len(xs))


def _blur(p, sigma_m, step):
    q = gaussian_filter(p, sigma_m / step, mode="constant") if sigma_m > 0 else p
    return q / q.sum()


def _moments(p, gx, gy):
    mx, my = (p * gx).sum(), (p * gy).sum()
    dx, dy = gx - mx, gy - my
    cov = np.array([[(p * dx * dx).sum(), (p * dx * dy).sum()], [(p * dx * dy).sum(), (p * dy * dy).sum()]])
    return np.array([mx, my]), cov


def grid_track(soft, valid, run_time, xs, ys, beta, speed, mask, smooth=True):
    """soft: runs × ny × nx soft cluster scores; valid: runs, False = no update; mask: ny × nx cells the person
    can be in. -> dict with filtered (and smoothed) posteriors, means and covariances."""
    n = len(soft)
    step = xs[1] - xs[0]
    gx, gy = np.meshgrid(xs, ys)
    dts = np.diff(run_time, prepend=run_time[0])
    log_lik = beta * np.log1p(soft)
    lik = np.exp(log_lik - log_lik.max(axis=(1, 2), keepdims=True))
    lik[~valid] = 1.0
    lik *= mask

    filt = np.empty_like(soft)
    p = mask / mask.sum()                                                 # start: anywhere in the support
    for k in range(n):
        p = _blur(p, speed * dts[k], step)
        p = p * lik[k]
        p /= p.sum()
        filt[k] = p

    out = {"filt": filt}
    out["filt_mean"], out["filt_cov"] = map(np.array, zip(*(_moments(q, gx, gy) for q in filt)))
    out["filt_map"] = np.array([(gx.ravel()[q.argmax()], gy.ravel()[q.argmax()]) for q in filt])
    if smooth:
        # backward pass: beta_k(x) = sum_x' T(x'|x) lik_{k+1}(x') beta_{k+1}(x'); T is a symmetric Gaussian blur
        sm = np.empty_like(soft)
        b = np.ones(soft.shape[1:])
        sm[-1] = filt[-1]
        for k in range(n - 2, -1, -1):
            b = _blur(lik[k + 1] * b, speed * dts[k + 1], step)
            q = filt[k] * b
            sm[k] = q / q.sum()
        out["smooth"] = sm
        out["smooth_mean"], out["smooth_cov"] = map(np.array, zip(*(_moments(q, gx, gy) for q in sm)))
    return out


# ---------------------------------------------------------------- Kalman filter

def kalman_track(z, score, valid, run_time, threshold, meas_std, accel_std, speed, smooth=True):
    """z: runs × 2 cluster locations; score: their cluster scores; valid: runs, False = prediction only.
    Constant-velocity model with white-noise acceleration accel_std (m/s^2); the initial velocity has std `speed`
    (m/s) per axis."""
    n = len(z)
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    x = np.r_[z[0], 0.0, 0.0]
    P = np.diag([1.0, 1.0, speed ** 2, speed ** 2])
    xs_f, Ps_f, xs_p, Ps_p, Fs, used = [], [], [], [], [], np.zeros(n, bool)
    for k in range(n):
        dt = run_time[k] - run_time[k - 1] if k else 0.0
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        Q = accel_std ** 2 * G @ G.T
        x, P = F @ x, F @ P @ F.T + Q
        xs_p.append(x)
        Ps_p.append(P)
        Fs.append(F)
        if valid[k]:
            R = meas_std ** 2 * threshold / score[k] * np.eye(2)
            S = H @ P @ H.T + R
            r = z[k] - H @ x
            if r @ np.linalg.solve(S, r) <= GATE ** 2:
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ r
                P = (np.eye(4) - K @ H) @ P
                used[k] = True
        xs_f.append(x)
        Ps_f.append(P)
    xs_f, Ps_f = np.array(xs_f), np.array(Ps_f)
    out = {"filt_mean": xs_f[:, :2], "filt_cov": Ps_f[:, :2, :2], "filt_vel": xs_f[:, 2:], "used": used}
    if smooth:
        xs_s, Ps_s = xs_f.copy(), Ps_f.copy()
        for k in range(n - 2, -1, -1):
            C = Ps_f[k] @ Fs[k + 1].T @ np.linalg.inv(Ps_p[k + 1])
            xs_s[k] = xs_f[k] + C @ (xs_s[k + 1] - xs_p[k + 1])
            Ps_s[k] = Ps_f[k] + C @ (Ps_s[k + 1] - Ps_p[k + 1]) @ C.T
        out["smooth_mean"], out["smooth_cov"] = xs_s[:, :2], Ps_s[:, :2, :2]
    return out
