# SPDX-License-Identifier: MIT
# Session-long robust calibration estimator (Stage 2/3): combine the per-window mount/drift estimates
# of a continuous VR session into one stable calibration that CANNOT explode, holds for the whole
# session, and has headroom for rare events.
#
# The mount R_BS and drift R_DG are constant per session, so the right estimator is a robust running
# SO(3) mean of the per-window estimates:
#   * bounded by construction — an average of rotations, re-projected to SO(3), can never diverge or
#     produce a non-rotation/NaN;
#   * outlier-rejecting — each window is weighted by its agreement with the running consensus (Huber)
#     and its observability health, so a fast-motion burst / dropout / magnetic spike / degenerate
#     (under-excited) window is down-weighted, not absorbed;
#   * last-good fallback — a non-finite or far-outlier window is dropped and the prior estimate held;
#   * forgetting — a slow decay lets a genuine slow change (re-seated strap) be tracked without
#     destabilising.
# This is the variance-reduction lever (~1/sqrt(N)) AND the session-stability mechanism in one.
import numpy as np


def project_so3(M):  # nearest rotation to a 3x3 matrix (SVD); the bound that prevents explosion
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def geo_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


class RunningCalib:
    """Robust running SO(3) estimate of one constant rotation from a stream of per-window estimates.

    update(R_est, health) -> current estimate.  health in [0,1] is the window's observability/quality
    (e.g. excitation rank / 3, or solver confidence); 0 means skip.  Returns the held estimate always.
    """

    def __init__(self, huber_deg=20.0, forget=0.995, reject_deg=70.0, warmup=5):
        self.M = None          # running (unprojected) weighted sum -> projects to the estimate
        self.R = None          # current SO(3) estimate (last-good)
        self.wsum = 0.0
        self.huber = huber_deg
        self.forget = forget
        self.reject = reject_deg
        self.warmup = warmup
        self.n = 0
        self.dropped = 0

    def update(self, R_est, health=1.0):
        # Guard 1: reject non-finite / non-rotation outright (last-good fallback).
        if R_est is None or not np.all(np.isfinite(R_est)) or health <= 0.0:
            self.dropped += 1
            return self.R
        R_est = project_so3(R_est)              # Guard 2: force onto SO(3)
        # Guard 3: consensus gate — after warmup, reject windows far from the running estimate.
        if self.R is not None and self.n >= self.warmup:
            dev = geo_deg(self.R, R_est)
            if dev > self.reject:
                self.dropped += 1
                return self.R
            w = health * min(1.0, self.huber / max(dev, 1e-6))   # Huber down-weight by deviation
        else:
            w = health
        # Robust weighted running mean with forgetting (bounded: it's an average).
        self.M = w * R_est if self.M is None else self.forget * self.M + w * R_est
        self.wsum = w if self.wsum == 0.0 else self.forget * self.wsum + w
        self.R = project_so3(self.M / self.wsum)
        self.n += 1
        return self.R


if __name__ == "__main__":  # self-test: bounded + tail-event resilient
    rng = np.random.default_rng(0)

    def aa(v):
        th = np.linalg.norm(v)
        if th < 1e-12:
            return np.eye(3)
        k = v / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

    truth = aa(rng.normal(size=3) * 0.4)
    rc = RunningCalib()
    pre = []
    for i in range(2000):
        # normal noisy window (~10° noise), with injected tail events
        R = truth @ aa(rng.normal(size=3) * np.radians(10))
        h = 1.0
        if i % 113 == 0:                      # fast-motion / magnetic spike: 90° off
            R = truth @ aa(rng.normal(size=3) * np.radians(90))
        if i % 200 == 0:                      # sensor dropout: NaN
            R = np.full((3, 3), np.nan)
        if i % 271 == 0:                      # degenerate window: health 0
            h = 0.0
        est = rc.update(R, h)
        if est is None:
            continue                          # no valid window yet (held last-good = nothing)
        assert np.all(np.isfinite(est)) and abs(np.linalg.det(est) - 1) < 1e-5, "explosion!"
        pre.append(geo_deg(est, truth))
    print(f"running estimate after 2000 windows: {geo_deg(rc.R, truth):.2f}° from truth "
          f"(single-window noise ~10°), {rc.dropped} windows dropped")
    print(f"worst excursion after warmup: {max(pre[10:]):.2f}°   final: {pre[-1]:.2f}°")
    print("no explosion: every estimate finite and on SO(3).")
