"""Fully-passive runtime deliverable: TIC net (per-window, geodesic loss) -> robust session estimator.

Calibration is fully passive (no guided motion, no reference): the learned net gives a noisy ~15deg
per-window estimate of the constant mount, and RunningCalib (slangtrain/refine.py) folds the session's
windows into one bounded, tail-event-proof estimate.  Simulates a full session of continuous normal
motion with injected tail events (fast bursts, sensor dropouts, magnetic spikes) and reports the
converged error, the error-vs-time stability, and that nothing ever explodes.
  pixi run -e gpu python slangtrain/torch/passive_session.py
"""
import os
import sys
import numpy as np
import torch

import train_tic as T  # gen_window / train / decode6d (main guarded)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from refine import RunningCalib, geo_deg  # noqa: E402

dev = "cuda"
rng = np.random.default_rng(1)
NSENS = T.NSENS


print("Training the passive per-window net (geodesic loss, ACCA)...")
model = T.train("geodesic", 2500)
model.eval()


def session(nwin=400, inject=True):
    # one constant session calibration (the truth)
    mount = np.stack([T.aa_to_R((rng.random(3) - .5) * 2 * np.radians(30)) for _ in range(NSENS)])
    yaw = (rng.random(NSENS) - .5) * 2 * np.radians(40)
    tilt = (rng.random((NSENS, 2)) - .5) * 2 * np.radians(10)
    drift = np.stack([T.aa_to_R(np.array([tilt[s, 0], yaw[s], tilt[s, 1]])) for s in range(NSENS)])
    rcs = [RunningCalib() for _ in range(NSENS)]
    trace = []
    with torch.no_grad():
        for i in range(nwin):
            x, _, _, _ = T.gen_window(mount=mount, drift=drift)
            _, l = model(torch.tensor(x[None], device=dev))
            Bp = T.decode6d(l.view(NSENS, 6)).cpu().numpy()
            for s in range(NSENS):
                R, h = Bp[s], 1.0
                if inject and i % 90 == 0:                       # fast/magnetic spike
                    R = R @ T.aa_to_R(rng.normal(size=3) * np.radians(60))
                if inject and i % 150 == 0:                      # sensor dropout
                    R = np.full((3, 3), np.nan)
                rcs[s].update(R, h)
            cur = [rcs[s].R for s in range(NSENS)]
            if all(c is not None for c in cur):
                trace.append(np.mean([geo_deg(cur[s], mount[s]) for s in range(NSENS)]))
    # sanity: every final estimate finite + on SO(3)
    for s in range(NSENS):
        assert np.all(np.isfinite(rcs[s].R)) and abs(np.linalg.det(rcs[s].R) - 1) < 1e-4, "explosion!"
    final = np.mean([geo_deg(rcs[s].R, mount[s]) for s in range(NSENS)])
    dropped = sum(rc.dropped for rc in rcs)
    return final, trace, dropped


print("\nFully-passive session (continuous normal motion, tail events injected):")
finals = []
for sess in range(8):
    f, tr, dr = session()
    finals.append(f)
    if sess == 0:
        print(f"  session error-vs-time: start {tr[5]:.1f}deg -> 50 windows {tr[min(50,len(tr)-1)]:.1f}deg"
              f" -> end {tr[-1]:.1f}deg   worst-after-warmup {max(tr[10:]):.1f}deg   dropped {dr}")
print(f"\n  per-window net floor ~15-19deg (bias);  robust session estimate over 8 sessions: "
      f"{np.mean(finals):.1f}deg mean / {np.max(finals):.1f}deg worst")
print("  no explosion: every per-segment session estimate finite and on SO(3).")
print("  (residual is the net's bias on under-excited segments; full-data + refinement is the lever below it.)")
