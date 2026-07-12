"""Stage 0: the exact two-sided solver's accuracy ceiling on CONTINUOUS normal motion vs A/T/S/B.

Shows that ordinary continuous motion (a smooth SO(3) walk that sweeps >=2 axes per segment) lets the
exact algebraic solver recover the mount to ~0deg (noise-free) / the noise floor (~0.6deg at 1deg
noise) — the ceiling the reference-free net + refinement chase — whereas the 4 held A/T/S/B stances
leave most segments under-excited (~17deg). Also reports the per-segment excitation rank.  Pure numpy.
  python slangtrain/torch/solver_ceiling.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from calib_rank import (aa_to_R, geo_deg, log_vec, build_frame, target_dir,  # noqa: E402
                        solve_two_sided, NSENS)

rng = np.random.default_rng(0)
OFFSET_DEG, DRIFT_DEG = 30, 40


def rand_mount():
    return aa_to_R((rng.random(3) - 0.5) * 2 * np.radians(OFFSET_DEG))


def rand_drift():
    yaw = (rng.random() - 0.5) * 2 * np.radians(DRIFT_DEG)
    tilt = (rng.random(2) - 0.5) * 2 * np.radians(10)
    return aa_to_R(np.array([tilt[0], yaw, tilt[1]]))


def walk(base, nframes, step_deg, axes):
    """Continuous motion: cumulative small rotations about `axes` (a (k,3) basis). Full 3-axis basis
    = rich normal activity; a single axis = a hinge/gait-limited segment."""
    C, R = [], base.copy()
    for _ in range(nframes):
        ax = axes[rng.integers(len(axes))]
        R = aa_to_R(ax * np.radians(step_deg) * (rng.random() * 2 - 1)) @ R
        C.append(R.copy())
    return C


def excit_rank(Cs):  # rank of the relative-rotation axes (the observability metric, calib_rank:77)
    us = []
    for i in range(len(Cs)):
        for j in range(i + 1, len(Cs)):
            a = log_vec(Cs[i] @ Cs[j].T)
            if np.linalg.norm(a) > np.radians(5):
                us.append(a / np.linalg.norm(a))
    return 0 if len(us) < 2 else np.linalg.matrix_rank(np.array(us), tol=0.1)


def run(label, motion_fn, nframes, noise_deg, trials=40):
    errs, ranks = [], []
    for _ in range(trials):
        for s in range(NSENS):
            base = build_frame(target_dir(0, s))
            Cs = motion_fn(base, nframes)
            B, Dg = rand_mount(), rand_drift()
            Os = [Dg @ C @ B @ aa_to_R((rng.random(3) - 0.5) * 2 * np.radians(noise_deg)) for C in Cs]
            _, Bh = solve_two_sided(Cs, Os)
            errs.append(geo_deg(Bh, B))
            ranks.append(excit_rank(Cs))
    errs = np.array(errs)
    print(f"  {label:34s} R_BS err {errs.mean():6.2f}deg mean / {np.median(errs):5.2f} median"
          f"   excit-rank {np.mean(ranks):.2f}")


I3 = (1, 0, 0), (0, 1, 0), (0, 0, 1)
rich = [np.array(a, float) for a in I3]
hinge = [np.array((0.0, 1, 0))]  # one axis = under-excited (roll about long axis unobservable)

print("Exact two-sided solver — accuracy ceiling by motion type:")
print("\n noise-free (the math ceiling):")
run("continuous normal motion (3-axis)", lambda b, n: walk(b, n, 15, rich), 40, 0.0)
run("continuous hinge motion (1-axis)", lambda b, n: walk(b, n, 15, hinge), 40, 0.0)

print("\n with 1deg measurement noise (realistic floor):")
run("continuous normal motion (3-axis)", lambda b, n: walk(b, n, 15, rich), 40, 1.0)
run("continuous normal motion, 80 frames", lambda b, n: walk(b, n, 15, rich), 80, 1.0)


# A/T/S/B held stances, for contrast (the 4 canonical poses per segment)
def atsb_motion(base_unused, n_unused, s):
    return [build_frame(target_dir(p, s)) for p in range(4)]


errs = []
for _ in range(40):
    for s in range(NSENS):
        Cs = atsb_motion(None, None, s)
        B, Dg = rand_mount(), rand_drift()
        Os = [Dg @ C @ B @ aa_to_R((rng.random(3) - 0.5) * 2 * np.radians(1.0)) for C in Cs]
        _, Bh = solve_two_sided(Cs, Os)
        errs.append(geo_deg(Bh, B))
print(f"\n  {'A/T/S/B 4 held stances (1deg noise)':34s} R_BS err {np.mean(errs):6.2f}deg mean / "
      f"{np.median(errs):5.2f} median   (held poses under-excite)")
print("\nceiling: continuous normal motion -> ~0deg (noise-free) / noise-floor; A/T/S/B stays high.")
