# SPDX-License-Identifier: MIT
# Does a modal (cohort) mount prior lower the impossibility floor?  The floor lives on the
# UNOBSERVABLE mount DOF (the axis the motion never excites); its error = the prior spread on that
# DOF (spec/Sinew/Impossibility.lean).  A mixture prior helps ONLY if the strapping modes are
# separable in the OBSERVABLE subspace — then the coarse calibration classifies the mode and the
# mode pins the unobservable DOF to its tight within-mode spread.  If the modes differ ONLY along
# the unobservable DOF, the observable signature is identical, classification is at chance, and the
# prior cannot help.  This sims both, on a 3-DOF mount with a 2-DOF observable / 1-DOF unobservable
# split.
import numpy as np
DEG = 180 / np.pi
rng = np.random.default_rng(0)
N = 20000
K = 3                                   # strapping cohorts
sw = np.radians(8.0)                    # within-mode spread (tight)
sb = np.radians(25.0)                   # between-mode spread (broad)
obs = [0, 1]; unobs = 2                 # observable / unobservable mount axes
noise = np.radians(2.0)                 # observable recovery noise


def trial(separable):
    # mode centres: differ in observable axes (separable) or ONLY in the unobservable axis (not)
    mu = np.zeros((K, 3))
    if separable:
        mu[:, 0] = np.linspace(-sb, sb, K)          # distinct in an OBSERVABLE axis
        mu[:, unobs] = np.linspace(sb, -sb, K)       # ...and a distinct unobservable value per mode
    else:
        mu[:, unobs] = np.linspace(-sb, sb, K)       # distinct ONLY in the unobservable axis
    k = rng.integers(0, K, N)                        # true cohort per session
    d = mu[k] + rng.normal(0, sw, (N, 3))            # true mount = mode centre + tight jitter
    d_obs = d[:, obs] + rng.normal(0, noise, (N, 2)) # what calibration recovers (observable only)
    # classify the mode from the observable recovery
    dist = ((d_obs[:, None, :] - mu[None, :, obs]) ** 2).sum(-1)
    khat = dist.argmin(1)
    acc = (khat == k).mean()
    # unobservable-DOF error: unimodal prior → global mean (0); modal prior → classified mode's value
    err_uni = np.abs(d[:, unobs]).mean() * DEG
    err_mod = np.abs(d[:, unobs] - mu[khat, unobs]).mean() * DEG
    return acc, err_uni, err_mod


print("unobservable-DOF mount error (deg) — floor = prior spread on the unexcited axis")
print(f"{'modes differ in':>22} {'classify acc':>12} {'unimodal':>9} {'modal':>7}")
for sep, label in [(True, "observable subspace"), (False, "unobservable only")]:
    acc, eu, em = trial(sep)
    print(f"{label:>22} {acc*100:>10.0f}% {eu:>8.1f}° {em:>6.1f}°")
print(f"\nwithin-mode spread {np.degrees(sw):.0f}°, between-mode {np.degrees(sb):.0f}°.  Modal beats unimodal "
      "only when modes are observably separable; otherwise classification is at chance and it does not help.")
