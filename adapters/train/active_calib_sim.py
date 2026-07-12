# SPDX-License-Identifier: MIT
# Active ("adversarial") calibration vs a passive gait baseline, in the observability
# model of spec/Sinew/Impossibility.lean.  A bone's accelerometer constrains the mount
# only ⊥ to the gravity-in-bone direction d at the current pose (J = [d]_×, null space d).
# Accumulated mount Fisher info  I = I_prior + Σ_k (Id - d_k d_kᵀ)/σm².  The worst-axis
# mount std is sqrt(1/λ_min(I)) — the proof's zero-info direction when the d_k cluster.
#
#   passive: gait keeps the bone near vertical → d_k in a narrow cone → one DOF starved.
#   active : pick each next d_k along the current least-observed eigenvector of I (the
#            adversarial motion that attacks the weakest DOF) → I full-rank fastest.
# Reports worst-axis mount error (deg) vs #motions for both — the error against baseline.
import numpy as np

DEG = 180.0 / np.pi
sigma0 = np.radians(30.0)   # mount prior spread (OFFSET_DEG)
sigma_m = np.radians(2.0)   # per-pose gravity-direction measurement noise
N = 12                      # calibration motions
rng = np.random.default_rng(0)


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def info_of(ds):
    """Accumulated mount Fisher information from gravity-in-bone directions ds."""
    I = np.eye(3) / sigma0**2                       # prior
    for d in ds:
        d = d / (np.linalg.norm(d) + 1e-12)
        H = skew(d)                                  # ∂(gravity dir)/∂(mount δ)
        I = I + (H.T @ H) / sigma_m**2               # = (Id - d dᵀ)/σm²
    return I


def worst_axis_deg(ds):
    cov = np.linalg.inv(info_of(ds))
    return np.sqrt(np.max(np.linalg.eigvalsh(cov))) * DEG   # worst mount DOF std


def gait_dir():
    """Gravity-in-bone for a gait pose: a narrow sagittal arc near a fixed tilt — the
    bone mostly points one way, so the d_k cluster (the realistic gait starvation)."""
    tilt = np.radians(20.0) + np.radians(5.0) * (rng.random() - 0.5)   # ~15–25°
    az = np.radians(rng.normal(0, 5))                                  # sagittal cluster
    return np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])


def active_next(ds):
    """Adversarial pick: drive λ_min(I) up fastest.  A pose at gravity-dir d observes the
    plane ⊥ d, so to cover the least-observed mount DOF (smallest eigenvector e_min) point
    gravity ⊥ e_min — i.e. along the MOST-observed eigenvector; then the ⊥-plane spans the
    weak subspace.  Adversarial against the current weakest direction."""
    I = info_of(ds)
    w, V = np.linalg.eigh(I)
    return V[:, 2]                                    # ⊥ to the least-observed DOF


passive, active = [], []
dp, da = [], []
for k in range(1, N + 1):
    dp.append(gait_dir())
    da.append(active_next(da))
    passive.append(worst_axis_deg(dp))
    active.append(worst_axis_deg(da))

print(f"mount prior {np.degrees(sigma0):.0f}°  meas noise {np.degrees(sigma_m):.0f}°")
print(f"{'motions':>7} {'passive(gait)':>14} {'active(dance)':>14}")
for k in range(N):
    print(f"{k+1:>7} {passive[k]:>12.1f}° {active[k]:>12.1f}°")
print(f"\nfinal worst-axis mount error: passive {passive[-1]:.1f}°  active {active[-1]:.1f}°  "
      f"→ active is {passive[-1]/active[-1]:.1f}× tighter")
