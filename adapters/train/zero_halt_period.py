# SPDX-License-Identifier: MIT
# Zero-halt calibration over a PCVR dance period.  Constraint: ZERO halting (stop-and-pose)
# calibrations within the session.  From calib_rank.py's "PCVR dance (incidental)" run, the
# expressive dance excites arms/legs/trunk/hips/head multi-axis (mount ≈0.5–0.8°) but leaves
# the near-planted FEET rank-1 (≈28.7°, unobservable from dance).  So:
#   - excited bones: continuous recal from the ongoing dance → held at the floor, period = ∞.
#   - feet: cannot be re-estimated mid-dance without a halt → set once at a deliberate pre-session
#     setup (multi-axis, ≈0.6°), then drift at the mount slip rate.  Zero halts within the period
#     ⟺ feet stay within the error budget for the whole session: slip · T_session ≤ budget − setup.
import numpy as np

BUDGET = 5.0          # per-bone mount error budget (deg) before a recal would be needed
SETUP = 0.6           # feet error right after the one-time pre-session deliberate setup (deg)
DANCE = {"arms": 0.5, "legs": 0.8, "trunk": 0.8, "head": 0.8, "feet": 28.7}  # calib_rank incidental

print("per-bone calibration under PCVR dance (incidental, from calib_rank.py):")
for k, v in DANCE.items():
    tag = "continuous recal → held, period ∞" if v < BUDGET else "rank-1 (planted) → not recoverable mid-dance"
    print(f"  {k:6s} {v:5.1f}°   {tag}")

print(f"\nfeet validity period  T_valid = (budget {BUDGET}° − setup {SETUP}°) / slip rate :")
slips = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]                     # mount slip, deg/min
print(f"  {'slip°/min':>9} {'T_valid(min)':>13}")
for s in slips:
    print(f"  {s:>9.2f} {(BUDGET - SETUP)/s:>12.0f}")

print("\nzero-halt feasibility — feet error at session end = setup + slip·T (✓ if ≤ budget):")
sessions = [30, 60, 90, 120]                                  # dance-playlist session lengths (min)
hdr = "  slip°/min " + " ".join(f"{T:>5}m" for T in sessions)
print(hdr)
for s in slips:
    cells = []
    for T in sessions:
        e = SETUP + s * T
        cells.append(f"{e:4.0f}°{'✓' if e <= BUDGET else '✗'}")
    print(f"  {s:>9.2f} " + " ".join(f"{c:>6}" for c in cells))

req = (BUDGET - SETUP) / 90.0
print(f"\nfor a 90-min dance session with zero halts, the foot mount must slip ≤ {req:.2f}°/min")
print("→ a rigid / known-placement FOOT mount; the other 13 bones are maintained free by the dance.")
