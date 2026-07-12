# SPDX-License-Identifier: MIT
# Is TRIALS=200 enough?  Sweep the trial count and re-seed; report each protocol's body-average
# mount error as mean ± std across seeds.  If the across-seed std at 200 is small and the mean
# matches larger N, 200 is sufficient (and we have the range, not just a point estimate).
#   pixi run -e gpu python calib_conv_check.py
import numpy as np
from calib_rank import run, PROTOCOLS

TRIALS = [50, 100, 200, 500, 1000]
SEEDS = range(6)
names = list(PROTOCOLS)
print("body-average mount error, mean ± std across 6 seeds:")
print(f"{'trials':>6}  " + "  ".join(f"{n[:20]:>20}" for n in names))
for T in TRIALS:
    col = {n: [] for n in names}
    for sd in SEEDS:
        r = run(T, sd)
        for n in names:
            col[n].append(float(r[n].mean()))
    print(f"{T:>6}  " + "  ".join(f"{np.mean(col[n]):6.2f}±{np.std(col[n]):4.2f}°" for n in names))
print("\nrule of thumb: 200 is sufficient where the ±std is small (≲0.3°) and the mean has stopped "
      "moving vs 500/1000.")
