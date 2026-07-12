# SPDX-License-Identifier: MIT
# General Adam check: 5 steps over a random tensor must match a numpy Adam reference
# (and torch.optim.Adam defaults: betas 0.9/0.999, eps 1e-8).
import subprocess, sys, os
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
n, lr, b1, b2, eps, steps = 257, 0.02, 0.9, 0.999, 1e-8, 5
rng = np.random.default_rng(0)
W0 = rng.standard_normal(n).astype(np.float32)
# fixed gradient each step (deterministic) so the driver and numpy see the same G
G = rng.standard_normal(n).astype(np.float32)

# numpy reference
W = W0.copy().astype(np.float64); m = np.zeros(n); v = np.zeros(n)
for t in range(1, steps + 1):
    g = G.astype(np.float64)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    W = W - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
ref = W.astype(np.float32)

# driver: writes init.bin [W0, G], runs `steps` adam_n updates, writes W
np.concatenate([W0, G]).astype(np.float32).tofile(os.path.join(here, "init.bin"))
subprocess.run([os.path.join(here, "adamn.exe"), os.path.join(here, "adamn.spv"),
                str(n), str(steps), str(lr)], cwd=here, check=True, stdout=subprocess.DEVNULL)
got = np.fromfile(os.path.join(here, "outputs.bin"), dtype=np.float32)

err = float(np.max(np.abs(got - ref)))
ok = err < 1e-4
print(f"adam {steps} steps max|Δ| = {err:.2e}  {'ok' if ok else 'FAIL'}")
print("PASS: general Adam matches numpy" if ok else "FAIL")
sys.exit(0 if ok else 1)
