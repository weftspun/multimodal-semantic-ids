# SPDX-License-Identifier: MIT
# Elementwise check: add / relu_fwd / relu_bwd vs numpy.
import subprocess, sys, os
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
n = 1000
rng = np.random.default_rng(0)


def run(entry, A, B):
    np.concatenate([A, B]).astype(np.float32).tofile(os.path.join(here, "inputs.bin"))
    subprocess.run([os.path.join(here, "ew.exe"), os.path.join(here, "ew.spv"), entry, str(n)],
                   cwd=here, check=True, stdout=subprocess.DEVNULL)
    return np.fromfile(os.path.join(here, "outputs.bin"), dtype=np.float32)


ok = True
A = rng.standard_normal(n).astype(np.float32); B = rng.standard_normal(n).astype(np.float32)
err = float(np.max(np.abs(run("add", A, B) - (A + B))))
print(f"add      max|Δ| = {err:.2e}"); ok &= err < 1e-5
err = float(np.max(np.abs(run("relu_fwd", A, B) - np.maximum(0, A))))
print(f"relu_fwd max|Δ| = {err:.2e}"); ok &= err < 1e-5
err = float(np.max(np.abs(run("relu_bwd", A, B) - np.where(A > 0, B, 0))))
print(f"relu_bwd max|Δ| = {err:.2e}"); ok &= err < 1e-5
print("PASS: elementwise kernels match numpy" if ok else "FAIL")
sys.exit(0 if ok else 1)
