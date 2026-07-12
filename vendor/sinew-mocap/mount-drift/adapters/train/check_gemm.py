# SPDX-License-Identifier: MIT
# Tiled GEMM check: the three forms (nn, nt, tn) must match torch at real-ish dims.
import subprocess, sys, os
import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
M, N, K = 64, 48, 32
torch.manual_seed(0)


def run(entry, A, B):
    np.concatenate([A.ravel(), B.ravel()]).astype(np.float32).tofile(os.path.join(here, "inputs.bin"))
    subprocess.run([os.path.join(here, "gemm.exe"), os.path.join(here, "gemm.spv"),
                    entry, str(M), str(N), str(K)], cwd=here, check=True, stdout=subprocess.DEVNULL)
    return np.fromfile(os.path.join(here, "outputs.bin"), dtype=np.float32).reshape(M, N)


ok = True
# nn: C = A(M,K) @ B(K,N)
A = np.random.randn(M, K).astype(np.float32); B = np.random.randn(K, N).astype(np.float32)
ref = A @ B
err = float(np.max(np.abs(run("gemm_nn", A, B) - ref)))
print(f"nn max|Δ| = {err:.2e}"); ok &= err < 1e-3
# nt: C = A(M,K) @ B(N,K)^T
A = np.random.randn(M, K).astype(np.float32); B = np.random.randn(N, K).astype(np.float32)
ref = A @ B.T
err = float(np.max(np.abs(run("gemm_nt", A, B) - ref)))
print(f"nt max|Δ| = {err:.2e}"); ok &= err < 1e-3
# tn: C = A(K,M)^T @ B(K,N)
A = np.random.randn(K, M).astype(np.float32); B = np.random.randn(K, N).astype(np.float32)
ref = A.T @ B
err = float(np.max(np.abs(run("gemm_tn", A, B) - ref)))
print(f"tn max|Δ| = {err:.2e}"); ok &= err < 1e-3

print("PASS: tiled GEMM matches torch" if ok else "FAIL")
sys.exit(0 if ok else 1)
