# SPDX-License-Identifier: MIT
# Tiled layernorm check: fwd/bwd must match torch F.layer_norm.
import subprocess, sys, os
import numpy as np
import torch
import torch.nn.functional as F

here = os.path.dirname(os.path.abspath(__file__))
R, D = 16, 32
torch.manual_seed(0)

x = torch.randn(R, D, dtype=torch.float64, requires_grad=True)
g = torch.randn(D, dtype=torch.float64, requires_grad=True)
b = torch.randn(D, dtype=torch.float64, requires_grad=True)
dy = torch.randn(R, D, dtype=torch.float64)
y = F.layer_norm(x, (D,), g, b, eps=1e-5)
y.backward(dy)

inp = np.concatenate([x.detach().numpy().ravel(), g.detach().numpy(), b.detach().numpy(),
                      dy.numpy().ravel()]).astype(np.float32)
inp.tofile(os.path.join(here, "inputs.bin"))
subprocess.run([os.path.join(here, "ln.exe"), os.path.join(here, "ln.spv"), str(R), str(D)],
               cwd=here, check=True, stdout=subprocess.DEVNULL)
out = np.fromfile(os.path.join(here, "outputs.bin"), dtype=np.float32)

o = 0
def take(n):
    global o
    v = out[o:o + n]; o += n; return v
ok = True
def cmp(name, got, ref):
    global ok
    err = float(np.max(np.abs(got - ref.ravel())))
    p = err < 1e-3
    ok &= p
    print(f"{name:3s} max|Δ| = {err:.2e}  {'ok' if p else 'FAIL'}")

cmp("y", take(R * D), y.detach().numpy())
cmp("dx", take(R * D), x.grad.numpy())
cmp("dg", take(D), g.grad.numpy())
cmp("db", take(D), b.grad.numpy())
print("PASS: tiled layernorm matches torch" if ok else "FAIL")
sys.exit(0 if ok else 1)
