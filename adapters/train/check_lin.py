# SPDX-License-Identifier: MIT
# Tiled linear check: the orchestrated gemm+bias fwd/bwd must match torch nn.Linear.
import subprocess, sys, os
import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
S, In, Out = 16, 32, 24
torch.manual_seed(0)

lin = torch.nn.Linear(In, Out).double()
X = torch.randn(S, In, dtype=torch.float64, requires_grad=True)
dY = torch.randn(S, Out, dtype=torch.float64)
Y = lin(X)
Y.backward(dY)

Wt = lin.weight.detach().numpy()   # (Out, In)
bt = lin.bias.detach().numpy()     # (Out,)
inp = np.concatenate([X.detach().numpy().ravel(), Wt.ravel(), bt.ravel(), dY.numpy().ravel()]).astype(np.float32)
inp.tofile(os.path.join(here, "inputs.bin"))

subprocess.run([os.path.join(here, "lin.exe"), os.path.join(here, "gemm.spv"),
                os.path.join(here, "lin.spv"), str(S), str(In), str(Out)], cwd=here, check=True,
               stdout=subprocess.DEVNULL)
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

cmp("Y", take(S * Out), Y.detach().numpy())
cmp("dX", take(S * In), X.grad.numpy())
cmp("dW", take(Out * In), lin.weight.grad.numpy())
cmp("db", take(Out), lin.bias.grad.numpy())
print("PASS: tiled linear fwd/bwd matches torch" if ok else "FAIL")
sys.exit(0 if ok else 1)
