# SPDX-License-Identifier: MIT
# Tiled attention check: multi-head scaled-dot-product attention fwd/bwd must match
# torch (head h = columns [h*DK,(h+1)*DK), matching the Aplus head split).
import subprocess, sys, os, math
import numpy as np
import torch
import torch.nn.functional as F

here = os.path.dirname(os.path.abspath(__file__))
S, D, H = 8, 16, 2
DK = D // H
torch.manual_seed(0)

Q = torch.randn(S, D, dtype=torch.float64, requires_grad=True)
K = torch.randn(S, D, dtype=torch.float64, requires_grad=True)
V = torch.randn(S, D, dtype=torch.float64, requires_grad=True)
dOut = torch.randn(S, D, dtype=torch.float64)

def heads(t):
    return t.view(S, H, DK).permute(1, 0, 2)  # (H,S,DK)
qh, kh, vh = heads(Q), heads(K), heads(V)
sim = qh @ kh.transpose(-2, -1) / math.sqrt(DK)   # (H,S,S)
P = F.softmax(sim, dim=-1)
outh = P @ vh                                       # (H,S,DK)
Out = outh.permute(1, 0, 2).reshape(S, D)
Out.backward(dOut)

inp = np.concatenate([Q.detach().numpy().ravel(), K.detach().numpy().ravel(),
                      V.detach().numpy().ravel(), dOut.numpy().ravel()]).astype(np.float32)
inp.tofile(os.path.join(here, "inputs.bin"))
subprocess.run([os.path.join(here, "attn.exe"), os.path.join(here, "attn.spv"),
                str(S), str(D), str(H)], cwd=here, check=True, stdout=subprocess.DEVNULL)
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
    print(f"{name:4s} max|Δ| = {err:.2e}  {'ok' if p else 'FAIL'}")

cmp("Out", take(S * D), Out.detach().numpy())
cmp("dQ", take(S * D), Q.grad.numpy())
cmp("dK", take(S * D), K.grad.numpy())
cmp("dV", take(S * D), V.grad.numpy())
print("PASS: tiled attention fwd/bwd matches torch" if ok else "FAIL")
sys.exit(0 if ok else 1)
