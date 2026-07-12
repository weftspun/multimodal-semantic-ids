# SPDX-License-Identifier: MIT
# Milestone 2 gradient-check: torch is the oracle.  For each primitive, generate
# inputs, run the Slang fwd+bwd kernel on the GPU, and compare forward + input
# gradients to torch.autograd.
import subprocess, sys, os
import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)


def run_gpu(entry, inp, outN):
    inp.astype(np.float32).tofile(os.path.join(here, "inputs.bin"))
    subprocess.run([os.path.join(here, "prims.exe"), os.path.join(here, "prims.spv"),
                    entry, str(len(inp)), str(outN)], cwd=here, check=True,
                   stdout=subprocess.DEVNULL)
    return np.fromfile(os.path.join(here, "outputs.bin"), dtype=np.float32)


def cmp(name, got, ref):
    err = float(np.max(np.abs(got - ref.ravel())))
    ok = err < 1e-4
    print(f"  {name:3s} max|Δ| = {err:.2e}  {'ok' if ok else 'FAIL'}")
    return ok


all_ok = True

# linear: y = x·Wᵀ + b
print("linear:")
B, N, M = 2, 4, 3
x = torch.randn(B, N, dtype=torch.float64, requires_grad=True)
W = torch.randn(M, N, dtype=torch.float64, requires_grad=True)
b = torch.randn(M, dtype=torch.float64, requires_grad=True)
dy = torch.randn(B, M, dtype=torch.float64)
y = x @ W.T + b
y.backward(dy)
inp = np.concatenate([x.detach().numpy().ravel(), W.detach().numpy().ravel(),
                      b.detach().numpy().ravel(), dy.numpy().ravel()])
out = run_gpu("linear_test", inp, B*M + B*N + M*N + M)
o = 0
def take(n):
    global o
    v = out[o:o+n]; o += n; return v
all_ok &= cmp("y", take(B*M), y.detach().numpy())
all_ok &= cmp("dx", take(B*N), x.grad.numpy())
all_ok &= cmp("dW", take(M*N), W.grad.numpy())
all_ok &= cmp("db", take(M), b.grad.numpy())

# 6D → rotation (Zhou 2019)
print("sixd_to_rotmat:")
d = torch.randn(6, dtype=torch.float64, requires_grad=True)
dR = torch.randn(9, dtype=torch.float64)
a1, a2 = d[:3], d[3:]
b1 = a1 / a1.norm()
b2 = a2 - (b1 @ a2) * b1
b2 = b2 / b2.norm()
b3 = torch.linalg.cross(b1, b2)
R = torch.stack([b1, b2, b3], dim=1)  # columns
R.reshape(-1).backward(dR)
out = run_gpu("sixd_test", np.concatenate([d.detach().numpy(), dR.numpy()]), 9 + 6)
o = 0
all_ok &= cmp("R", take(9), R.detach().numpy())
all_ok &= cmp("dd", take(6), d.grad.numpy())

import torch.nn.functional as F
LN = 8

# GELU (tanh approximation)
print("gelu:")
x = torch.randn(LN, dtype=torch.float64, requires_grad=True)
dy = torch.randn(LN, dtype=torch.float64)
y = F.gelu(x, approximate="tanh")
y.backward(dy)
out = run_gpu("gelu_test", np.concatenate([x.detach().numpy(), dy.numpy()]), 2 * LN)
o = 0
all_ok &= cmp("y", take(LN), y.detach().numpy())
all_ok &= cmp("dx", take(LN), x.grad.numpy())

# LayerNorm
print("layernorm:")
x = torch.randn(LN, dtype=torch.float64, requires_grad=True)
g = torch.randn(LN, dtype=torch.float64, requires_grad=True)
b = torch.randn(LN, dtype=torch.float64, requires_grad=True)
dy = torch.randn(LN, dtype=torch.float64)
y = F.layer_norm(x, (LN,), g, b, eps=1e-5)
y.backward(dy)
out = run_gpu("layernorm_test", np.concatenate([x.detach().numpy(), g.detach().numpy(),
              b.detach().numpy(), dy.numpy()]), 4 * LN)
o = 0
all_ok &= cmp("y", take(LN), y.detach().numpy())
all_ok &= cmp("dx", take(LN), x.grad.numpy())
all_ok &= cmp("dg", take(LN), g.grad.numpy())
all_ok &= cmp("db", take(LN), b.grad.numpy())

# Softmax
print("softmax:")
x = torch.randn(LN, dtype=torch.float64, requires_grad=True)
dy = torch.randn(LN, dtype=torch.float64)
y = F.softmax(x, dim=0)
y.backward(dy)
out = run_gpu("softmax_test", np.concatenate([x.detach().numpy(), dy.numpy()]), 2 * LN)
o = 0
all_ok &= cmp("y", take(LN), y.detach().numpy())
all_ok &= cmp("dx", take(LN), x.grad.numpy())

# MSE
print("mse:")
p = torch.randn(LN, dtype=torch.float64, requires_grad=True)
t = torch.randn(LN, dtype=torch.float64)
loss = ((p - t) ** 2).mean()
loss.backward()
out = run_gpu("mse_test", np.concatenate([p.detach().numpy(), t.numpy()]), 1 + LN)
o = 0
all_ok &= cmp("loss", take(1), loss.detach().numpy().reshape(1))
all_ok &= cmp("dp", take(LN), p.grad.numpy())

print("PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
