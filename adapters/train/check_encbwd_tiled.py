# SPDX-License-Identifier: MIT
# Backward keystone check: the tiled-orchestrated EncoderLayer backward must match
# torch EncoderLayer grads (dX + all 16 weight tensors) at real-ish dims.
import subprocess, sys, os
import numpy as np
import torch

sys.path.insert(0, "E:/tmp/tic-calib")
from Aplus.models.transformer import EncoderLayer  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
S, D, H, F = 16, 32, 4, 64
torch.manual_seed(3)

layer = EncoderLayer(head_number=H, d_model=D, d_ff=F, dropout=0.0).double()
X = torch.randn(1, S, D, dtype=torch.float64, requires_grad=True)
dOut = torch.randn(1, S, D, dtype=torch.float64)
layer(X).backward(dOut)

sd = layer.state_dict()
def W(n):
    return sd[n].numpy().ravel()
names = ["norm1.weight", "norm1.bias",
         "mha.q_linear.weight", "mha.q_linear.bias", "mha.k_linear.weight", "mha.k_linear.bias",
         "mha.v_linear.weight", "mha.v_linear.bias", "mha.output.weight", "mha.output.bias",
         "norm2.weight", "norm2.bias",
         "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"]
inp = np.concatenate([X[0].detach().numpy().ravel()] + [W(n) for n in names] +
                     [dOut[0].numpy().ravel()]).astype(np.float32)
inp.tofile(os.path.join(here, "inputs.bin"))

spv = lambda n: os.path.join(here, n)
subprocess.run([os.path.join(here, "encbwd.exe"), spv("gemm.spv"), spv("lin.spv"), spv("ln.spv"),
                spv("attn.spv"), spv("ew.spv"), str(S), str(D), str(H), str(F)],
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
    if not p:
        print(f"  FAIL d{name} max|Δ| = {err:.2e}")
    return err

errx = cmp("X", take(S * D), X.grad[0].numpy())
worst = errx
params = dict(layer.named_parameters())
for n in names:
    g = params[n].grad.numpy().ravel()
    worst = max(worst, cmp(n, take(len(g)), g))
print(f"dX max|Δ| = {errx:.2e}; worst overall = {worst:.2e}")
print("PASS: tiled EncoderLayer backward matches torch" if ok else "FAIL")
sys.exit(0 if ok else 1)
