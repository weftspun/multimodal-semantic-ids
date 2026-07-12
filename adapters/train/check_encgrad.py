# SPDX-License-Identifier: MIT
# Milestone 4 gradient-check: reverse-mode through the EncoderLayer must match
# torch.autograd for every weight (and the input) under loss = MSE(layer(X), target).
import subprocess, sys, os
import numpy as np
import torch

sys.path.insert(0, "E:/tmp/tic-calib")
from Aplus.models.transformer import EncoderLayer  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
S, D, H, F = 4, 8, 2, 16
torch.manual_seed(1)

layer = EncoderLayer(head_number=H, d_model=D, d_ff=F, dropout=0.0).double()
X = torch.randn(1, S, D, dtype=torch.float64, requires_grad=True)
target = torch.randn(1, S, D, dtype=torch.float64)
loss = ((layer(X) - target) ** 2).mean()
loss.backward()

sd = layer.state_dict()
def W(n):
    return sd[n].numpy().ravel()

# weight order matches encgrad.slang's loadLayer / storeLayerGrad
names = ["norm1.weight", "norm1.bias",
         "mha.q_linear.weight", "mha.q_linear.bias", "mha.k_linear.weight", "mha.k_linear.bias",
         "mha.v_linear.weight", "mha.v_linear.bias", "mha.output.weight", "mha.output.bias",
         "norm2.weight", "norm2.bias",
         "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"]
wpack = np.concatenate([W(n) for n in names])
inp = np.concatenate([X[0].detach().numpy().ravel(), wpack, target[0].numpy().ravel()]).astype(np.float32)
inp.tofile(os.path.join(here, "inputs.bin"))

grad_count = len(wpack)
out_count = 1 + grad_count + S * D
subprocess.run([os.path.join(here, "prims.exe"), os.path.join(here, "encgrad.spv"),
                "enc_grad", str(len(inp)), str(out_count)], cwd=here, check=True,
               stdout=subprocess.DEVNULL)
out = np.fromfile(os.path.join(here, "outputs.bin"), dtype=np.float32)

ok = True
print(f"loss fwd Δ = {abs(out[0] - loss.item()):.2e}")
ok &= abs(out[0] - loss.item()) < 1e-4
ggrad = out[1:1 + grad_count]
ref = {n: sd[n].grad if hasattr(sd[n], "grad") else None for n in names}
# torch grads live on layer params, not state_dict copies — fetch from named_parameters
params = dict(layer.named_parameters())
o = 0
for n in names:
    g = params[n].grad.numpy().ravel()
    got = ggrad[o:o + len(g)]; o += len(g)
    err = float(np.max(np.abs(got - g)))
    passed = err < 1e-3
    ok &= passed
    print(f"  d{n:22s} max|Δ| = {err:.2e}  {'ok' if passed else 'FAIL'}")
dX = out[1 + grad_count:].reshape(S, D)
errx = float(np.max(np.abs(dX - X.grad[0].numpy())))
ok &= errx < 1e-3
print(f"  dX max|Δ| = {errx:.2e}")
print("PASS: reverse-mode through EncoderLayer matches torch" if ok else "FAIL")
sys.exit(0 if ok else 1)
