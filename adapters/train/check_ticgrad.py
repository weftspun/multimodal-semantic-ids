# SPDX-License-Identifier: MIT
# Milestone 4b gradient-check: reverse-mode through the whole TIC net must match
# torch.autograd for every weight under loss = MSE(cat(global, local), target).
import subprocess, sys, os
import numpy as np
import torch

sys.path.insert(0, "E:/tmp/tic-calib")
from my_model import TIC  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
S, NIN, D, H, F, NOUT, STACK = 4, 12, 8, 2, 16, 6, 3
torch.manual_seed(2)

model = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=F).double()
x = torch.randn(1, S, NIN, dtype=torch.float64)
target = torch.randn(2 * NOUT, dtype=torch.float64)
g, l = model(x)
loss = ((torch.cat([g[0], l[0]]) - target) ** 2).mean()
loss.backward()

sd = model.state_dict()
def W(n):
    return sd[n].numpy().ravel()

ENC = ["norm1.weight", "norm1.bias",
       "mha.q_linear.weight", "mha.q_linear.bias", "mha.k_linear.weight", "mha.k_linear.bias",
       "mha.v_linear.weight", "mha.v_linear.bias", "mha.output.weight", "mha.output.bias",
       "norm2.weight", "norm2.bias",
       "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"]

def enc_pack(prefix):
    return np.concatenate([W(f"{prefix}.{n}") for n in ENC])

# weight order matches ticgrad.slang
wparts = [W("input_embedding_layer.embed.weight"), W("input_embedding_layer.embed.bias")]
for i in range(STACK):
    wparts.append(enc_pack(f"encoder_banckbone.{i}"))
wparts.append(enc_pack("TPM_global.encoder"))
wparts.append(W("TPM_global.mapping.weight")); wparts.append(W("TPM_global.mapping.bias"))
wparts.append(enc_pack("TPM_local.encoder"))
wparts.append(W("TPM_local.mapping.weight")); wparts.append(W("TPM_local.mapping.bias"))
wpack = np.concatenate(wparts)

inp = np.concatenate([x[0].numpy().ravel(), wpack, target.numpy()]).astype(np.float32)
inp.tofile(os.path.join(here, "inputs.bin"))

out_count = 1 + len(wpack)
subprocess.run([os.path.join(here, "prims.exe"), os.path.join(here, "ticgrad.spv"),
                "tic_grad", str(len(inp)), str(out_count)], cwd=here, check=True,
               stdout=subprocess.DEVNULL)
out = np.fromfile(os.path.join(here, "outputs.bin"), dtype=np.float32)

ok = abs(out[0] - loss.item()) < 1e-4
print(f"loss fwd Δ = {abs(out[0] - loss.item()):.2e}")

params = dict(model.named_parameters())
pnames = ["input_embedding_layer.embed.weight", "input_embedding_layer.embed.bias"]
for i in range(STACK):
    pnames += [f"encoder_banckbone.{i}.{n}" for n in ENC]
pnames += [f"TPM_global.encoder.{n}" for n in ENC]
pnames += ["TPM_global.mapping.weight", "TPM_global.mapping.bias"]
pnames += [f"TPM_local.encoder.{n}" for n in ENC]
pnames += ["TPM_local.mapping.weight", "TPM_local.mapping.bias"]

o = 1
worst = 0.0
for n in pnames:
    g_ref = params[n].grad.numpy().ravel()
    got = out[o:o + len(g_ref)]; o += len(g_ref)
    err = float(np.max(np.abs(got - g_ref)))
    worst = max(worst, err)
    if err >= 1e-3:
        ok = False
        print(f"  FAIL d{n} max|Δ| = {err:.2e}")
print(f"worst weight-grad max|Δ| = {worst:.2e}")
print("PASS: full-net reverse-mode matches torch" if ok else "FAIL")
sys.exit(0 if ok else 1)
