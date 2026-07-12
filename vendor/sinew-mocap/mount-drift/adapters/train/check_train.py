# SPDX-License-Identifier: MIT
# Milestone 5: the Slang Adam training loss curve must track a torch Adam run from
# the same init/data (float32 both; small per-step reduction-order drift expected).
import subprocess, sys, os
import numpy as np
import torch

sys.path.insert(0, "E:/tmp/tic-calib")
from my_model import TIC  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
S, NIN, D, H, F, NOUT, STACK = 4, 12, 8, 2, 16, 6, 3
WLAYER = 4 * D * D + 2 * F * D + 9 * D + F
NW = D * NIN + D + (STACK + 2) * WLAYER + 2 * (NOUT * D + NOUT)
STEPS, LR = 40, 0.02
torch.manual_seed(3)

model = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=F).float()
x = torch.randn(1, S, NIN)
target = torch.randn(2 * NOUT)

ENC = ["norm1.weight", "norm1.bias",
       "mha.q_linear.weight", "mha.q_linear.bias", "mha.k_linear.weight", "mha.k_linear.bias",
       "mha.v_linear.weight", "mha.v_linear.bias", "mha.output.weight", "mha.output.bias",
       "norm2.weight", "norm2.bias",
       "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"]
sd = model.state_dict()
def W(n):
    return sd[n].numpy().ravel()
def enc_pack(p):
    return np.concatenate([W(f"{p}.{n}") for n in ENC])
wparts = [W("input_embedding_layer.embed.weight"), W("input_embedding_layer.embed.bias")]
for i in range(STACK):
    wparts.append(enc_pack(f"encoder_banckbone.{i}"))
wparts.append(enc_pack("TPM_global.encoder"))
wparts += [W("TPM_global.mapping.weight"), W("TPM_global.mapping.bias")]
wparts.append(enc_pack("TPM_local.encoder"))
wparts += [W("TPM_local.mapping.weight"), W("TPM_local.mapping.bias")]
wpack = np.concatenate(wparts)
assert len(wpack) == NW, (len(wpack), NW)

np.concatenate([wpack, x[0].numpy().ravel(), target.numpy()]).astype(np.float32).tofile(
    os.path.join(here, "init.bin"))

# torch Adam reference (loss recorded before each step's update)
opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
tloss = []
for _ in range(STEPS):
    g, l = model(x)
    loss = ((torch.cat([g[0], l[0]]) - target) ** 2).mean()
    tloss.append(loss.item())
    opt.zero_grad(); loss.backward(); opt.step()
tloss = np.array(tloss)

subprocess.run([os.path.join(here, "train.exe"), os.path.join(here, "train.spv"),
                str(NW), str(S * NIN), str(2 * NOUT), str(STEPS), str(LR)], cwd=here, check=True)
sloss = np.fromfile(os.path.join(here, "losses.bin"), dtype=np.float32)

print(f"{'step':>4} {'torch':>10} {'slang':>10}")
for i in [0, 1, 2, 5, 10, 20, STEPS - 1]:
    print(f"{i:>4} {tloss[i]:>10.5f} {sloss[i]:>10.5f}")
d0 = abs(sloss[0] - tloss[0])
dfin = abs(sloss[-1] - tloss[-1])
ok = (d0 < 1e-4) and (sloss[-1] < sloss[0] * 0.5) and (dfin < 0.05 * max(1.0, tloss[-1]) + 0.02)
print(f"step0 Δ={d0:.2e}  final torch={tloss[-1]:.5f} slang={sloss[-1]:.5f} Δ={dfin:.2e}")
print("PASS: Slang Adam tracks torch and the loss converges" if ok else "FAIL")
sys.exit(0 if ok else 1)
