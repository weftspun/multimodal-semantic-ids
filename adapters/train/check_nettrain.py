# SPDX-License-Identifier: MIT
# Finale: the engine-free real-dim tiled training loop must track a torch Adam run
# from the same init (float32 both; small per-step reduction-order drift expected).
import subprocess, sys, os
import numpy as np
import torch

sys.path.insert(0, "E:/tmp/tic-calib")
from my_model import TIC  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
S, NIN, D, H, F, NOUT, STACK = 16, 12, 32, 4, 64, 6, 3
STEPS, LR = 30, 0.02
torch.manual_seed(0)

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
def enc(p):
    return np.concatenate([W(f"{p}.{n}") for n in ENC])
parts = [W("input_embedding_layer.embed.weight"), W("input_embedding_layer.embed.bias")]
for i in range(STACK):
    parts.append(enc(f"encoder_banckbone.{i}"))
parts.append(enc("TPM_global.encoder"))
parts += [W("TPM_global.mapping.weight"), W("TPM_global.mapping.bias")]
parts.append(enc("TPM_local.encoder"))
parts += [W("TPM_local.mapping.weight"), W("TPM_local.mapping.bias")]
wpack = np.concatenate(parts)
np.concatenate([x[0].numpy().ravel(), wpack, target.numpy()]).astype(np.float32).tofile(
    os.path.join(here, "init.bin"))

opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
tl = []
for _ in range(STEPS):
    g, l = model(x)
    loss = ((torch.cat([g[0], l[0]]) - target) ** 2).mean()
    tl.append(loss.item())
    opt.zero_grad(); loss.backward(); opt.step()
tl = np.array(tl)

spv = lambda n: os.path.join(here, n)
subprocess.run([os.path.join(here, "nettrain.exe"), spv("gemm.spv"), spv("lin.spv"), spv("ln.spv"),
                spv("attn.spv"), spv("ew.spv"), spv("adamn.spv"), str(S), str(NIN), str(D), str(H),
                str(F), str(NOUT), str(STACK), str(STEPS), str(LR)], cwd=here, check=True)
sl = np.fromfile(os.path.join(here, "losses.bin"), dtype=np.float32)

print(f"{'step':>4} {'torch':>10} {'slang':>10}")
for i in [0, 1, 2, 5, 10, 20, STEPS - 1]:
    print(f"{i:>4} {tl[i]:>10.5f} {sl[i]:>10.5f}")
d0 = abs(sl[0] - tl[0])
ok = (d0 < 1e-4) and (sl[-1] < sl[0] * 0.6) and (abs(sl[-1] - tl[-1]) < 0.05 * max(1.0, tl[-1]) + 0.02)
print(f"step0 Δ={d0:.2e}  final torch={tl[-1]:.5f} slang={sl[-1]:.5f}")
print("PASS: engine-free tiled training tracks torch and converges" if ok else "FAIL")
sys.exit(0 if ok else 1)
