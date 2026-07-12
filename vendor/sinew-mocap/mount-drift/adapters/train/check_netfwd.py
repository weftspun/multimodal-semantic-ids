# SPDX-License-Identifier: MIT
# Full real-dim forward check: the tiled-orchestrated TIC forward must match torch
# TIC.forward (global_shift, local_shift) at moderate dims.
import subprocess, sys, os
import numpy as np
import torch

sys.path.insert(0, "E:/tmp/tic-calib")
from my_model import TIC  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
S, NIN, D, H, F, NOUT, STACK = 16, 12, 32, 4, 64, 6, 3
torch.manual_seed(0)

model = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=F).double().eval()
x = torch.randn(1, S, NIN, dtype=torch.float64)
with torch.no_grad():
    g, l = model(x)
ref = np.concatenate([g[0].numpy(), l[0].numpy()])

sd = model.state_dict()
def W(n):
    return sd[n].numpy().ravel()
ENC = ["norm1.weight", "norm1.bias",
       "mha.q_linear.weight", "mha.q_linear.bias", "mha.k_linear.weight", "mha.k_linear.bias",
       "mha.v_linear.weight", "mha.v_linear.bias", "mha.output.weight", "mha.output.bias",
       "norm2.weight", "norm2.bias",
       "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"]
def enc_pack(p):
    return np.concatenate([W(f"{p}.{n}") for n in ENC])
parts = [x[0].numpy().ravel(),
         W("input_embedding_layer.embed.weight"), W("input_embedding_layer.embed.bias")]
for i in range(STACK):
    parts.append(enc_pack(f"encoder_banckbone.{i}"))
parts.append(enc_pack("TPM_global.encoder"))
parts += [W("TPM_global.mapping.weight"), W("TPM_global.mapping.bias")]
parts.append(enc_pack("TPM_local.encoder"))
parts += [W("TPM_local.mapping.weight"), W("TPM_local.mapping.bias")]
np.concatenate(parts).astype(np.float32).tofile(os.path.join(here, "inputs.bin"))

spv = lambda n: os.path.join(here, n)
subprocess.run([os.path.join(here, "netfwd.exe"), spv("gemm.spv"), spv("lin.spv"), spv("ln.spv"),
                spv("attn.spv"), spv("ew.spv"), str(S), str(NIN), str(D), str(H), str(F),
                str(NOUT), str(STACK)], cwd=here, check=True, stdout=subprocess.DEVNULL)
got = np.fromfile(os.path.join(here, "outputs.bin"), dtype=np.float32)

err = float(np.max(np.abs(got - ref)))
ok = err < 1e-3
print(f"tiled TIC forward (global+local) max|Δ| = {err:.2e}  {'ok' if ok else 'FAIL'}")
print("PASS: tiled full TIC forward matches torch" if ok else "FAIL")
sys.exit(0 if ok else 1)
