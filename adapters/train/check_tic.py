# SPDX-License-Identifier: MIT
# Milestone 3b forward parity: the Slang full TIC forward must match torch's
# TIC.forward (global_shift, local_shift) for the same weights and input.
import subprocess, sys, os
import numpy as np
import torch

sys.path.insert(0, "E:/tmp/tic-calib")
from my_model import TIC  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
S, NIN, D, H, F, NOUT, STACK = 4, 12, 8, 2, 16, 6, 3
torch.manual_seed(0)

model = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=F).double().eval()
x = torch.randn(1, S, NIN, dtype=torch.float64)
with torch.no_grad():
    g, l = model(x)
ref = np.concatenate([g[0].numpy(), l[0].numpy()])

sd = model.state_dict()
def W(name):
    return sd[name].numpy().ravel()

def enc_pack(prefix):
    return np.concatenate([
        W(f"{prefix}.norm1.weight"), W(f"{prefix}.norm1.bias"),
        W(f"{prefix}.mha.q_linear.weight"), W(f"{prefix}.mha.q_linear.bias"),
        W(f"{prefix}.mha.k_linear.weight"), W(f"{prefix}.mha.k_linear.bias"),
        W(f"{prefix}.mha.v_linear.weight"), W(f"{prefix}.mha.v_linear.bias"),
        W(f"{prefix}.mha.output.weight"), W(f"{prefix}.mha.output.bias"),
        W(f"{prefix}.norm2.weight"), W(f"{prefix}.norm2.bias"),
        W(f"{prefix}.mlp.fc1.weight"), W(f"{prefix}.mlp.fc1.bias"),
        W(f"{prefix}.mlp.fc2.weight"), W(f"{prefix}.mlp.fc2.bias"),
    ])

parts = [x[0].numpy().ravel(),
         W("input_embedding_layer.embed.weight"), W("input_embedding_layer.embed.bias")]
for i in range(STACK):
    parts.append(enc_pack(f"encoder_banckbone.{i}"))
parts.append(enc_pack("TPM_global.encoder"))
parts.append(W("TPM_global.mapping.weight")); parts.append(W("TPM_global.mapping.bias"))
parts.append(enc_pack("TPM_local.encoder"))
parts.append(W("TPM_local.mapping.weight")); parts.append(W("TPM_local.mapping.bias"))
inp = np.concatenate(parts).astype(np.float32)
inp.tofile(os.path.join(here, "inputs.bin"))

subprocess.run([os.path.join(here, "prims.exe"), os.path.join(here, "tic.spv"),
                "tic_test", str(len(inp)), str(2 * NOUT)], cwd=here, check=True,
               stdout=subprocess.DEVNULL)
got = np.fromfile(os.path.join(here, "outputs.bin"), dtype=np.float32)

err = float(np.max(np.abs(got - ref)))
ok = err < 1e-4
print(f"TIC forward (global+local) max|Δ| = {err:.2e}  {'ok' if ok else 'FAIL'}")
print("PASS: Slang full TIC forward matches torch" if ok else "FAIL")
sys.exit(0 if ok else 1)
