# SPDX-License-Identifier: MIT
# Milestone 3 forward parity: the Slang EncoderLayer must match the Aplus torch
# EncoderLayer (dropout 0) for the same weights and input.
import subprocess, sys, os
import numpy as np
import torch

sys.path.insert(0, "E:/tmp/tic-calib")
from Aplus.models.transformer import EncoderLayer  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
S, D, H, F = 4, 8, 2, 16
torch.manual_seed(0)

layer = EncoderLayer(head_number=H, d_model=D, d_ff=F, dropout=0.0).double().eval()
x = torch.randn(1, S, D, dtype=torch.float64)
with torch.no_grad():
    y_ref = layer(x)[0].numpy()  # (S, D)

sd = layer.state_dict()
def W(name):
    return sd[name].numpy().ravel()

# pack inputs in enc.slang's read order
inp = np.concatenate([
    x[0].numpy().ravel(),
    W("norm1.weight"), W("norm1.bias"),
    W("mha.q_linear.weight"), W("mha.q_linear.bias"),
    W("mha.k_linear.weight"), W("mha.k_linear.bias"),
    W("mha.v_linear.weight"), W("mha.v_linear.bias"),
    W("mha.output.weight"), W("mha.output.bias"),
    W("norm2.weight"), W("norm2.bias"),
    W("mlp.fc1.weight"), W("mlp.fc1.bias"),
    W("mlp.fc2.weight"), W("mlp.fc2.bias"),
]).astype(np.float32)
inp.tofile(os.path.join(here, "inputs.bin"))

subprocess.run([os.path.join(here, "prims.exe"), os.path.join(here, "enc.spv"),
                "encoder_test", str(len(inp)), str(S * D)], cwd=here, check=True,
               stdout=subprocess.DEVNULL)
got = np.fromfile(os.path.join(here, "outputs.bin"), dtype=np.float32).reshape(S, D)

err = float(np.max(np.abs(got - y_ref)))
ok = err < 1e-4
print(f"EncoderLayer forward max|Δ| = {err:.2e}  {'ok' if ok else 'FAIL'}")
print("PASS: Slang EncoderLayer matches torch" if ok else "FAIL")
sys.exit(0 if ok else 1)
