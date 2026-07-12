# SPDX-License-Identifier: MIT
# Keystone check: the tiled-orchestrated EncoderLayer forward must match torch's
# Aplus EncoderLayer at real-ish dims.
import subprocess, sys, os
import numpy as np
import torch

sys.path.insert(0, "E:/tmp/tic-calib")
from Aplus.models.transformer import EncoderLayer  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
S, D, H, F = 32, 64, 4, 128
torch.manual_seed(0)

layer = EncoderLayer(head_number=H, d_model=D, d_ff=F, dropout=0.0).double().eval()
x = torch.randn(1, S, D, dtype=torch.float64)
with torch.no_grad():
    y_ref = layer(x)[0].numpy()

sd = layer.state_dict()
def W(n):
    return sd[n].numpy().ravel()
names = ["norm1.weight", "norm1.bias",
         "mha.q_linear.weight", "mha.q_linear.bias", "mha.k_linear.weight", "mha.k_linear.bias",
         "mha.v_linear.weight", "mha.v_linear.bias", "mha.output.weight", "mha.output.bias",
         "norm2.weight", "norm2.bias",
         "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"]
inp = np.concatenate([x[0].numpy().ravel()] + [W(n) for n in names]).astype(np.float32)
inp.tofile(os.path.join(here, "inputs.bin"))

subprocess.run([os.path.join(here, "enc_tiled.exe"),
                os.path.join(here, "gemm.spv"), os.path.join(here, "lin.spv"),
                os.path.join(here, "ln.spv"), os.path.join(here, "attn.spv"),
                os.path.join(here, "ew.spv"), str(S), str(D), str(H), str(F)],
               cwd=here, check=True, stdout=subprocess.DEVNULL)
got = np.fromfile(os.path.join(here, "outputs.bin"), dtype=np.float32).reshape(S, D)

err = float(np.max(np.abs(got - y_ref)))
ok = err < 1e-3
print(f"tiled EncoderLayer forward max|Δ| = {err:.2e}  {'ok' if ok else 'FAIL'}")
print("PASS: tiled-orchestrated EncoderLayer matches torch" if ok else "FAIL")
sys.exit(0 if ok else 1)
