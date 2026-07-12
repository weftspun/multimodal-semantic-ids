# SPDX-License-Identifier: MIT
# Train the engine-free calibrator on OUR OWN dataset (AddBiomechanics → ANNY, RCSF,
# rot+accel, drift+offset, phenotype-balanced).  Generates the init weights (torch TIC
# at NIN=180, NOUT=90), places caldata.bin (from sample_caldata.py), runs caltrain.exe,
# reports the loss curve.
import subprocess, sys, os, shutil
import numpy as np
import torch

sys.path.insert(0, "E:/tmp/tic-calib")
from my_model import TIC  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
S, NIN, NOUT, D, H, Fd, STACK = 32, 180, 90, 64, 4, 128, 2
STEPS, LR = 2000, 0.003
torch.manual_seed(0)

shutil.copy("E:/tmp/caldata.bin", os.path.join(here, "caldata.bin"))
# Window count follows the sampled file (X[S*NIN] + Y[2*NOUT] float32 per window).
NWIN = os.path.getsize(os.path.join(here, "caldata.bin")) // ((S * NIN + 2 * NOUT) * 4)

model = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=Fd).float()
sd = model.state_dict()
def W(n):
    return sd[n].numpy().ravel()
ENC = ["norm1.weight", "norm1.bias",
       "mha.q_linear.weight", "mha.q_linear.bias", "mha.k_linear.weight", "mha.k_linear.bias",
       "mha.v_linear.weight", "mha.v_linear.bias", "mha.output.weight", "mha.output.bias",
       "norm2.weight", "norm2.bias",
       "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"]
def enc(p):
    return np.concatenate([W(f"{p}.{n}") for n in ENC])
parts = [W("input_embedding_layer.embed.weight"), W("input_embedding_layer.embed.bias")]
for i in range(STACK):
    parts.append(enc(f"encoder_banckbone.{i}"))
parts.append(enc("TPM_global.encoder"))
parts += [W("TPM_global.mapping.weight"), W("TPM_global.mapping.bias")]
parts.append(enc("TPM_local.encoder"))
parts += [W("TPM_local.mapping.weight"), W("TPM_local.mapping.bias")]
np.concatenate(parts).astype(np.float32).tofile(os.path.join(here, "winit.bin"))

spv = lambda n: os.path.join(here, n)
subprocess.run([os.path.join(here, "caltrain.exe"), spv("gemm.spv"), spv("lin.spv"), spv("ln.spv"),
                spv("attn.spv"), spv("ew.spv"), spv("adamn.spv"), str(S), str(NIN), str(D), str(H),
                str(Fd), str(NOUT), str(STACK), str(STEPS), str(LR), str(NWIN)], cwd=here, check=True)
loss = np.fromfile(os.path.join(here, "losses.bin"), dtype=np.float32)
k = 50
ma = np.convolve(loss, np.ones(k) / k, mode="valid")
print(f"loss[0]={loss[0]:.4f}  ma_start={ma[0]:.4f}  ma_end={ma[-1]:.4f}")
print("PASS: engine-free calibrator trains on our own AddBiomechanics→ANNY dataset"
      if ma[-1] < ma[0] * 0.7 else "INCONCLUSIVE")
