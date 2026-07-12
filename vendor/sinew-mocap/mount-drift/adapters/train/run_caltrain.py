# SPDX-License-Identifier: MIT
# M7 step 2 driver: init weights (torch TIC, in caltrain's load order) + the dataset,
# run the engine-free caltrain loop on real 15-sensor data, report the loss curve.
import subprocess, sys, os
import numpy as np
import torch

sys.path.insert(0, os.environ.get("SINEW_TIC_REF", "/mnt/e/tmp/tic-calib"))
from my_model import TIC  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from pack_weights import pack_winit  # noqa: E402
d = np.load(os.path.join(here, "caldata.npz"))
X, Y = d["X"], d["Y"]
S, NIN, NOUT = int(d["S"]), int(d["NIN"]), int(d["NOUT"])
D, H, Fd, STACK = 64, 4, 128, 2
NWIN, STEPS, LR = 800, 1600, 0.004
torch.manual_seed(0)

# dataset subset → caldata.bin (X then Y)
Xs, Ys = X[:NWIN].astype(np.float32), Y[:NWIN].astype(np.float32)
with open(os.path.join(here, "caldata.bin"), "wb") as fb:
    Xs.tofile(fb); Ys.tofile(fb)

# weight init from a torch TIC, packed in caltrain's load() order (shared layout: pack_weights.py)
model = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=Fd).float()
pack_winit(model.state_dict(), STACK).tofile(os.path.join(here, "winit.bin"))

spv = lambda n: os.path.join(here, n)
subprocess.run([os.path.join(here, "caltrain.exe"), spv("gemm.spv"), spv("lin.spv"), spv("ln.spv"),
                spv("attn.spv"), spv("ew.spv"), spv("adamn.spv"), str(S), str(NIN), str(D), str(H),
                str(Fd), str(NOUT), str(STACK), str(STEPS), str(LR), str(NWIN)], cwd=here, check=True)
loss = np.fromfile(os.path.join(here, "losses.bin"), dtype=np.float32)
# moving average to read the trend through SGD noise
k = 50
ma = np.convolve(loss, np.ones(k) / k, mode="valid")
print(f"loss[0]={loss[0]:.4f}  ma_start={ma[0]:.4f}  ma_end={ma[-1]:.4f}")
ok = ma[-1] < ma[0] * 0.6
print("PASS: engine-free trainer learns the calibration on real data" if ok else
      "INCONCLUSIVE: loss did not drop enough")
sys.exit(0 if ok else 1)
