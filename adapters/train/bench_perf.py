# SPDX-License-Identifier: MIT
# Ground the speed claim: time the engine-free caltrain loop vs torch-CPU (single thread)
# and torch-CUDA at the real caltrain dims, batch=1 per step (caltrain's actual semantics).
# Vulkan init is cancelled by timing two step counts and taking the per-step difference.
# Measured on the 4090: a caltrain step is dispatch-latency-bound at ~42 ms vs ~9.7 ms for the
# same batch=1 TIC step in torch-CPU (~4-5x slower) — caltrain host-syncs after every kernel.
# The engine-free path's value is portability (one Vulkan dependency, no CUDA/PyTorch runtime),
# not speed; batched dispatch without per-kernel host sync is what would close the gap.
import subprocess, sys, os, time
import numpy as np
import torch

sys.path.insert(0, "E:/tmp/tic-calib")
from my_model import TIC  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(here, "caldata.npz"))
X, Y = d["X"].astype(np.float32), d["Y"].astype(np.float32)
S, NIN, NOUT = int(d["S"]), int(d["NIN"]), int(d["NOUT"])
D, H, Fd, STACK, LR = 64, 4, 128, 2, 0.004
NWIN = 800
print(f"dims: S={S} NIN={NIN} D={D} H={H} Fd={Fd} STACK={STACK} NOUT={NOUT}  batch=1/step")

# ---- dataset + matching weight init for caltrain (same packing as run_caltrain) ----
Xs, Ys = X[:NWIN], Y[:NWIN]
with open(os.path.join(here, "caldata.bin"), "wb") as fb:
    Xs.tofile(fb); Ys.tofile(fb)
torch.manual_seed(0)
model = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=Fd).float()
sd = model.state_dict()
W = lambda n: sd[n].numpy().ravel()
ENC = ["norm1.weight", "norm1.bias", "mha.q_linear.weight", "mha.q_linear.bias",
       "mha.k_linear.weight", "mha.k_linear.bias", "mha.v_linear.weight", "mha.v_linear.bias",
       "mha.output.weight", "mha.output.bias", "norm2.weight", "norm2.bias",
       "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"]
enc = lambda p: np.concatenate([W(f"{p}.{n}") for n in ENC])
parts = [W("input_embedding_layer.embed.weight"), W("input_embedding_layer.embed.bias")]
for i in range(STACK):
    parts.append(enc(f"encoder_banckbone.{i}"))
parts.append(enc("TPM_global.encoder")); parts += [W("TPM_global.mapping.weight"), W("TPM_global.mapping.bias")]
parts.append(enc("TPM_local.encoder")); parts += [W("TPM_local.mapping.weight"), W("TPM_local.mapping.bias")]
np.concatenate(parts).astype(np.float32).tofile(os.path.join(here, "winit.bin"))

spv = lambda n: os.path.join(here, n)
def caltrain(steps):
    t = time.perf_counter()
    subprocess.run([os.path.join(here, "caltrain.exe"), spv("gemm.spv"), spv("lin.spv"), spv("ln.spv"),
                    spv("attn.spv"), spv("ew.spv"), spv("adamn.spv"), str(S), str(NIN), str(D), str(H),
                    str(Fd), str(NOUT), str(STACK), str(steps), str(LR), str(NWIN)],
                   cwd=here, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - t

# two-point: (t[hi]-t[lo])/(hi-lo) = pure per-step loop time, init cancels
LO, HI = 200, 1200
caltrain(LO)  # warm shader cache
t_lo, t_hi = caltrain(LO), caltrain(HI)
slang_per = (t_hi - t_lo) / (HI - LO)
print(f"\nengine-free caltrain.exe (Vulkan, RTX 4090):")
print(f"  {LO} steps {t_lo*1000:.0f} ms, {HI} steps {t_hi*1000:.0f} ms  -> {slang_per*1e6:.1f} us/step, "
      f"init~{(t_lo-LO*slang_per)*1000:.0f} ms")
print(f"  1000 steps (loop only) = {slang_per*1000*1000:.0f} ms")

def torch_loop(dev, steps=1000):
    m = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=Fd).float().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    xb = torch.from_numpy(Xs).to(dev)
    yb = torch.from_numpy(Ys).to(dev)
    for w in range(5):  # warmup
        x = xb[w:w + 1]; g, l = m(x)
        loss = ((torch.cat([g[0], l[0]]) - yb[w]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for step in range(steps):
        x = xb[step % NWIN:step % NWIN + 1]
        g, l = m(x)
        loss = ((torch.cat([g[0], l[0]]) - yb[step % NWIN]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / steps

torch.set_num_threads(1)
cpu_per = torch_loop("cpu")
print(f"\ntorch-CPU single-thread: {cpu_per*1e6:.1f} us/step, 1000 steps = {cpu_per*1e6:.0f} ms")
print(f"  -> engine-free is {cpu_per/slang_per:.1f}x  (vs single-thread torch-CPU)")

print(f"\ntorch threads available; testing default-thread CPU too:")
torch.set_num_threads(0) if False else torch.set_num_threads(os.cpu_count())
mt_per = torch_loop("cpu")
print(f"torch-CPU {os.cpu_count()} threads: {mt_per*1e6:.1f} us/step  -> engine-free {mt_per/slang_per:.1f}x")

if torch.cuda.is_available():
    cuda_per = torch_loop("cuda")
    print(f"\ntorch-CUDA (RTX 4090): {cuda_per*1e6:.1f} us/step, 1000 steps = {cuda_per*1e6:.0f} ms")
    print(f"  -> engine-free is {cuda_per/slang_per:.2f}x torch-CUDA "
          f"({'faster' if cuda_per>slang_per else 'SLOWER'})")
else:
    print("\ntorch-CUDA: not available in this env")
