"""Full-TIC parity: our slangtorch-assembled TIC (tk_tic) vs the torch reference my_model.TIC.
Forward (<2e-3), backward grads on every weight (<5e-3, fp32 over a deep net), and one Adam step.
  pixi run -e gpu python slangtrain/torch/check_tic_tk.py
"""
import os
import sys
import torch
import tk_ops as tk
from tk_tic import TICKernels

sys.path.insert(0, os.environ.get("SINEW_TIC_REF", "/mnt/e/tmp/tic-calib"))
from my_model import TIC  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
S, NIN, D, H, F, STACK, NOUT = 8, 24, 32, 4, 64, 2, 12
fails = []


def check(name, got, ref, tol):
    err = (got - ref).abs().max().item()
    ok = err < tol
    print(f"  {'OK ' if ok else 'FAIL'} {name:22s} max|err|={err:.2e}  (tol {tol:.0e})")
    if not ok:
        fails.append(name)


model = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=F).to(dev).float().eval()
tic = TICKernels(model.state_dict(), STACK, D, H, F, requires_grad=True)

x = torch.randn(S, NIN, device=dev)
xt = x.unsqueeze(0).clone().requires_grad_(True)
xo = x.clone().requires_grad_(True)

print("Full TIC forward vs my_model.TIC:")
gt, lt = model(xt)
go, lo = tic.forward(xo)
check("global_shift", go, gt.squeeze(0), 2e-3)
check("local_shift", lo, lt.squeeze(0), 2e-3)

print("\nFull TIC backward (per-weight grads) vs torch:")
rg = torch.randn(NOUT, device=dev)
rl = torch.randn(NOUT, device=dev)
model.zero_grad()
((gt.squeeze(0) * rg).sum() + (lt.squeeze(0) * rl).sum()).backward()
((go * rg).sum() + (lo * rl).sum()).backward()
worst = 0.0
for name, p in model.named_parameters():
    if p.grad is None or name not in tic.w:
        continue
    g_ours = tic.w[name].grad
    if g_ours is None:
        fails.append(name + "(no grad)")
        continue
    worst = max(worst, (g_ours - p.grad).abs().max().item())
check("max over all weights", torch.tensor(worst), torch.tensor(0.0), 5e-3)
check("input grad dx", xo.grad, xt.grad.squeeze(0), 5e-3)

print("\nAdam step vs torch.optim.Adam (single step):")
w0 = torch.randn(256, device=dev)
g0 = torch.randn(256, device=dev)
wo = w0.clone()
Mo = torch.zeros_like(wo)
Vo = torch.zeros_like(wo)
tk.adam_step(wo, g0.clone(), Mo, Vo, t=1, lr=1e-3)
wt = w0.clone().requires_grad_(True)
opt = torch.optim.Adam([wt], lr=1e-3)
wt.grad = g0.clone()
opt.step()
check("adam_n update", wo, wt.detach(), 1e-5)

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
raise SystemExit(1 if fails else 0)
