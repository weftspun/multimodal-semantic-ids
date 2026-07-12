"""Training-step benchmark (primary metric): full fwd+bwd+adam, ours (slangtorch kernels) vs torch.

Our TICKernels currently processes ONE window per step (T7 will batch).  This measures:
  (a) ours, 1 window/step  vs  (b) torch, 1 window/step  vs  (c) torch, batch B/step,
so the per-window overhead and the batching headroom (the biggest training lever) are both visible.
  pixi run -e gpu python slangtrain/torch/bench_train.py
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
S, NIN, D, H, F, STACK, NOUT = 32, 180, 64, 4, 128, 2, 90
B = 256


def time_ms(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


model = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=F).to(dev).float()
rg, rl = torch.randn(NOUT, device=dev), torch.randn(NOUT, device=dev)

# (a) ours, 1 window/step
tic = TICKernels(model.state_dict(), STACK, D, H, F, requires_grad=True)
mv = {k: (torch.zeros_like(w), torch.zeros_like(w)) for k, w in tic.w.items()}
x1 = torch.randn(S, NIN, device=dev)
_t = [0]


def ours_step():
    _t[0] += 1
    for w in tic.w.values():
        w.grad = None
    g, l = tic.forward(x1)
    ((g * rg).sum() + (l * rl).sum()).backward()
    with torch.no_grad():
        for k, w in tic.w.items():
            tk.adam_step(w.data, w.grad, mv[k][0], mv[k][1], _t[0])


# torch step at batch BB
def torch_stepper(BB):
    m = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=F).to(dev).float()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    xb = torch.randn(BB, S, NIN, device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        g, l = m(xb)
        ((g * rg).sum() + (l * rl).sum()).backward()
        opt.step()
    return step


# (d) ours, B windows/step (T7 batched)
ticb = TICKernels(model.state_dict(), STACK, D, H, F, requires_grad=True)
mvb = {k: (torch.zeros_like(w), torch.zeros_like(w)) for k, w in ticb.w.items()}
Xb = torch.randn(B, S, NIN, device=dev)
_tb = [0]


def ours_step_b():
    _tb[0] += 1
    for w in ticb.w.values():
        w.grad = None
    g, l = ticb.forward_batch(Xb)
    ((g * rg).sum() + (l * rl).sum()).backward()
    with torch.no_grad():
        for k, w in ticb.w.items():
            tk.adam_step(w.data, w.grad, mvb[k][0], mvb[k][1], _tb[0])


# one-time batched-forward parity vs torch (same weights)
with torch.no_grad():
    go_b, lo_b = ticb.forward_batch(Xb)
    gt_b, lt_b = model(Xb)
    perr = max((go_b - gt_b).abs().max().item(), (lo_b - lt_b).abs().max().item())
print(f"batched forward parity vs torch: max|err|={perr:.2e}  ({'OK' if perr < 2e-3 else 'FAIL'})\n")

a = time_ms(ours_step)
b = time_ms(torch_stepper(1))
c = time_ms(torch_stepper(B))
d = time_ms(ours_step_b)
print(f"train step (fwd+bwd+adam)   {torch.cuda.get_device_name(0)}   dims S={S} D={D} F={F} STACK={STACK}")
print(f"  (a) ours   1 window /step : {a:8.3f} ms   ({1/a*1000:9.0f} windows/s)")
print(f"  (b) torch  1 window /step : {b:8.3f} ms   ({1/b*1000:9.0f} windows/s)")
print(f"  (c) torch  {B} windows/step: {c:8.3f} ms   ({B/c*1000:9.0f} windows/s)")
print(f"  (d) ours   {B} windows/step: {d:8.3f} ms   ({B/d*1000:9.0f} windows/s)")
print(f"\n  batching lifted OURS {(B/d)/(1/a):.0f}× (a->d);  ours/torch throughput @batch {(B/d)/(B/c):.2f}×")

# ── fraction of the 4090's theoretical peak ──────────────────────────────────
# Forward MACs per window (the matmul-dominated cost); training ≈ 3× (fwd + dX + dW).
enc = STACK + 2  # backbone layers + the two TPM-head encoder layers
fwd_macs = (S * NIN * D                                   # embedder
            + enc * (3 * S * D * D                        # Q,K,V
                     + 2 * S * S * D                       # attn QKᵀ + PV (summed over heads)
                     + S * D * D                           # output proj
                     + 2 * S * D * F)                      # fc1 + fc2
            + 2 * NOUT * D)                                # two mapping heads (rows=1 after mean)
train_flop = 3 * 2 * fwd_macs                              # ×3 fwd/bwd, ×2 flop per MAC
FP32_PEAK = 82.6e12                                        # RTX 4090 FP32 (= TF32-tensor dense)
BF16_PEAK = 165.2e12                                       # RTX 4090 FP16/BF16-tensor dense
gf_ours = train_flop * B / (d / 1000) / 1e9
gf_torch = train_flop * B / (c / 1000) / 1e9
print(f"\n  TIC train-step FLOPs/window ≈ {train_flop/1e6:.1f} MFLOP  (fwd+bwd, matmul-dominated)")
print(f"  ours  : {gf_ours:7.0f} GFLOP/s = {gf_ours*1e9/FP32_PEAK*100:.2f}% of FP32 peak (82.6 TFLOP/s)")
print(f"  torch : {gf_torch:7.0f} GFLOP/s = {gf_torch*1e9/FP32_PEAK*100:.2f}% of FP32 peak")
print(f"  (TIC @ D={D} is far too small to saturate a 4090 — both are overhead/occupancy bound, "
      f"not FLOP bound; bf16 tensor peak is {BF16_PEAK/1e12:.0f} TFLOP/s)")
