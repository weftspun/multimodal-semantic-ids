"""Fill the GPU with many models at once: train K TIC configs/folds in parallel via torch.func.vmap.

One small TIC underfills a 4090, so the training step is overhead/occupancy-bound.  Vmapping K
models (different inits = different hyperparam configs or CV folds) turns K tiny steps into one big
batched step that uses the idle SMs.  This measures total throughput (model·windows/s) vs K to find
where the GPU saturates — pure torch, no custom kernels.
  pixi run -e gpu python slangtrain/torch/bench_parallel.py
"""
import sys
import torch
from torch.func import stack_module_state, functional_call, vmap, grad

sys.path.insert(0, "E:/tmp/tic-calib")
from my_model import TIC  # noqa: E402

torch.manual_seed(0)
torch.set_float32_matmul_precision("high")  # let cuBLAS use TF32 tensor cores (free)
dev = "cuda"
S, NIN, D, H, F, STACK, NOUT = 32, 180, 64, 4, 128, 2, 90
B = 64                       # windows per model per step
rg = torch.randn(NOUT, device=dev)
rl = torch.randn(NOUT, device=dev)
base = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=F).to(dev)


def time_ms(fn, iters=20, warmup=8):
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


def floss(p, b, x):
    g, l = functional_call(base, (p, b), (x,))
    return ((g * rg).sum() + (l * rl).sum())


gradf = vmap(grad(floss), in_dims=(0, 0, 0))


def make_parallel_step(K):
    models = [TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=F).to(dev)
              for _ in range(K)]
    params, buffers = stack_module_state(models)
    m = {k: torch.zeros_like(v) for k, v in params.items()}
    v = {k: torch.zeros_like(v) for k, v in params.items()}
    X = torch.randn(K, B, S, NIN, device=dev)
    t = [0]

    def step():
        t[0] += 1
        grads = gradf(params, buffers, X)              # K models differentiated at once
        with torch.no_grad():                          # fused Adam over the stacked params
            bc1 = 1 - 0.9 ** t[0]
            bc2 = 1 - 0.999 ** t[0]
            for k in params:
                g = grads[k]
                m[k].mul_(0.9).add_(g, alpha=0.1)
                v[k].mul_(0.999).addcmul_(g, g, value=0.001)
                params[k].addcdiv_(m[k] / bc1, (v[k] / bc2).sqrt_().add_(1e-8), value=-1e-3)
    return step


print(f"Parallel TIC training (vmap, TF32 on)   {torch.cuda.get_device_name(0)}   "
      f"B={B} windows/model  dims D={D} F={F} STACK={STACK}")
print(f"  {'K models':>9}  {'step ms':>8}  {'model·win/s':>12}  {'vs K=1 scaling':>15}")
base_rate = None
for K in [1, 2, 4, 8, 16, 32, 64, 128]:
    try:
        ms = time_ms(make_parallel_step(K))
    except RuntimeError as ex:
        print(f"  {K:9d}  OOM/err: {str(ex)[:50]}")
        break
    rate = K * B / ms * 1000
    if base_rate is None:
        base_rate = rate
    print(f"  {K:9d}  {ms:8.3f}  {rate:12.0f}  {rate / base_rate:13.1f}×")
print("\n  throughput keeps rising with K until the GPU saturates; that knee is your free capacity.")
