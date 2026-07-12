"""Benchmark our slangtorch kernels vs torch-native on the GPU (grows with milestones).
  pixi run -e gpu python slangtrain/torch/bench_tk.py
"""
import torch
import tk_ops as tk

dev = "cuda"
torch.manual_seed(0)


def time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


print(f"GEMM = C·Bᵀ   naive vs tiled (shared-mem) vs torch cuBLAS   {torch.cuda.get_device_name(0)}")
print(f"  {'M×N×K':>16}  {'naive GF/s':>10}  {'tiled GF/s':>10}  {'cuBLAS GF/s':>11}  {'tiled/naive':>11}  {'tiled/cuBLAS':>12}")
for M, N, K in [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (480, 256, 256)]:
    X = torch.randn(M, K, device=dev)
    W = torch.randn(N, K, device=dev)
    flop = 2 * M * N * K
    # parity: tiled must match torch
    perr = (tk.gemm_nt_tiled(X, W, M, N, K) - X @ W.t()).abs().max().item()
    assert perr < 1e-2, f"tiled GEMM parity FAIL {M}×{N}×{K}: {perr}"
    o = flop / time_ms(lambda: tk.gemm_nt(X, W, M, N, K)) / 1e6
    ti = flop / time_ms(lambda: tk.gemm_nt_tiled(X, W, M, N, K)) / 1e6
    t = flop / time_ms(lambda: X @ W.t()) / 1e6
    print(f"  {M:4d}×{N:4d}×{K:4d}  {o:10.0f}  {ti:10.0f}  {t:11.0f}  {ti / o:10.1f}×  {ti / t * 100:10.1f}%")
