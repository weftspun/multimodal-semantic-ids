# SPDX-License-Identifier: MIT
# Apples-to-apples compute throughput: the tiled Vulkan GEMM kernel vs torch.matmul at
# matched M=N=K, compute only.  Vulkan init+readback is cancelled by subtracting a tiny
# 16^3 GEMM (same fixed overhead) from each large GEMM, leaving kernel compute time.
# Measured on the 4090: the naive one-thread-per-output GEMM peaks ~78 GFLOP/s at 2048^3
# (~0.1% of the card's ~80 TFLOP/s FP32) — roughly one MKL core, ~5x slower than 16-thread MKL,
# ~11x slower than a single CPU core at 512^3.  A shared-memory-tiled or cooperative-matrix GEMM
# closes the gap.
import subprocess, time, os
import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
ENTRY = "gemm_nn"  # C = A @ B


def vk(M, N, K):
    A = np.random.randn(M * K).astype(np.float32)
    B = np.random.randn(K * N).astype(np.float32)
    np.concatenate([A, B]).tofile(os.path.join(here, "inputs.bin"))
    t = time.perf_counter()
    subprocess.run([os.path.join(here, "gemm.exe"), os.path.join(here, "gemm.spv"), ENTRY,
                    str(M), str(N), str(K)], cwd=here, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - t


def vk_compute(n, reps=3):
    base = min(vk(16, 16, 16) for _ in range(reps))           # ~ init + tiny dispatch + readback
    big = min(vk(n, n, n) for _ in range(reps))
    return big - base, big


def torch_matmul(n, dev, reps=20):
    a = torch.randn(n, n, device=dev); b = torch.randn(n, n, device=dev)
    for _ in range(3):
        (a @ b)
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        c = a @ b
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


print("entry:", ENTRY, " (subtracting 16^3 GEMM to cancel Vulkan init+readback)\n")
print(f"{'N':>6} {'GFLOP':>8} {'vk_kernel':>11} {'vk_GFLOPs':>10} | "
      f"{'cpu1':>9} {'cpu16':>9} {'vk/cpu1':>8}")
for n in (256, 512, 1024, 2048):
    flop = 2 * n ** 3 / 1e9
    vkt, vkraw = vk_compute(n)
    torch.set_num_threads(1); c1 = torch_matmul(n, "cpu")
    torch.set_num_threads(os.cpu_count()); c16 = torch_matmul(n, "cpu")
    vkt = max(vkt, 1e-6)
    print(f"{n:>6} {flop:>8.2f} {vkt*1e3:>9.1f}ms {flop/vkt:>10.1f} | "
          f"{c1*1e3:>7.1f}ms {c16*1e3:>7.1f}ms {c1/vkt:>7.2f}x  (raw vk incl init {vkraw*1e3:.0f}ms)")

if torch.cuda.is_available():
    print("\nwith torch-CUDA:")
    for n in (1024, 2048, 4096):
        flop = 2 * n ** 3 / 1e9
        vkt, _ = vk_compute(n)
        cu = torch_matmul(n, "cuda")
        print(f"{n:>6}  vk {flop/max(vkt,1e-6):>7.0f} GFLOPs   cuda {flop/cu:>8.0f} GFLOPs   "
              f"vk is {cu/max(vkt,1e-6):.3f}x cuda")
else:
    print("\ntorch-CUDA: not installed (GPU reached only via Vulkan here)")
