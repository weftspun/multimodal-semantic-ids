"""T0 probe: JIT-compile probe_add.slang via slangtorch and run it on the GPU."""
import os
import torch
import slangtorch

here = os.path.dirname(os.path.abspath(__file__))
m = slangtorch.loadModule(os.path.join(here, "probe_add.slang"))
n = 1 << 16
a = torch.randn(n, device="cuda")
b = torch.randn(n, device="cuda")
r = torch.zeros(n, device="cuda")
block = 256
m.add(a=a, b=b, result=r).launchRaw(blockSize=(block, 1, 1), gridSize=((n + block - 1) // block, 1, 1))
torch.cuda.synchronize()
err = (r - (a + b)).abs().max().item()
print(f"slangtorch add max|err| = {err:.3e}  ->  {'OK' if err < 1e-6 else 'FAIL'}")
