"""Parity gate for the slangtorch TIC kernels vs torch-native ops (and, later, the full TIC).

Grows with the milestones; each check asserts max|err| under tolerance on the GPU.
  pixi run -e gpu python slangtrain/torch/check_tk.py
"""
import torch
import tk_ops as tk

torch.manual_seed(0)
dev = "cuda"
fails = []


def check(name, got, ref, tol):
    err = (got - ref).abs().max().item()
    ok = err < tol
    print(f"  {'OK ' if ok else 'FAIL'} {name:20s} max|err|={err:.2e}  (tol {tol:.0e})")
    if not ok:
        fails.append(name)


print("GEMM (brick 1) vs torch:")
M, N, K = 96, 64, 80
X = torch.randn(M, K, device=dev)
W = torch.randn(N, K, device=dev)
B = torch.randn(K, N, device=dev)
check("gemm_nt  Y=X·Wᵀ", tk.gemm_nt(X, W, M, N, K), X @ W.t(), 1e-3)
check("gemm_nn  C=X·B", tk.gemm_nn(X, B, M, N, K), X @ B, 1e-3)
A_km = torch.randn(K, M, device=dev)
check("gemm_tn  C=Aᵀ·B", tk.gemm_tn(A_km, B, M, N, K), A_km.t() @ B, 1e-3)
check("gemm_nt_tiled", tk.gemm_nt_tiled(X, W, M, N, K), X @ W.t(), 1e-3)

print("\nForward bricks (lin/ln/attn/ew) vs torch:")
import torch.nn.functional as F
# linear: Y = X·Wᵀ + b
bvec = torch.randn(N, device=dev)
check("linear  X·Wᵀ+b", tk.linear(X, W, bvec, M, N, K), X @ W.t() + bvec, 1e-3)
check("col_sum (bias grad)", tk.col_sum(X), X.sum(0), 1e-3)
# layernorm: per-row, eps 1e-5, affine g,b
R, Dd = 40, 48
xln = torch.randn(R, Dd, device=dev)
g = torch.randn(Dd, device=dev)
bl = torch.randn(Dd, device=dev)
y_ours, _, _ = tk.ln_fwd(xln, g, bl)
check("ln_fwd", y_ours, F.layer_norm(xln, (Dd,), g, bl, eps=1e-5), 1e-3)
# attention: multi-head SDPA, H heads
Sn, Dn, Hn = 16, 32, 4
Q = torch.randn(Sn, Dn, device=dev)
Kk = torch.randn(Sn, Dn, device=dev)
Vv = torch.randn(Sn, Dn, device=dev)
out_ours, _ = tk.attn_fwd(Q, Kk, Vv, Hn)
# torch ref: reshape to (H, S, DK), scaled dot-product per head
DKn = Dn // Hn
qr = Q.view(Sn, Hn, DKn).permute(1, 0, 2)
kr = Kk.view(Sn, Hn, DKn).permute(1, 0, 2)
vr = Vv.view(Sn, Hn, DKn).permute(1, 0, 2)
ref_attn = F.scaled_dot_product_attention(qr, kr, vr).permute(1, 0, 2).reshape(Sn, Dn)
check("attn_fwd (MHSA)", out_ours, ref_attn, 1e-3)
# elementwise
a = torch.randn(100, device=dev)
b2 = torch.randn(100, device=dev)
check("add (residual)", tk.add(a, b2), a + b2, 1e-4)
check("relu_fwd", tk.relu_fwd(a), F.relu(a), 1e-4)

print("\nBackward (autograd.Function grads) vs torch:")


def dup(t):  # two independent leaves with the same data
    return t.detach().clone().requires_grad_(True), t.detach().clone().requires_grad_(True)


# Linear
Xo, Xt = dup(torch.randn(M, K, device=dev))
Wo, Wt = dup(torch.randn(N, K, device=dev))
bo, bt = dup(torch.randn(N, device=dev))
go = torch.randn(M, N, device=dev)
(tk.Linear.apply(Xo, Wo, bo) * go).sum().backward()
((Xt @ Wt.t() + bt) * go).sum().backward()
check("Linear dX", Xo.grad, Xt.grad, 2e-3)
check("Linear dW", Wo.grad, Wt.grad, 2e-3)
check("Linear db", bo.grad, bt.grad, 2e-3)

# LayerNorm
xo, xt = dup(torch.randn(R, Dd, device=dev))
go2, gt2 = dup(g)
bo2, bt2 = dup(bl)
gln = torch.randn(R, Dd, device=dev)
(tk.LayerNorm.apply(xo, go2, bo2) * gln).sum().backward()
(F.layer_norm(xt, (Dd,), gt2, bt2, eps=1e-5) * gln).sum().backward()
check("LayerNorm dx", xo.grad, xt.grad, 2e-3)
check("LayerNorm dg", go2.grad, gt2.grad, 2e-3)
check("LayerNorm db", bo2.grad, bt2.grad, 2e-3)

# Attention
Qo, Qt = dup(Q)
Ko, Kt = dup(Kk)
Vo, Vt = dup(Vv)
ga = torch.randn(Sn, Dn, device=dev)
(tk.Attention.apply(Qo, Ko, Vo, Hn) * ga).sum().backward()
qr2 = Qt.view(Sn, Hn, DKn).permute(1, 0, 2)
kr2 = Kt.view(Sn, Hn, DKn).permute(1, 0, 2)
vr2 = Vt.view(Sn, Hn, DKn).permute(1, 0, 2)
(F.scaled_dot_product_attention(qr2, kr2, vr2).permute(1, 0, 2).reshape(Sn, Dn) * ga).sum().backward()
check("Attention dQ", Qo.grad, Qt.grad, 2e-3)
check("Attention dK", Ko.grad, Kt.grad, 2e-3)
check("Attention dV", Vo.grad, Vt.grad, 2e-3)

# ReLU
ro, rt = dup(torch.randn(200, device=dev))
gr = torch.randn(200, device=dev)
(tk.ReLU.apply(ro) * gr).sum().backward()
(F.relu(rt) * gr).sum().backward()
check("ReLU dx", ro.grad, rt.grad, 1e-4)

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
raise SystemExit(1 if fails else 0)
