"""PyTorch ops backed by our Lean→Slang kernels, compiled for CUDA via slangtorch.

Loads slangtrain/torch/kernels.slang once and exposes thin launchers that mirror the Vulkan host
dispatch (slangtrain/*_main.cpp): 16×16 blocks, grid x=N (col), y=M (row).  Tensors are passed
reshaped to 1-D contiguous to match the kernels' flat indexing.  Higher-level autograd.Function
ops (added in T3) compose these.
"""
import os
import torch
import slangtorch

_here = os.path.dirname(os.path.abspath(__file__))
_m = slangtorch.loadModule(os.path.join(_here, "kernels.slang"))

_BLK = 16


def _grid(N, M):  # x covers N (columns), y covers M (rows)
    return ((N + _BLK - 1) // _BLK, (M + _BLK - 1) // _BLK, 1)


def _f(t):  # 1-D contiguous float32 cuda view for flat kernel indexing
    return t.reshape(-1).contiguous()


def gemm_nn(A, B, M, N, K):  # C = A·B,  A:M×K  B:K×N
    C = torch.empty(M * N, device=A.device, dtype=torch.float32)
    _m.gemm_nn(A=_f(A), B=_f(B), C=C, M=M, N=N, K=K).launchRaw(
        blockSize=(_BLK, _BLK, 1), gridSize=_grid(N, M))
    return C.view(M, N)


def gemm_nt(A, B, M, N, K):  # C = A·Bᵀ,  A:M×K  B:N×K
    C = torch.empty(M * N, device=A.device, dtype=torch.float32)
    _m.gemm_nt(A=_f(A), B=_f(B), C=C, M=M, N=N, K=K).launchRaw(
        blockSize=(_BLK, _BLK, 1), gridSize=_grid(N, M))
    return C.view(M, N)


def gemm_tn(A, B, M, N, K):  # C = Aᵀ·B,  A:K×M  B:K×N
    C = torch.empty(M * N, device=A.device, dtype=torch.float32)
    _m.gemm_tn(A=_f(A), B=_f(B), C=C, M=M, N=N, K=K).launchRaw(
        blockSize=(_BLK, _BLK, 1), gridSize=_grid(N, M))
    return C.view(M, N)


def gemm_nt_tiled(A, B, M, N, K):  # C = A·Bᵀ, shared-memory tiled (same result as gemm_nt)
    C = torch.empty(M * N, device=A.device, dtype=torch.float32)
    _m.gemm_nt_tiled(A=_f(A), B=_f(B), C=C, M=M, N=N, K=K).launchRaw(
        blockSize=(_BLK, _BLK, 1), gridSize=_grid(N, M))
    return C.view(M, N)


def _grid1(n, blk=64):
    return ((n + blk - 1) // blk, 1, 1)


def bias_add_(Y, vec):  # in-place Y[r,c] += vec[c]
    rows, cols = Y.shape
    _m.bias_add(Y=_f(Y), vec=_f(vec), rows=rows, cols=cols).launchRaw(
        blockSize=(_BLK, _BLK, 1), gridSize=_grid(cols, rows))
    return Y


def col_sum(Y):  # vec[c] = Σ_r Y[r,c]
    rows, cols = Y.shape
    vec = torch.empty(cols, device=Y.device, dtype=torch.float32)
    _m.col_sum(Y=_f(Y), vec=vec, rows=rows, cols=cols).launchRaw(
        blockSize=(64, 1, 1), gridSize=_grid1(cols))
    return vec


def linear(X, W, b, M, N, K):  # Y = X·Wᵀ + b ;  X:M×K  W:N×K  b:N
    Y = gemm_nt(X, W, M, N, K)
    return bias_add_(Y, b)


def ln_fwd(x, g, b):  # y = layernorm(x)·g + b ; x:R×D  g,b:D  -> y, (mean, inv) cache
    R, D = x.shape
    y = torch.empty(R * D, device=x.device, dtype=torch.float32)
    mean = torch.empty(R, device=x.device, dtype=torch.float32)
    inv = torch.empty(R, device=x.device, dtype=torch.float32)
    _m.ln_fwd(x=_f(x), g=_f(g), b=_f(b), y=y, mean=mean, inv=inv, R=R, D=D).launchRaw(
        blockSize=(64, 1, 1), gridSize=_grid1(R))
    return y.view(R, D), mean, inv


def attn_fwd(Q, K, V, H):  # multi-head SDPA ; Q,K,V:S×D -> Out:S×D, P:H*S*S cache
    S, D = Q.shape
    DK = D // H
    Out = torch.empty(S * D, device=Q.device, dtype=torch.float32)
    P = torch.empty(H * S * S, device=Q.device, dtype=torch.float32)
    _m.attn_fwd(Q=_f(Q), K=_f(K), V=_f(V), Out=Out, P=P, S=S, D=D, H=H, DK=DK).launchRaw(
        blockSize=(_BLK, _BLK, 1), gridSize=_grid(S, H))
    return Out.view(S, D), P


def add(A, B):  # residual C = A + B
    n = A.numel()
    C = torch.empty(n, device=A.device, dtype=torch.float32)
    _m.add(A=_f(A), B=_f(B), C=C, n=n).launchRaw(blockSize=(64, 1, 1), gridSize=_grid1(n))
    return C.view_as(A)


def relu_fwd(A):
    n = A.numel()
    C = torch.empty(n, device=A.device, dtype=torch.float32)
    _m.relu_fwd(A=_f(A), C=C, n=n).launchRaw(blockSize=(64, 1, 1), gridSize=_grid1(n))
    return C.view_as(A)


# ── backward launchers ───────────────────────────────────────────────────────
def ln_bwd(x, g, dy, mean, inv):  # -> dx, dg, db
    R, D = x.shape
    dx = torch.empty(R * D, device=x.device, dtype=torch.float32)
    _m.ln_bwd_dx(x=_f(x), g=_f(g), dy=_f(dy), dx=dx, mean=mean, inv=inv, R=R, D=D).launchRaw(
        blockSize=(64, 1, 1), gridSize=_grid1(R))
    dg = torch.empty(D, device=x.device, dtype=torch.float32)
    db = torch.empty(D, device=x.device, dtype=torch.float32)
    _m.ln_bwd_dgdb(x=_f(x), dy=_f(dy), dg=dg, db=db, mean=mean, inv=inv, R=R, D=D).launchRaw(
        blockSize=(64, 1, 1), gridSize=_grid1(D))
    return dx.view(R, D), dg, db


def relu_bwd(x, dy):
    n = x.numel()
    dx = torch.empty(n, device=x.device, dtype=torch.float32)
    _m.relu_bwd(A=_f(x), B=_f(dy), C=dx, n=n).launchRaw(blockSize=(64, 1, 1), gridSize=_grid1(n))
    return dx.view_as(x)


def attn_bwd(Q, K, V, P, dOut, H):  # -> dQ, dK, dV
    S, D = Q.shape
    DK = D // H
    dsim = torch.empty(H * S * S, device=Q.device, dtype=torch.float32)
    _m.attn_bwd_dsim(V=_f(V), P=P, dOut=_f(dOut), dsim=dsim, S=S, D=D, H=H, DK=DK).launchRaw(
        blockSize=(_BLK, _BLK, 1), gridSize=_grid(S, H))
    dQ = torch.empty(S * D, device=Q.device, dtype=torch.float32)
    dK = torch.empty(S * D, device=Q.device, dtype=torch.float32)
    dV = torch.empty(S * D, device=Q.device, dtype=torch.float32)
    _m.attn_bwd_dq(K=_f(K), dsim=dsim, dQ=dQ, S=S, D=D, H=H, DK=DK).launchRaw(
        blockSize=(_BLK, _BLK, 1), gridSize=_grid(S, H))
    _m.attn_bwd_dk(Q=_f(Q), dsim=dsim, dK=dK, S=S, D=D, H=H, DK=DK).launchRaw(
        blockSize=(_BLK, _BLK, 1), gridSize=_grid(S, H))
    _m.attn_bwd_dv(P=P, dOut=_f(dOut), dV=dV, S=S, D=D, H=H, DK=DK).launchRaw(
        blockSize=(_BLK, _BLK, 1), gridSize=_grid(S, H))
    return dQ.view(S, D), dK.view(S, D), dV.view(S, D)


# ── autograd.Function ops (compose fwd/bwd kernels into differentiable torch ops) ─
class Linear(torch.autograd.Function):  # Y = X·Wᵀ + b ; X:M×K  W:N×K  b:N
    @staticmethod
    def forward(ctx, X, W, b):
        M, K = X.shape
        N = W.shape[0]
        ctx.save_for_backward(X, W)
        return linear(X, W, b, M, N, K)

    @staticmethod
    def backward(ctx, dY):
        X, W = ctx.saved_tensors
        M, K = X.shape
        N = W.shape[0]
        dY = dY.contiguous()
        dX = gemm_nn(dY, W, M, K, N)        # dX = dY·W      (M×N)(N×K)
        dW = gemm_tn(dY, X, N, K, M)        # dW = dYᵀ·X     (N×M)(M×K)
        db = col_sum(dY)
        return dX, dW, db


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, b):
        y, mean, inv = ln_fwd(x, g, b)
        ctx.save_for_backward(x, g, mean, inv)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, g, mean, inv = ctx.saved_tensors
        dx, dg, db = ln_bwd(x, g, dy.contiguous(), mean, inv)
        return dx, dg, db


class Attention(torch.autograd.Function):  # multi-head SDPA over Q,K,V:S×D
    @staticmethod
    def forward(ctx, Q, K, V, H):
        Out, P = attn_fwd(Q, K, V, H)
        ctx.save_for_backward(Q, K, V, P)
        ctx.H = H
        return Out

    @staticmethod
    def backward(ctx, dOut):
        Q, K, V, P = ctx.saved_tensors
        dQ, dK, dV = attn_bwd(Q, K, V, P, dOut.contiguous(), ctx.H)
        return dQ, dK, dV, None


class ReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return relu_fwd(x)

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        return relu_bwd(x, dy.contiguous())


def _grid3(N, M, Z):  # x covers N, y covers M, z = batch
    return ((N + _BLK - 1) // _BLK, (M + _BLK - 1) // _BLK, Z)


def attn_fwd_b(Q, K, V, H, B, S):  # Q,K,V:(B·S, D) -> Out:(B·S, D), P:(B·H·S·S)
    D = Q.shape[1]
    DK = D // H
    Out = torch.empty(B * S * D, device=Q.device, dtype=torch.float32)
    P = torch.empty(B * H * S * S, device=Q.device, dtype=torch.float32)
    _m.attn_fwd_b(Q=_f(Q), K=_f(K), V=_f(V), Out=Out, P=P, B=B, S=S, D=D, H=H, DK=DK).launchRaw(
        blockSize=(_BLK, _BLK, 1), gridSize=_grid3(S, H, B))
    return Out.view(B * S, D), P


def attn_bwd_b(Q, K, V, P, dOut, H, B, S):  # -> dQ, dK, dV  (each (B·S, D))
    D = Q.shape[1]
    DK = D // H
    dsim = torch.empty(B * H * S * S, device=Q.device, dtype=torch.float32)
    _m.attn_bwd_dsim_b(V=_f(V), P=P, dOut=_f(dOut), dsim=dsim, B=B, S=S, D=D, H=H, DK=DK).launchRaw(
        blockSize=(_BLK, _BLK, 1), gridSize=_grid3(S, H, B))
    dQ = torch.empty(B * S * D, device=Q.device, dtype=torch.float32)
    dK = torch.empty(B * S * D, device=Q.device, dtype=torch.float32)
    dV = torch.empty(B * S * D, device=Q.device, dtype=torch.float32)
    _m.attn_bwd_dq_b(K=_f(K), dsim=dsim, dQ=dQ, B=B, S=S, D=D, H=H, DK=DK).launchRaw(
        blockSize=(_BLK, _BLK, 1), gridSize=_grid3(S, H, B))
    _m.attn_bwd_dk_b(Q=_f(Q), dsim=dsim, dK=dK, B=B, S=S, D=D, H=H, DK=DK).launchRaw(
        blockSize=(_BLK, _BLK, 1), gridSize=_grid3(S, H, B))
    _m.attn_bwd_dv_b(P=P, dOut=_f(dOut), dV=dV, B=B, S=S, D=D, H=H, DK=DK).launchRaw(
        blockSize=(_BLK, _BLK, 1), gridSize=_grid3(S, H, B))
    return dQ.view(B * S, D), dK.view(B * S, D), dV.view(B * S, D)


class AttentionB(torch.autograd.Function):  # batched MHSA over B windows, Q,K,V:(B·S, D)
    @staticmethod
    def forward(ctx, Q, K, V, H, B, S):
        Out, P = attn_fwd_b(Q, K, V, H, B, S)
        ctx.save_for_backward(Q, K, V, P)
        ctx.H, ctx.B, ctx.S = H, B, S
        return Out

    @staticmethod
    def backward(ctx, dOut):
        Q, K, V, P = ctx.saved_tensors
        dQ, dK, dV = attn_bwd_b(Q, K, V, P, dOut.contiguous(), ctx.H, ctx.B, ctx.S)
        return dQ, dK, dV, None, None, None


def adam_step(W, G, M, V, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """In-place Adam update of W (and moments M,V) from grad G. Matches torch.optim.Adam."""
    n = W.numel()
    _m.adam_n(W=_f(W), G=_f(G), M=_f(M), V=_f(V), t=float(t), lr=lr, b1=b1, b2=b2,
              eps=eps, n=n).launchRaw(blockSize=(64, 1, 1), gridSize=_grid1(n))
