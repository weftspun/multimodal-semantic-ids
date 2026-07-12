# SPDX-License-Identifier: MIT
# Simulate-real with the ML calibrator (not the algebraic solver): train a pure-torch TIC
# (same architecture as tk_tic / netfwd) on calibration windows from a motion, and measure
# the recovered-mount error distribution.  Compares the active "dance" (multi-axis, rank-3
# excitation) against a gait-like hinge (1-axis) — the same contrast calib_rank.py shows for
# the exact solver, now through the net the deploy actually runs.  CUDA via torch's runtime
# (no nvcc / slangtorch needed).
#   pixi run -e gpu python ml_calib_eval.py
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); rng = np.random.default_rng(0)
S, NSENS, NOUT, D, H, Fd, STACK = 32, 15, 90, 64, 4, 128, 2
NIN = NSENS * 15
OFFSET_DEG, DRIFT_DEG = 30.0, 40.0
MAGW = np.array([0., -0.5, 0.866]); MAGW /= np.linalg.norm(MAGW); UP = np.array([0., 0., 1.])


def aa_to_R(v):                                   # batched Rodrigues: (...,3) -> (...,3,3)
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    k = v / (th + 1e-12)
    z = np.zeros(k.shape[:-1]); kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    K = np.stack([np.stack([z, -kz, ky], -1), np.stack([kz, z, -kx], -1),
                  np.stack([-ky, kx, z], -1)], -2)
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def motion(kind, phase):                          # S bone orientations
    Cs = np.empty((S, 3, 3)); amp = np.radians(45.0)
    for t in range(S):
        u = 2 * np.pi * t / S
        if kind == "dance":                       # rank-3: all three axes
            v = np.array([amp * np.sin(u + phase), amp * np.sin(2 * u + 1 + phase),
                          amp * np.sin(3 * u + 2 + phase)])
        else:                                     # gait/hinge: one axis only
            v = np.array([amp * np.sin(u + phase), 0.0, 0.0])
        Cs[t] = aa_to_R(v)
    return Cs


def r6d(R): return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)   # (...,3,3)->(...,6)


def gen(kind, K):
    Cs = np.stack([motion(kind, s * 0.7) for s in range(NSENS)])           # (NSENS,S,3,3)
    RBS = aa_to_R((rng.random((K, NSENS, 3)) - .5) * 2 * np.radians(OFFSET_DEG))
    yaw = (rng.random((K, NSENS)) - .5) * 2 * np.radians(DRIFT_DEG)
    tl = (rng.random((K, NSENS, 2)) - .5) * 2 * np.radians(10.)
    RDG = aa_to_R(np.stack([tl[..., 0], yaw, tl[..., 1]], -1))
    RIMU = np.einsum('ksij,stjl,kslm->kstim', RDG, Cs, RBS)                 # (K,NSENS,S,3,3)
    ag = np.einsum('ksij,j->ksi', RDG, UP); ag /= np.linalg.norm(ag, axis=-1, keepdims=True)
    mg = np.einsum('ksij,j->ksi', RDG, MAGW); mg /= np.linalg.norm(mg, axis=-1, keepdims=True)
    x = np.zeros((K, S, NSENS, 15), np.float32)
    x[..., :3] = ag[:, None, :, :]                                         # broadcast over t
    x[..., 3:6] = mg[:, None, :, :]
    x[..., 6:] = np.transpose(RIMU, (0, 2, 1, 3, 4)).reshape(K, S, NSENS, 9)
    y = np.concatenate([r6d(RDG).reshape(K, -1), r6d(RBS).reshape(K, -1)], -1).astype(np.float32)
    return torch.tensor(x.reshape(K, S, NIN)).to(dev), torch.tensor(y).to(dev)


class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.n1, s.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.q, s.k, s.v, s.o = (nn.Linear(D, D) for _ in range(4))
        s.f1, s.f2 = nn.Linear(D, Fd), nn.Linear(Fd, D)
    def forward(s, x):
        B, T, _ = x.shape; h = s.n1(x); dk = D // H
        q, k, v = (l(h).view(B, T, H, dk).transpose(1, 2) for l in (s.q, s.k, s.v))
        a = (q @ k.transpose(-1, -2) / math.sqrt(dk)).softmax(-1) @ v
        x = x + s.o(a.transpose(1, 2).reshape(B, T, D))
        return x + s.f2(F.relu(s.f1(s.n2(x))))


class TIC(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Linear(NIN, D)
        s.bb = nn.ModuleList([Enc() for _ in range(STACK)])
        s.hg, s.mg = Enc(), nn.Linear(D, NOUT); s.hl, s.ml = Enc(), nn.Linear(D, NOUT)
    def forward(s, x):
        h = s.emb(x) * math.sqrt(D)
        for l in s.bb: h = l(h)
        return s.mg(s.hg(h).mean(1)), s.ml(s.hl(h).mean(1))   # drift, mount


def decode6d(v):
    a = F.normalize(v[..., :3], dim=-1); b = v[..., 3:]
    b = F.normalize(b - (a * b).sum(-1, keepdim=True) * a, dim=-1)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], -1)


def geo(A, B):
    tr = (A.transpose(-1, -2) @ B).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))


def run(kind):
    Xtr, Ytr = gen(kind, 4000); Xte, Yte = gen(kind, 1000)
    m = TIC().to(dev); opt = torch.optim.Adam(m.parameters(), 3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
    for _ in range(3000):
        i = torch.randint(0, len(Xtr), (256,), device=dev)
        g, l = m(Xtr[i]); loss = F.mse_loss(torch.cat([g, l], 1), Ytr[i])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        _, l = m(Xte)
        e = geo(decode6d(l.view(-1, NSENS, 6)), decode6d(Yte[:, NOUT:].view(-1, NSENS, 6))).flatten().cpu().numpy()
    return e


print(f"device {dev}  (mount prior {OFFSET_DEG:.0f}°, drift {DRIFT_DEG:.0f}°, 1° meas via window dynamics)")
print(f"{'motion':>10} {'mean':>6} {'med':>6} {'p90':>6} {'P<5°':>6} {'P<2°':>6}")
for kind in ("gait", "dance"):
    e = run(kind)
    print(f"{kind:>10} {e.mean():5.1f}° {np.median(e):5.1f}° {np.percentile(e,90):5.1f}° "
          f"{100*np.mean(e<5):5.0f}% {100*np.mean(e<2):5.0f}%")
