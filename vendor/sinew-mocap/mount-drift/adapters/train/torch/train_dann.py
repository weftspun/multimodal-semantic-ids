"""Domain-adversarial calibration training (DANN) — remove the domain-shift carriers from the learned
features.  A gradient-reversal subject classifier on the TIC backbone penalises the encoder for
encoding subject/study identity, forcing study-invariant features (the inverse of domain_probe, which
*measures* that identity).  Reports cross-study held-out mount R_BS for adversarial λ vs λ=0 (plain).

  pixi run -e gpu python slangtrain/torch/train_dann.py <train.pq> --test <test.pq> [LAMBDA STEPS]
"""
import os
import sys
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.environ.get("SINEW_TIC_REF", "/mnt/e/tmp/tic-calib"))
from my_model import TIC  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
argv = sys.argv[1:]
TEST = argv[argv.index("--test") + 1] if "--test" in argv else None
if "--test" in argv:
    i = argv.index("--test"); del argv[i:i + 2]
TRAIN = argv[0]
LAMBDA = float(argv[1]) if len(argv) > 1 else 0.3
STEPS = int(argv[2]) if len(argv) > 2 else 3000
D, H, Fd, STACK, BATCH, LR = 64, 4, 128, 2, 256, 0.003


def load(p):
    t = pq.read_table(p); md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    s, n, o = int(md["S"]), int(md["NIN"]), int(md["NOUT"])
    x = t["x"].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, s, n).astype(np.float32)
    y = t["y"].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(len(t), -1).astype(np.float32)
    return np.array(t["subject"].to_pylist()), x, y, s, n, o


def grav(X, ns):
    z = X.reshape(len(X), X.shape[1], ns, 12).copy()
    a = z[..., :3]; z[..., :3] = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-6)
    return z.reshape(X.shape)


def decode6d(v):
    a = v[:3] / (np.linalg.norm(v[:3]) + 1e-9); b = v[3:] - a * (a @ v[3:]); b /= (np.linalg.norm(b) + 1e-9)
    return np.stack([a, b, np.cross(a, b)], axis=1)


def geo(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


class GRL(torch.autograd.Function):  # gradient reversal: identity forward, −λ·grad backward
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


sub, X, Y, S, NIN, NOUT = load(TRAIN); NSENS = NOUT // 6
X = grav(X, NSENS)
teSub, teX, teY, _, _, _ = load(TEST); teX = grav(teX, NSENS)
uniq, dlab = np.unique(sub, return_inverse=True); NDOM = len(uniq)
Xt = torch.from_numpy(X).to(dev); Yt = torch.from_numpy(Y).to(dev); Dt = torch.from_numpy(dlab).to(dev)
teXt = torch.from_numpy(teX).to(dev)
print(f"train {len(X)} windows / {NDOM} domains(subjects), test {len(teX)} / {len(set(teSub))} subjects, "
      f"LAMBDA={LAMBDA} STEPS={STEPS}")

model = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=Fd).to(dev)
dom_head = torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.ReLU(), torch.nn.Linear(256, NDOM)).to(dev)
opt = torch.optim.Adam(list(model.parameters()) + list(dom_head.parameters()), lr=LR)
mse = torch.nn.MSELoss(); ce = torch.nn.CrossEntropyLoss()


def backbone(xb):  # replicate TIC.forward but tap the shared encoder features for the domain head
    x = model.input_embedding_layer(xb)
    for enc in model.encoder_banckbone:
        x = enc(x, None)
    g = model.TPM_global(x); l = model.TPM_local(x)
    feat = x.mean(dim=1)                              # (B, D) pooled backbone features
    return torch.cat([g, l], 1), feat


def held_out_RBS():
    model.eval()
    with torch.no_grad():
        pred = backbone(teXt)[0].cpu().numpy()
    model.train()
    e, b = [], []
    for ti in range(len(teY)):
        for s in range(NSENS):
            Rp = decode6d(pred[ti][NOUT + s * 6:NOUT + s * 6 + 6]); Rt = decode6d(teY[ti][NOUT + s * 6:NOUT + s * 6 + 6])
            e.append(geo(Rp, Rt)); b.append(geo(np.eye(3), Rt))
    return np.mean(e), np.median(e), np.mean(b)


model.train()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, len(Xt), (BATCH,), device=dev)
    out, feat = backbone(Xt[idx])
    loss_cal = mse(out, Yt[idx])
    lam = LAMBDA * (2 / (1 + np.exp(-10 * step / STEPS)) - 1)   # ramp λ in (DANN schedule)
    loss_dom = ce(dom_head(GRL.apply(feat, lam)), Dt[idx])
    opt.zero_grad(); (loss_cal + loss_dom).backward(); opt.step()
    if step % 500 == 0 or step == STEPS:
        with torch.no_grad():
            dacc = (dom_head(feat).argmax(1) == Dt[idx]).float().mean().item()
        m, md, base = held_out_RBS()
        print(f"step {step:5d}  cal {loss_cal.item():.4f}  dom-acc {dacc:.2f}  "
              f"held-out R_BS {m:5.1f}°/{md:5.1f}° (id {base:.1f}°)")

m, md, base = held_out_RBS()
print(f"\nLAMBDA={LAMBDA}: cross-study held-out R_BS {m:.1f}° mean / {md:.1f}° median (identity {base:.1f}°)")
print("read: vs plain (LAMBDA=0) — lower = adversarial invariance removed carriers without killing signal.")
