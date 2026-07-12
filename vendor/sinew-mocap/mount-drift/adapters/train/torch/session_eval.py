"""Session-aggregated cross-study mount recovery — the deployment metric.

The mount R_BS is constant per session, so the runtime estimate is a robust SO(3) mean of the net's
per-window estimates over a session (slangtrain/refine.py RunningCalib), not the per-window error we
have been measuring (the worst case).  This trains the TIC on the per-window train parquet, then on a
SESSION test parquet (SINEW_SESSION=1 in b3d_to_caldata: one mount/drift per subject) runs the net per
window and aggregates per subject, reporting per-window vs session-aggregated mount error cross-study.

  pixi run -e gpu python slangtrain/torch/session_eval.py <train.parquet> <session_test.parquet> [STEPS]
"""
import os
import sys
import numpy as np
import pyarrow.parquet as pq
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # slangtrain/ for refine
sys.path.insert(0, os.environ.get("SINEW_TIC_REF", "/mnt/e/tmp/tic-calib"))
from my_model import TIC                        # noqa: E402  (the reference dir also has a train_tic.py
from refine import RunningCalib, geo_deg        # noqa: E402  — don't import it; inline the one helper)


def decode6d_np(v6):  # 6D -> R (Gram-Schmidt), columns [a, b, a×b]
    a = v6[:3] / (np.linalg.norm(v6[:3]) + 1e-9)
    b = v6[3:] - a * (a @ v6[3:])
    b /= (np.linalg.norm(b) + 1e-9)
    return np.stack([a, b, np.cross(a, b)], axis=1)

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
TRAIN = sys.argv[1] if len(sys.argv) > 1 else "/mnt/e/tmp/caldata_tr_ds.parquet"
TEST = sys.argv[2] if len(sys.argv) > 2 else "/mnt/e/tmp/caldata_te_session.parquet"
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
D, H, Fd, STACK, BATCH, LR = 64, 4, 128, 2, 256, 0.003


def load(path):
    t = pq.read_table(path)
    md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    s, n, o = int(md["S"]), int(md["NIN"]), int(md["NOUT"])
    x = t["x"].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, s, n).astype(np.float32)
    y = t["y"].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(len(t), -1).astype(np.float32)
    return (np.array(t["subject"].to_pylist()), x, y, s, n, o)


def grav(X, nsens):                              # match the trainer's default accel handling
    z = X.reshape(len(X), X.shape[1], nsens, 12).copy()
    a = z[..., :3]
    z[..., :3] = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-6)
    return z.reshape(X.shape)


_, trX, trY, S, NIN, NOUT = load(TRAIN)
NSENS = NOUT // 6
trX = grav(trX, NSENS)
trX_t = torch.from_numpy(trX).to(dev); trY_t = torch.from_numpy(trY).to(dev)

model = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=Fd).to(dev).float()
opt = torch.optim.Adam(model.parameters(), lr=LR)
mse = torch.nn.MSELoss()
print(f"training on {len(trX)} per-window train windows ({STEPS} steps)...")
model.train()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, len(trX_t), (BATCH,), device=dev)
    g, l = model(trX_t[idx])
    loss = mse(torch.cat([g, l], 1), trY_t[idx])
    opt.zero_grad(); loss.backward(); opt.step()

# ── session-aggregated eval on the fixed-mount-per-subject test ────────────────────────────────────
teSub, teX, teY, _, _, _ = load(TEST)
teX = grav(teX, NSENS)
model.eval()
with torch.no_grad():
    pred = model(torch.from_numpy(teX).to(dev))[1].cpu().numpy()    # local head = mount R_BS, (N, NOUT)

pw, agg, base = [], [], []
for su in sorted(set(teSub)):
    idx = np.where(teSub == su)[0]
    for s in range(NSENS):
        Rt = decode6d_np(teY[idx[0]][NOUT + s * 6:NOUT + s * 6 + 6])   # mount is constant in a session
        rc = RunningCalib()
        for i in idx:
            Rp = decode6d_np(pred[i][s * 6:s * 6 + 6])
            rc.update(Rp, 1.0)
            pw.append(geo_deg(Rp, Rt))
        agg.append(geo_deg(rc.R, Rt))
        base.append(geo_deg(np.eye(3), Rt))


def m(v):
    return float(np.mean(v)), float(np.median(v))


print(f"\nsession-aggregated cross-study mount ({len(set(teSub))} unseen-subject sessions, "
      f"{len(teX)} windows, {NSENS} sensors):")
print(f"  per-window  R_BS : {m(pw)[0]:5.1f}° mean / {m(pw)[1]:5.1f}° median")
print(f"  AGGREGATED  R_BS : {m(agg)[0]:5.1f}° mean / {m(agg)[1]:5.1f}° median   (identity {m(base)[0]:.1f}°)")
print(f"  aggregation gain : {m(pw)[0] - m(agg)[0]:+.1f}° mean   "
      + ("— session aggregation beats identity" if m(agg)[0] < m(base)[0] else "— still ≥ identity"))
