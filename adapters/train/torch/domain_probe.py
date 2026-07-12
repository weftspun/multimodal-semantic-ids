"""Find the domain-shift carriers: train a DOMAIN classifier on the calibration windows under several
feature transforms.  The calibration target (R_BS, R_DG) is injected at random, independent of
subject/study, so any feature a classifier can use to predict which SUBJECT a window came from is
domain shift with no calibration signal.  Whatever transform drops the accuracy toward chance is the
carrier-removal to apply.

TRAINING DATA ONLY — the held-out test studies are never read here, so the carriers (and the fix they
motivate) are found without leaking the test split; the cross-study held-out stays an honest metric.

  pixi run -e gpu python slangtrain/torch/domain_probe.py [train_parquet ...]   (default: tr_ds)
Reads the `subject`/`study` columns (added to b3d_to_caldata).  Domain = subject (finest, many
classes); `study` is reported too.  Transforms:
  full      raw X                          accel|rot raw, all info
  grav      unit-normalize accel + rot     the trainer's default
  rot       rotation only (accel zeroed)   is the carrier in rotation?
  accel     grav accel only (rot zeroed)   residual accel domain (p̈ direction, after grav)
  rot_rel   per-bone R_t·R_0ᵀ              motion only — absolute orientation removed
  rot_abs   per-bone R_0 (first frame)     absolute facing/rest-pose only — motion removed
  ctrl_Y    the injected R_BS/R_DG labels  control: must be ~chance (injected independent of study)
"""
import sys
import numpy as np
import pyarrow.parquet as pq
import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)
PARQUETS = sys.argv[1:] or ["/mnt/e/tmp/caldata_tr_ds.parquet"]   # TRAINING ONLY — never the test split

Xs, Ys, subjects, studies = [], [], [], []
S = NIN = NOUT = None
for p in PARQUETS:
    t = pq.read_table(p)
    md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    S, NIN, NOUT = int(md["S"]), int(md["NIN"]), int(md["NOUT"])
    Xs.append(t["x"].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, S, NIN).astype(np.float32))
    Ys.append(t["y"].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(len(t), -1).astype(np.float32))
    subjects += t["subject"].to_pylist()
    studies += t["study"].to_pylist()
X = np.concatenate(Xs)
Y = np.concatenate(Ys)
NSENS = NOUT // 6
uniq, lab = np.unique(np.array(subjects), return_inverse=True)   # domain = subject (finest)
K = len(uniq)
chance = max(np.bincount(lab)) / len(lab)   # majority-class baseline
print(f"{len(X)} train windows, domain=subject ({K} subjects), studies={sorted(set(studies))}, "
      f"S={S} NIN={NIN}  majority baseline {chance:.2f}")


def to_R(rot9):                       # (...,9) -> (...,3,3)
    return rot9.reshape(rot9.shape[:-1] + (3, 3))


def feat(kind):
    A = X.reshape(len(X), S, NSENS, 12)
    acc, rot = A[..., :3], A[..., 3:]                       # (N,S,NSENS,3) , (N,S,NSENS,9)
    gacc = acc / (np.linalg.norm(acc, axis=-1, keepdims=True) + 1e-6)
    if kind == "full":
        return X.reshape(len(X), -1)
    if kind == "grav":
        return np.concatenate([gacc, rot], -1).reshape(len(X), -1)
    if kind == "rot":
        return rot.reshape(len(X), -1)
    if kind == "accel":
        return gacc.reshape(len(X), -1)
    if kind == "rot_abs":
        return rot[:, 0].reshape(len(X), -1)                # first-frame orientation per bone
    if kind == "rot_rel":
        R = to_R(rot)                                       # (N,S,NSENS,3,3)
        R0 = R[:, :1]                                       # (N,1,NSENS,3,3)
        rel = np.einsum("ntsij,ntskj->ntsik", R, R0)        # R_t · R_0ᵀ
        return rel.reshape(len(X), -1)
    if kind == "ctrl_Y":
        return Y.copy()
    raise SystemExit(f"unknown feature {kind}")


def probe(kind, steps=1500):
    F = feat(kind).astype(np.float32)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)                 # standardize
    n = len(F)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    ntr = int(n * 0.8)
    tr, va = perm[:ntr], perm[ntr:]
    Ft = torch.tensor(F[tr], device=dev); yt = torch.tensor(lab[tr], device=dev)
    Fv = torch.tensor(F[va], device=dev); yv = torch.tensor(lab[va], device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(F.shape[1], 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, K)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    best = 0.0
    for step in range(steps):
        idx = torch.randint(0, len(Ft), (512,), device=dev)
        opt.zero_grad(); lossf(net(Ft[idx]), yt[idx]).backward(); opt.step()
        if step % 150 == 0 or step == steps - 1:
            net.eval()
            with torch.no_grad():
                acc = (net(Fv).argmax(1) == yv).float().mean().item()
            net.train()
            best = max(best, acc)
    return best, F.shape[1]


print(f"\nsubject(domain)-classification accuracy (majority baseline = {chance:.2f}):")
for kind in ["ctrl_Y", "accel", "rot_abs", "rot_rel", "rot", "grav", "full"]:
    acc, dim = probe(kind)
    bar = "█" * int(acc * 40)
    print(f"  {kind:8s} (d={dim:5d})  acc {acc:.2f}  {bar}")
print("\nread: acc ≫ baseline ⇒ that feature carries subject/domain identity (a domain-shift carrier). "
      "ctrl_Y must be ~baseline (target is injected independent of subject). rot_abs high ⇒ "
      "facing/rest-pose is a carrier; rot_rel high ⇒ motion repertoire; accel high after grav ⇒ "
      "residual p̈-direction.")
