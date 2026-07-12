"""Stage B: recover the 11-dim ANNY phenotype per AddBiomechanics subject by fitting SOMALayer's
ANNY skeleton segment lengths to the subject's seglen.  Runs in the export pixi env (py-soma-x).

  pixi run -e export python stage_b_fit.py [pheno_probe.json] [out pheno_fitted.json]

The 11 identity params exceed the 10 segment lengths, so the fit is underdetermined; a small ridge
pulls the unobserved directions to the 0.5 prior.  The recovered latent is loose (~120 mm seglen
RMSE on the salvaged subjects), so the equity weighting (pheno_equity) keys on measured anthropometry
(sex, BMI), not this latent — Stage B is the diagnostic that quantifies how well ANNY's shape space
reproduces real subjects.
"""
import sys, json
import numpy as np
import torch
from soma import SOMALayer

ANNY_SOMA = ["Hips", "LeftLeg", "RightLeg", "LeftShin", "RightShin", "LeftFoot", "RightFoot",
             "Chest", "Head", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand"]
SEG_CHILD = {0: 7, 1: 3, 2: 4, 3: 5, 4: 6, 7: 8, 9: 11, 10: 12, 11: 13, 12: 14}
SEG_NODES = sorted(SEG_CHILD)
RIDGE = 0.03
IN = sys.argv[1] if len(sys.argv) > 1 else "E:/tmp/pheno_probe.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "E:/tmp/pheno_fitted.json"

L = SOMALayer(identity_model_type="anny", device="cpu", mode="torch",
              enable_procedural_transforms=False).eval()
names = list(L.public_joint_names)
npose = len(names) - 1
anny_idx = [names.index(n) for n in ANNY_SOMA]
node_i = [anny_idx[n] for n in SEG_NODES]
child_i = [anny_idx[SEG_CHILD[n]] for n in SEG_NODES]


def seg_batch(idents):  # (N,11) -> (N,10) ANNY segment lengths, or None if batch not honoured
    idents = torch.as_tensor(idents, dtype=torch.float32)
    n = idents.shape[0]
    pose = torch.eye(3).repeat(n, npose, 1, 1)
    with torch.no_grad():
        L(pose, idents, scale_params={}, pose2rot=False, apply_correctives=False)
    bind = torch.linalg.inv(L.batched_skinning.inverse_bind_transform)
    if bind.ndim == 3:
        bind = bind.unsqueeze(0)
    pos = bind[..., :3, 3]
    if pos.shape[0] == 1 and n > 1:
        return None
    return torch.linalg.norm(pos[:, child_i] - pos[:, node_i], dim=-1).numpy()


BATCHED = seg_batch(np.full((2, 11), 0.5, np.float32)) is not None


def evalN(mat):
    if BATCHED:
        return seg_batch(mat)
    return np.stack([seg_batch(mat[i:i + 1])[0] for i in range(mat.shape[0])])


def fit(seg_target, iters=12):  # Levenberg-Marquardt, finite-diff jacobian + ridge
    x = np.full(11, 0.5, np.float64)
    lam, eps = 1e-3, 1e-3
    tgt = np.asarray(seg_target, np.float64)
    for _ in range(iters):
        base = evalN(x[None].astype(np.float32))[0].astype(np.float64)
        r = np.concatenate([base - tgt, RIDGE * (x - 0.5)])
        pert = np.repeat(x[None], 11, 0) + np.eye(11) * eps
        segp = evalN(pert.astype(np.float32)).astype(np.float64)
        J = np.zeros((21, 11))
        J[:10] = (segp - base).T / eps
        J[10:] = RIDGE * np.eye(11)
        dx = np.linalg.solve(J.T @ J + lam * np.eye(11), -J.T @ r)
        x = np.clip(x + dx, 0.0, 1.0)
    return x, float(np.sqrt(np.mean((evalN(x[None].astype(np.float32))[0] - tgt) ** 2)))


subs = json.load(open(IN))
print(f"batched={BATCHED}  fitting {len(subs)} subjects...")
rmse = []
for i, s in enumerate(subs):
    x, e = fit(s["seglen"])
    s["ident"] = [round(float(v), 4) for v in x]
    s["fit_rmse_m"] = round(e, 4)
    rmse.append(e)
    if i % 10 == 0:
        print(f"  {i}/{len(subs)}  rmse {e*1000:.1f} mm")
json.dump(subs, open(OUT, "w"))
print(f"fit RMSE: {np.mean(rmse)*1000:.1f} mm mean / {np.median(rmse)*1000:.1f} mm median -> {OUT}")
