"""Inverse-ANNY fit: recover a real subject's ANNY phenotype from its OpenSim fitted skeleton.

The B3D processing pass gives 20 fitted joint world positions (`jointCenters`, a GET — no segfault-prone
setPositions FK).  Their pose-invariant joint-pair lengths are the target; ANNY is a differentiable
torch body (SOMALayer), so we gradient-descend its 11-dim identity until ANNY's joint-pair lengths
match.  The recovered identity is the subject's ANNY phenotype — the basis the synthetic generator
perturbs to cover the equity bands (esp. obese) the real corpus lacks, with a *modeled* soft-tissue
mount perturbation on top (real STA is not extractable from this data — measured ~0.05deg, todo.md).

  step 1 (addb env):   bone-length targets from a B3D -> bonelen.json   (see the extraction in-repo)
  step 2 (export env): pixi run -e export python fit_anny.py bonelen.json
"""
import sys
import json
import numpy as np
import torch
from soma import SOMALayer

# OpenSim joint-pair -> ANNY joint-pair for the same anatomical segment (names: L.public_joint_names).
# forearm (OpenSim elbow->radius_hand is ~half a forearm) and torso (lumbar span != Hips->Chest) have
# unreliable cross-model joint correspondences (measured 95 / 33 mm bias), so they are excluded from
# the fit — the reliable limb + width segments determine the phenotype.
PAIRS = {
    "thigh_r": ("RightLeg", "RightShin"), "shin_r": ("RightShin", "RightFoot"),
    "foot_r": ("RightFoot", "RightToeBase"), "thigh_l": ("LeftLeg", "LeftShin"),
    "shin_l": ("LeftShin", "LeftFoot"), "foot_l": ("LeftFoot", "LeftToeBase"),
    "uparm_r": ("RightArm", "RightForeArm"), "uparm_l": ("LeftArm", "LeftForeArm"),
    "shoulder_w": ("RightArm", "LeftArm"), "hip_w": ("RightLeg", "LeftLeg"),
}


def main(target_json):
    tgt = json.load(open(target_json))
    L = SOMALayer(identity_model_type="anny", device="cpu", mode="torch",
                  enable_procedural_transforms=False).eval()
    name_i = {n: i for i, n in enumerate(L.public_joint_names)}
    npose = len(L.public_joint_names) - 1
    faces = L.faces.detach().cpu().numpy().astype(np.int64)
    pose = torch.eye(3).repeat(1, npose, 1, 1)
    keys = [k for k in PAIRS if k in tgt]
    t = torch.tensor([tgt[k] for k in keys], dtype=torch.float32)

    raw = torch.zeros(11, requires_grad=True)            # identity = sigmoid(raw) in (0,1)
    opt = torch.optim.Adam([raw], lr=0.05)
    for step in range(801):
        ident = torch.sigmoid(raw).unsqueeze(0)
        out = L(pose, ident, scale_params={}, pose2rot=False, apply_correctives=False)
        J = out["transforms"][0, :, :3, 3]               # (78,3) joint world positions at rest
        lens = torch.stack([torch.linalg.norm(J[name_i[a]] - J[name_i[b]]) for a, b in (PAIRS[k] for k in keys)])
        loss = torch.mean((lens - t) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            print(f"  step {step:4d}  rms {torch.sqrt(loss).item()*1000:6.1f} mm")

    ident = torch.sigmoid(raw).detach()
    with torch.no_grad():
        out = L(pose, ident.unsqueeze(0), scale_params={}, pose2rot=False, apply_correctives=False)
    J = out["transforms"][0, :, :3, 3].numpy()
    v = out["vertices"][0].numpy()
    h = float(v[:, 1].max() - v[:, 1].min())
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    mass = abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0) * 1000.0
    print(f"\nfitted ANNY identity (11d): {np.round(ident.numpy(),3)}")
    print(f"{'bone':10s} {'target':>8s} {'anny':>8s} {'err(mm)':>8s}")
    for k in keys:
        a2, b2 = PAIRS[k]
        ln = np.linalg.norm(J[name_i[a2]] - J[name_i[b2]])
        print(f"{k:10s} {tgt[k]*1000:8.0f} {ln*1000:8.0f} {(ln-tgt[k])*1000:8.1f}")
    print(f"\nrecovered phenotype:  height {h:.2f} m   mass {mass:.1f} kg   BMI {mass/(h*h):.1f}")
    print(f"subject recorded:     height {tgt.get('_height_m','?')} m   mass {tgt.get('_mass_kg','?')} kg"
          f"   sex {tgt.get('_sex','?')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/bonelen.json")
