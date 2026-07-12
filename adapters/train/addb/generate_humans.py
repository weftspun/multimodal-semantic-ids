"""Generate synthetic humans from ANNY's 11-dim identity and measure their phenotype, to check the
reachable body-shape range — including the obese band the real data lacks.  Runs in the export env.

  pixi run -e export python generate_humans.py

ANNY reaches BMI ~11..32 (obese is reachable in SHAPE), so it can synthesise the missing band for
training breadth.  Caveat: ANNY is a rigid skeleton + LBS, so synthetic-obese carries the geometry
but NOT the soft-tissue artifact that makes high-BMI calibration hard; synthetic is train-only and
the equity gate stays real (see todo.md).  A BMI-scaled soft-tissue mount perturbation is what would
make synthetic-obese honest.
"""
import numpy as np
import torch
from soma import SOMALayer

L = SOMALayer(identity_model_type="anny", device="cpu", mode="torch",
              enable_procedural_transforms=False).eval()
npose = len(L.public_joint_names) - 1
faces = L.faces.detach().cpu().numpy().astype(np.int64)
DENS = 1000.0  # kg/m^3, ~ body density


def bodies(idents):  # (N,11) -> vertices (N,V,3)
    idents = torch.as_tensor(idents, dtype=torch.float32)
    pose = torch.eye(3).repeat(idents.shape[0], npose, 1, 1)
    with torch.no_grad():
        out = L(pose, idents, scale_params={}, pose2rot=False, apply_correctives=False)
    return out["vertices"].numpy()


def measure(v):  # height (m), mass (kg) from enclosed mesh volume, BMI
    h = float(v[:, 1].max() - v[:, 1].min())
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    mass = abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0) * DENS
    return h, mass, mass / (h * h)


if __name__ == "__main__":
    tests = [("baseline 0.5", np.full(11, 0.5))]
    for d in range(11):
        for val in (0.0, 1.0):
            x = np.full(11, 0.5); x[d] = val
            tests.append((f"dim{d}={val:.0f}", x))
    rng = np.random.default_rng(0)
    for k in range(6):
        tests.append((f"random{k}", rng.random(11)))
    V = bodies(np.stack([x for _, x in tests]).astype(np.float32))
    print(f"generated {len(tests)} synthetic ANNY humans  (mesh V={V.shape[1]} verts)\n")
    bmis = []
    for (name, _), v in zip(tests, V):
        h, m, bmi = measure(v)
        bmis.append(bmi)
        tag = "  <-- OBESE" if bmi >= 30 else ("  <- overweight" if bmi >= 25 else "")
        print(f"  {name:14s} h={h:.2f}m mass={m:5.1f}kg BMI={bmi:4.1f}{tag}")
    bmis = np.array(bmis)
    print(f"\nBMI range: {bmis.min():.1f} .. {bmis.max():.1f}   obese reachable:",
          "YES" if bmis.max() >= 30 else "NO")
