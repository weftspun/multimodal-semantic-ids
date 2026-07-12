# SPDX-License-Identifier: MIT
# Rank calibration protocols on synthetic ground truth and choose the best.
#
# The learned net underfits even fully-observable bones (synth_recover.py: arms recover only
# ~18° though the poses excite them 80°), so a protocol must be scored NET-INDEPENDENTLY —
# by how well the EXACT algebraic two-sided solver recovers a known mount from that protocol's
# bone orientations.  Per sensor we observe O_k = R_DG · C_k · R_BS over the protocol's clean
# orientations C_k (known); the solver eliminates R_BS via relative rotations
# (O_i O_jᵀ = R_DG (C_i C_jᵀ) R_DGᵀ) to get R_DG, then R_BS = C_kᵀ R_DGᵀ O_k.  A protocol that
# moves every segment about ≥2 independent axes makes the mount fully observable; a 1-axis hinge
# leaves the roll about the long axis under-excited.  Score = recovered mount error (deg), averaged
# over random mounts and sensors, with 1° measurement noise for realism.  Lowest wins.
import numpy as np

NSENS = 15
NAMES = ["Hips", "LLeg", "RLeg", "LShin", "RShin", "LFoot", "RFoot", "Chest", "Head",
         "LArm", "RArm", "LForeA", "RForeA", "LHand", "RHand"]
GROUPS = {"arms": range(9, 15), "legs": range(1, 5), "feet": range(5, 7), "trunk": (0, 7, 8)}
OFFSET_DEG, DRIFT_DEG, NOISE_DEG, TRIALS = 30, 40, 1.0, 200
rng = np.random.default_rng(0)


def aa_to_R(aa):
    th = np.linalg.norm(aa)
    if th < 1e-8:
        return np.eye(3)
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def build_frame(d):
    d = d / (np.linalg.norm(d) + 1e-9)
    x0 = np.cross(d, np.array([0.0, 0, 1]))
    x = np.cross(d, np.array([0.0, 1, 0])) if np.linalg.norm(x0) < 0.1 else x0
    x /= (np.linalg.norm(x) + 1e-9)
    return np.stack([x, d, np.cross(x, d)], axis=1)


def target_dir(p, n):
    s = 1.0 if n in (1, 3, 5, 9, 11, 13) else (-1.0 if n in (2, 4, 6, 10, 12, 14) else 0.0)
    if n in (5, 6):
        return np.array([0.0, 0, 1])
    if n in (9, 10, 11, 12, 13, 14):
        return {0: np.array([0.0, -1, 0]), 1: np.array([s, 0, 0]), 2: np.array([0.0, 0, 1])}.get(p, np.array([0.0, -1, 0]))
    if n in (1, 2, 3, 4):
        return np.array([0.0, -0.8660254, 0.5]) if p == 2 else np.array([0.0, -1, 0])
    return np.array([0.0, 0.8660254, 0.5]) if p == 3 else np.array([0.0, 1, 0])


def geo_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def log_vec(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-7:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * np.sin(th)) * th


def kabsch(U, V):  # rotation A minimising ||A U - V|| for paired unit vectors (rows)
    M = V.T @ U
    Uu, _, Vt = np.linalg.svd(M)
    Dd = np.diag([1, 1, np.sign(np.linalg.det(Uu @ Vt))])
    return Uu @ Dd @ Vt


def solve_two_sided(Cs, Os):  # recover R_DG, R_BS from clean Cs and observed Os = R_DG·C·R_BS
    us, vs = [], []
    for i in range(len(Cs)):
        for j in range(i + 1, len(Cs)):
            ac, ao = log_vec(Cs[i] @ Cs[j].T), log_vec(Os[i] @ Os[j].T)
            if np.linalg.norm(ac) > np.radians(5) and np.linalg.norm(ao) > np.radians(5):
                us.append(ac / np.linalg.norm(ac)); vs.append(ao / np.linalg.norm(ao))
    if len(us) < 2 or np.linalg.matrix_rank(np.array(us), tol=0.1) < 2:
        return np.eye(3), np.eye(3)            # under-excited: mount unobservable
    A = kabsch(np.array(us), np.array(vs))     # R_DG
    Bm = sum(Cs[k].T @ A.T @ Os[k] for k in range(len(Cs))) / len(Cs)
    Uu, _, Vt = np.linalg.svd(Bm)              # orthogonalise -> R_BS
    return A, Uu @ np.diag([1, 1, np.sign(np.linalg.det(Uu @ Vt))]) @ Vt


# Candidate protocols: each returns, per sensor, a list of clean world bone orientations.
def _sweep(n, axes, angles):
    rest = build_frame(target_dir(0, n))       # A-pose orientation of the bone
    return [aa_to_R(np.array(ax) * np.radians(a)) @ rest for ax in axes for a in angles]


def proto_hinge():   # per-segment, ONE axis (bend) — roll about long axis stays unobservable
    return [_sweep(n, [(1, 0, 0)], [-60, -30, 30, 60]) for n in range(NSENS)]


def proto_multiaxis():  # per-segment, all 3 axes — full multi-axis excitation
    return [_sweep(n, [(1, 0, 0), (0, 1, 0), (0, 0, 1)], [-50, 50]) for n in range(NSENS)]


# Typical PCVR dance (VRChat, blog.vive.com / steamcommunity FBT threads): arms/torso/head are
# expressive multi-axis, legs move via hip swing, but footwork is limited — "rules out stomping"
# for tracker stability — so the feet stay near-planted (≈1-axis ankle flex).  The realistic
# incidental-motion baseline: which bones calibrate without a deliberate sequence.
def proto_vr_dance():
    out = []
    for n in range(NSENS):
        if n in (5, 6):                                   # feet: planted, ~1-axis
            out.append(_sweep(n, [(1, 0, 0)], [-20, 20]))
        elif n in range(9, 15):                           # arms/hands: expressive, full multi-axis
            out.append(_sweep(n, [(1, 0, 0), (0, 1, 0), (0, 0, 1)], [-60, 60]))
        else:                                             # hips/legs/trunk/head: moderate multi-axis
            out.append(_sweep(n, [(1, 0, 0), (0, 1, 0), (0, 0, 1)], [-35, 35]))
    return out


# Footwork-heavy dance (shuffle / tap / kicks / heel-toe / pivots): the feet now rotate about
# ≥2 axes (pitch from heel-toe, roll from weight shift, yaw from pivot), so they become observable
# — the same expressive upper body as PCVR dance, but with the feet excited.
def proto_footwork():
    out = proto_vr_dance()
    for n in (5, 6):                                      # feet: multi-axis footwork
        out[n] = _sweep(n, [(1, 0, 0), (0, 1, 0), (0, 0, 1)], [-30, 30])
    return out


PROTOCOLS = {"per-seg hinge (1-axis)": proto_hinge, "PCVR dance (incidental)": proto_vr_dance,
             "footwork dance (feet excited)": proto_footwork, "per-seg multi-axis": proto_multiaxis}

def run(trials=TRIALS, seed=0, verbose=False):  # per-protocol per-sensor mean mount error (deg)
    g = np.random.default_rng(seed)
    results = {}
    for name, fn in PROTOCOLS.items():
        Cs_all = fn()
        err = np.zeros((trials, NSENS))
        for t in range(trials):
            for n in range(NSENS):
                B = aa_to_R((g.random(3) - 0.5) * 2 * np.radians(OFFSET_DEG))
                A = aa_to_R(np.array([(g.random() - 0.5) * 2 * np.radians(10),
                                      (g.random() - 0.5) * 2 * np.radians(DRIFT_DEG),
                                      (g.random() - 0.5) * 2 * np.radians(10)]))
                Cs = Cs_all[n]
                Os = [A @ C @ B @ aa_to_R((g.random(3) - 0.5) * 2 * np.radians(NOISE_DEG)) for C in Cs]
                _, Bh = solve_two_sided(Cs, Os)
                err[t, n] = geo_deg(Bh, B)
        results[name] = err.mean(0)
        if verbose:
            grp = "  ".join(f"{gp}:{np.mean([err[:, s].mean() for s in idx]):4.1f}°" for gp, idx in GROUPS.items())
            print(f"  {name:24s} mount recovery {err.mean():5.1f}° mean / {np.median(err):4.1f}° med   [{grp}]")
    return results


if __name__ == "__main__":
    print(f"calibration-protocol ranking — exact solver, {TRIALS} random mounts, {NOISE_DEG}° noise\n")
    results = run(TRIALS, 0, verbose=True)
    best = min(results, key=lambda k: results[k].mean())
    print(f"\nCHOOSE: '{best}' — {results[best].mean():.1f}° mean mount recovery"
          f" (vs {results['per-seg hinge (1-axis)'].mean():.1f}° for the 1-axis hinge baseline).")
    print("per-sensor (best):  " + " ".join(f"{NAMES[s]}:{results[best][s]:.0f}" for s in range(NSENS)))
