# SPDX-License-Identifier: MIT
# Dump a real-motion animation for the native viewer: read a B3D, recover each frame's clean
# ANNY-world bone rotations (the same marker-Kabsch pipeline as b3d_to_caldata), convert to
# quaternions per device channel, and write frames the streamer (anim_pheno.py --real) loops
# into anny_demo over OSC.  Runs in the WSL addb pixi env (nimblephysics).
#   pixi run python dump_anim.py <subject.b3d> [out.json] [max_frames]
import sys, os, json
import numpy as np
import nimblephysics as nimble
sys.path.insert(0, "/mnt/e/sinew-moved/slangtrain/addb")
import b3d_to_caldata as B

DEVICE = ["Hips", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg", "LeftFoot",
           "RightFoot", "Chest", "Head", "LeftUpperArm", "RightUpperArm", "LeftLowerArm",
           "RightLowerArm", "LeftHand", "RightHand"]
N = B.NSENS
# Shoulders (LeftArm 9, RightArm 10) are ball joints — drive them by the long axis to their forearm
# child (LeftForeArm 11, RightForeArm 12) with canonical roll, so a noisy marker roll can't twist
# the deltoid (matches anny_demo's ball-joint handling).
ARM_SWING = {9: 11, 10: 12}
PATH = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/e/tmp/real_anim.json"
MAXF = int(sys.argv[3]) if len(sys.argv) > 3 else 600


def swing_rot(a, b):  # minimal rotation taking unit a -> unit b (pure swing, no roll/twist)
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-6:  # antiparallel: 180° about any perpendicular
        perp = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1, 0])
        perp /= np.linalg.norm(perp)
        K = np.array([[0, -perp[2], perp[1]], [perp[2], 0, -perp[0]], [-perp[1], perp[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def mat2quat(R):  # 3x3 -> (w,x,y,z)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return q


subj = nimble.biomechanics.SubjectOnDisk(PATH)
arcsf = B.anny_rcsf_R()  # LabRCSF rest frames the extractor's Rw is expressed in
body, by_body, rest, restp = B.body_setup(subj)
if any(b is None for b in body):
    sys.exit("subject's body mapping is incomplete")
R_align = [rest[body[n]].T @ B.R_W.T @ arcsf[n] for n in range(N)]
# Convert Rw from the dataset LabRCSF convention to the viewer's rig rest convention:
# send = Rw · arcsfᵀ · rest_R_rig applies the bone's world delta-from-rest onto the rig rest
# orientation, so rest maps to the standing rig pose and motion follows physically.
_rig = np.load(B.RIG_NPZ, allow_pickle=True)
_rignames = list(_rig["joint_names"])
_restR = _rig["rest_R"].astype(np.float64)
restR_rig = np.stack([_restR[_rignames.index(B.ANNY_SOMA[n])] for n in range(N)])
conv = [arcsf[n].T @ restR_rig[n] for n in range(N)]  # constant per-bone right factor
# Pick the trial with the most TRAVEL (body-centroid displacement), so root motion is visible —
# a treadmill/in-place exercise has lots of limb motion but no travel; overground locomotion moves.
best_t, best_travel = 0, -1.0
for t in range(subj.getNumTrials()):
    nt = subj.getTrialLength(t)
    if nt < 60:
        continue
    fr = subj.readFrames(t, 0, min(nt, 300), includeSensorData=True, includeProcessingPasses=False)
    cents = [np.mean([np.array(o, dtype=np.float64) for _, o in f.markerObservations], axis=0)
             for f in fr if f.markerObservations]
    travel = float(np.linalg.norm(np.stack(cents).max(0) - np.stack(cents).min(0))) if len(cents) > 1 else 0.0
    if travel > best_travel:
        best_travel, best_t = travel, t
trial = best_t
nframes = min(subj.getTrialLength(trial), MAXF)
print(f"max-travel trial {trial} ({best_travel:.2f} m centroid travel) of {subj.getNumTrials()}")
frames = subj.readFrames(trial, 0, nframes, includeSensorData=True, includeProcessingPasses=False)

out = []
lastR = [None] * N
prev = None
hips0 = None  # frame-0 hip position; root motion is streamed as the delta from it
for fr in frames:
    obs = {m[0]: np.array(m[1], dtype=np.float64) for m in fr.markerObservations}
    pos = np.zeros((N, 3))
    Rmark = [None] * N
    for s in range(N):
        ms = [(mn, o) for mn, o in by_body.get(body[s], []) if mn in obs]
        if len(ms) >= B.MIN_MARKERS:
            R, tt = B.kabsch(np.array([o for _, o in ms]), np.array([obs[mn] for mn, _ in ms]))
            Rmark[s] = B.R_W @ R @ R_align[s]
            pos[s] = B.R_W @ tt
        elif ms:
            pos[s] = B.R_W @ obs[ms[0][0]]
        else:
            pos[s] = prev[s] if prev is not None else 0.0
    Rw = [None] * N
    for s in range(N):
        if s in ARM_SWING:  # shoulders: pure geodesic swing re-pointing the rest arm — preserves
            # the rest roll so the deltoid neither twists nor rolls wrong.
            Rw[s] = swing_rot(arcsf[s][:, 1], pos[ARM_SWING[s]] - pos[s]) @ arcsf[s]
        elif Rmark[s] is not None:
            Rw[s] = Rmark[s]
        elif s in B.SWING:
            a, b = B.LONG_AXIS[s]
            Rw[s] = B.build_frame(pos[b] - pos[a])
        elif lastR[s] is not None:
            Rw[s] = lastR[s]
        else:
            Rw[s] = np.eye(3)
        lastR[s] = Rw[s]
    prev = pos.copy()
    if hips0 is None:
        hips0 = pos[0].copy()
    # Rw · conv: rebase each bone from the LabRCSF roll convention to the ANNY rig convention.
    frame = {DEVICE[s]: [round(float(x), 5) for x in mat2quat(Rw[s] @ conv[s])] for s in range(N)}
    frame["_root"] = [round(float(x), 5) for x in (pos[0] - hips0)]  # hip translation delta (m)
    out.append(frame)

dt = float(subj.getTrialTimestep(trial))  # seconds/frame at the trial's native rate
json.dump({"dt": dt, "frames": out}, open(OUT, "w"))
print(f"wrote {len(out)} frames dt={dt:.4f}s ({1/dt:.0f} Hz) from {os.path.basename(PATH)} trial {trial} -> {OUT}")
