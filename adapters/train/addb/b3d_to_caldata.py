# SPDX-License-Identifier: MIT
# Builds our own 15-sensor calibration dataset from AddBiomechanics B3D (werling2024addbiomechanics),
# fit to ANNY with the acceleration channel (SIP-style body fitting needs accel — von Marcard 2017),
# marker-based orientation (recovers roll; no FK, since setPositions segfaults in the wheel),
# and phenotype-equity columns across subjects.  Per body, per frame, Kabsch(marker local
# offsets → observed world) gives the OpenSim body world transform (R with roll, t); R maps
# to the ANNY LabRCSF bone frame via a constant R_align, and t becomes the specific force
# p̈ - g in the sensor frame (+9.81 up at rest; same sign as synthetic + the device feed).  A
# per-sensor offset + drift per window yields (corrupted
# accel+rot, offset+drift) pairs.  NIN = 15*(3+9) = 180.
# Runs in the WSL pixi env (linux-64 / py3.9 / nimblephysics):
#   pixi run python b3d_to_caldata.py --probe <some.b3d>
#   pixi run python b3d_to_caldata.py <dir-of-b3d> <out.parquet> [seconds_of_motion_per_subject]
import sys, os, glob
import numpy as np
import nimblephysics as nimble

S = int(os.environ.get("SINEW_S", "32"))   # window length (frames); sweep with SINEW_S to move the floor
STRIDE, OFFSET_DEG, DRIFT_DEG = S, 30, 40
NSENS = 15
GRAV = np.array([0.0, -9.81, 0.0])
SESSION = os.environ.get("SINEW_SESSION") == "1"  # 1: one mount/drift per subject (a real session, for
#                                                   session-aggregated eval); default: fresh per window
# Windows sit at each trial's native rate; the per-subject cap is SI (seconds of motion), converted to
# a window count with the subject's native timestep.  (Fixed-rate resampling to the 62.5 Hz device rate
# was tried and reverted: it equalizes the 100–250 Hz spread but does not move cross-study held-out.)
# All orientations/positions/accel live in ONE frame — ANNY world: forward=+Z, up=+Y,
# handedness=RH (X=left).  OpenSim world is forward=+X, up=+Y, right=+Z; R_W maps it to ANNY.
WORLD_FRAME = "forward=+Z,up=+Y,handedness=RH"
R_W = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
RIG_NPZ = "/mnt/e/sinew-moved/soma_rig.npz"
MIN_MARKERS = 3  # need ≥3 non-collinear markers for a full Kabsch orientation
MAG = os.environ.get("SINEW_MAG") == "1"  # SINEW_MAG=1: append an emulated magnetometer channel per
#   sensor.  A magnetometer reads the world field in the sensor's TRUE frame, but the IMU reports it
#   through its own DRIFTED orientation R_IMU=R_DG·R_clean·R_BS, giving R_IMU·(R_clean·R_BS)ᵀ·m_world =
#   R_DG·m_world: constant over the window, mount-independent (like ACCA), and it images world-North
#   through the drift.  Gravity (accel) pins R_DG's tilt but is yaw-symmetric; mag adds the missing yaw,
#   so R_DG becomes fully observable — the principled removal of the drift-facing carrier (which the net
#   otherwise has to memorize, and which adversarial training can't strip without killing signal).
MAG_WORLD = np.array([0.0, -0.5, 0.866])           # ANNY world: strong horizontal North (+Z) + ~30° dip
MAG_WORLD = MAG_WORLD / np.linalg.norm(MAG_WORLD)  # so the horizontal component carries the yaw cue
PER = 15 if MAG else 12  # input channels per sensor: [accel(3), (mag(3),) rot(9)]

ANNY_SOMA = ["Hips", "LeftLeg", "RightLeg", "LeftShin", "RightShin", "LeftFoot", "RightFoot",
             "Chest", "Head", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm",
             "LeftHand", "RightHand"]
OS_BODY = {0: ["pelvis"], 1: ["femur_l"], 2: ["femur_r"], 3: ["tibia_l"], 4: ["tibia_r"],
           5: ["calcn_l", "talus_l"], 6: ["calcn_r", "talus_r"], 7: ["torso"], 8: ["torso"],
           9: ["humerus_l"], 10: ["humerus_r"], 11: ["ulna_l", "radius_l"],
           12: ["ulna_r", "radius_r"], 13: ["hand_l"], 14: ["hand_r"]}


def aa_to_R(aa):
    th = np.linalg.norm(aa)
    if th < 1e-8:
        return np.eye(3)
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def r6d(R):
    return np.concatenate([R[:, 0], R[:, 1]])


# Distal arm bones have <3 markers → orientation by swing (long axis from positions,
# canonical roll).  LONG_AXIS[node] = (from_node, to_node) for the long-axis direction.
SWING = {11, 12, 13, 14}
LONG_AXIS = {11: (11, 13), 12: (12, 14), 13: (11, 13), 14: (12, 14)}
# The forearms/hands have <3 markers in every study (0 in Carter, 1–2 elsewhere), so a marker-derived
# orientation is impossible.  The OpenSim fit places the elbow + wrist joint centers regardless of
# forearm markers (whole-body fit), so the swing comes from those instead — real motion vs the
# degenerate marker fallback.  Hands follow the forearm direction (no hand-tip joint); the forearm
# twist (pronation) stays unobservable from joint centers.
SWING_JC = {1: ("hip_l", "walker_knee_l"), 2: ("hip_r", "walker_knee_r"),       # femur
            3: ("walker_knee_l", "ankle_l"), 4: ("walker_knee_r", "ankle_r"),   # tibia
            5: ("ankle_l", "mtp_l"), 6: ("ankle_r", "mtp_r"),                   # foot
            9: ("acromial_l", "elbow_l"), 10: ("acromial_r", "elbow_r"),        # humerus
            11: ("elbow_l", "radius_hand_l"), 12: ("elbow_r", "radius_hand_r"), # forearm
            13: ("elbow_l", "radius_hand_l"), 14: ("elbow_r", "radius_hand_r")} # hand (follows forearm)


def build_frame(d):  # LabRCSF bone frame from the toward-child long axis d (= +Y);
    d = d / (np.linalg.norm(d) + 1e-9)  # +X bend axis, +Z right-handed (play_log convention)
    x0 = np.cross(d, np.array([0.0, 0, 1]))
    x = np.cross(d, np.array([0.0, 1, 0])) if np.linalg.norm(x0) < 0.1 else x0
    x /= (np.linalg.norm(x) + 1e-9)
    z = np.cross(x, d)
    return np.stack([x, d, z], axis=1)  # columns [x, y(=d), z]


def kabsch(local, world):  # (n,3),(n,3) -> R(3,3), t(3): world ≈ R@local + t
    lc, wc = local.mean(0), world.mean(0)
    H = (local - lc).T @ (world - wc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, wc - R @ lc


# Child SOMA joint per device node (toward-child defines the LabRCSF +Y long axis).
ANNY_CHILD = ["Spine1", "LeftShin", "RightShin", "LeftFoot", "RightFoot", "LeftToeBase",
              "RightToeBase", "Neck1", "HeadEnd", "LeftForeArm", "RightForeArm", "LeftHand",
              "RightHand", "LeftHandMiddle1", "RightHandMiddle1"]


def anny_rcsf_R():  # ANNY bones in the LabRCSF rest frame (not the raw mirrored rig frame)
    d = np.load(RIG_NPZ, allow_pickle=True)
    names = list(d["joint_names"])
    pos = d["rest_pos"].astype(np.float64)
    out = []
    for n in range(NSENS):
        j, c = names.index(ANNY_SOMA[n]), names.index(ANNY_CHILD[n])
        out.append(build_frame(pos[c] - pos[j]))
    return np.stack(out)


def body_setup(subj):
    osim = subj.readOpenSimFile(subj.getNumProcessingPasses() - 1, "")
    skel = osim.skeleton
    names = [skel.getBodyNode(i).getName() for i in range(skel.getNumBodyNodes())]
    body = [next((c for c in OS_BODY[n] if c in names), None) for n in range(NSENS)]
    # markers grouped by OpenSim body name (rest pose getWorldTransform works; no setPositions)
    by_body = {}
    for mname, (bn, off) in osim.markersMap.items():
        by_body.setdefault(bn.getName(), []).append((mname, np.array(off, dtype=np.float64)))
    rest = {b: np.array(skel.getBodyNode(b).getWorldTransform().rotation()) for b in names}
    restp = {b: np.array(skel.getBodyNode(b).getWorldTransform().translation()) for b in names}
    return body, by_body, rest, restp


# Primary RCSF child per node → segment whose rest length characterises the phenotype.
SEG_CHILD = {0: 7, 1: 3, 2: 4, 3: 5, 4: 6, 7: 8, 9: 11, 10: 12, 11: 13, 12: 14}
SEG_NODES = sorted(SEG_CHILD)


def seg_lengths(body, restp):  # rest segment lengths (m) for the phenotype fit
    return [float(np.linalg.norm(restp[body[SEG_CHILD[n]]] - restp[body[n]])) for n in SEG_NODES]


def probe(path):
    subj = nimble.biomechanics.SubjectOnDisk(path)
    body, by_body, _, _ = body_setup(subj)
    print("sex:", subj.getBiologicalSex(), "H:", round(subj.getHeightM(), 2), "M:",
          round(subj.getMassKg(), 1), "trials:", subj.getNumTrials())
    print("15-bone map:", body)
    print("markers/body:", {n: len(by_body.get(body[n], [])) for n in range(NSENS)})


def extract(b3d_dir, out_npz, cap_sec):
    arcsf = anny_rcsf_R()
    paths = sorted(glob.glob(os.path.join(b3d_dir, "**", "*.b3d"), recursive=True))
    print(f"{len(paths)} B3D files; cap {cap_sec:g} s of motion/subject (windows derived at native rate)")
    rng = np.random.default_rng(0)
    Xs, Ys, meta = [], [], []
    for pi, path in enumerate(paths):
        try:
            subj = nimble.biomechanics.SubjectOnDisk(path)
            body, by_body, rest, restp = body_setup(subj)
            if any(b is None for b in body):
                continue
            sk = subj.readSkel(0, ignoreGeometry=True)   # joint name → index, for the jointCenter swing
            JI = {sk.getJoint(i).getName(): i for i in range(sk.getNumJoints())}
            # R_align maps OpenSim body world → ANNY LabRCSF bone (in ANNY world), so
            # R_W·R_os(t)·R_align == anny RCSF bone orientation; at rest it = arcsf.
            R_align = [rest[body[n]].T @ R_W.T @ arcsf[n] for n in range(NSENS)]
            seg = seg_lengths(body, restp)
            # Source dataset's recorded sex — kept only as one anthropometric equity
            # stratifier alongside the recovered phenotype; not gender.
            recorded_sex = subj.getBiologicalSex()
            subj_wins = []
            cap = None
            for t in range(subj.getNumTrials()):
                n = subj.getTrialLength(t)
                dt = subj.getTrialTimestep(t)
                if cap is None:   # SI: seconds of motion/subject → windows at this subject's native rate
                    cap = max(1, round(cap_sec / (S * dt)))
                if n < S + 2:
                    continue
                frames = subj.readFrames(t, 0, n, includeSensorData=True,
                                         includeProcessingPasses=True)   # for jointCenters (swing)
                Rw = np.zeros((len(frames), NSENS, 3, 3))
                pos = np.zeros((len(frames), NSENS, 3))
                prev = None
                lastR = [None] * NSENS
                for fi, fr in enumerate(frames):
                    obs = {m[0]: np.array(m[1], dtype=np.float64) for m in fr.markerObservations}
                    jc = (np.array(fr.processingPasses[0].jointCenters).reshape(-1, 3) @ R_W.T
                          if fr.processingPasses else None)   # fitted joint world centers → ANNY frame
                    Rmark = [None] * NSENS
                    for s in range(NSENS):  # Kabsch where ≥3 markers; carry position forward
                        ms = [(mn, o) for mn, o in by_body.get(body[s], []) if mn in obs]
                        if len(ms) >= MIN_MARKERS:
                            R, tt = kabsch(np.array([o for _, o in ms]),
                                           np.array([obs[mn] for mn, _ in ms]))
                            Rmark[s] = R_W @ R @ R_align[s]  # → ANNY world frame
                            pos[fi, s] = R_W @ tt
                        elif ms:
                            pos[fi, s] = R_W @ obs[ms[0][0]]
                        else:
                            pos[fi, s] = prev[s] if prev is not None else 0.0
                    for s in range(NSENS):  # orientations: Kabsch; distal=swing; else carry forward
                        if Rmark[s] is not None:
                            Rw[fi, s] = Rmark[s]
                        elif s in SWING_JC and jc is not None and SWING_JC[s][0] in JI and SWING_JC[s][1] in JI:
                            pj, dj = SWING_JC[s]  # marker-sparse bone: real swing from the OpenSim fit
                            Rw[fi, s] = build_frame(jc[JI[dj]] - jc[JI[pj]])
                        elif s in SWING:
                            a, b = LONG_AXIS[s]
                            Rw[fi, s] = build_frame(pos[fi, b] - pos[fi, a])  # marker fallback (no joints)
                        elif lastR[s] is not None:
                            Rw[fi, s] = lastR[s]
                        else:
                            Rw[fi, s] = np.eye(3)
                        lastR[s] = Rw[fi, s]
                    prev = pos[fi].copy()
                # Specific force an accelerometer reads = coordinate accel − gravity field = p̈ − GRAV
                # (GRAV points down, so at rest this is +9.81 UP).  Matches the synthetic Stage-1 channel
                # (train_tic.py G=[0,+9.81,0], R_DG·G) and the device_apply device feed (raw accel is
                # +up specific force) — the earlier GRAV − p̈ was the negated sign, inconsistent with both.
                accw = np.stack([np.gradient(np.gradient(pos[:, s], dt, axis=0), dt, axis=0) - GRAV
                                 for s in range(NSENS)], axis=1)
                accS = np.einsum("fsij,fsj->fsi", Rw.transpose(0, 1, 3, 2), accw)
                for st in range(0, len(Rw) - S, STRIDE):
                    subj_wins.append((Rw[st:st + S].copy(), accS[st:st + S].copy()))
                if len(subj_wins) >= cap * 3:  # enough to sample cap; bounds huge subjects
                    break
            rng.shuffle(subj_wins)
            sub_name = os.path.splitext(os.path.basename(path))[0]
            study = os.path.basename(os.path.dirname(os.path.dirname(path))).replace("_Formatted_With_Arm", "")
            hm, mk = round(subj.getHeightM(), 2), round(subj.getMassKg(), 1)

            def mkmount():  # random per-sensor mount R_BS (≤OFFSET_DEG) + yaw-dominant drift R_DG
                aa = (rng.random((NSENS, 3)) - 0.5) * 2 * (OFFSET_DEG / 180 * np.pi)
                yaw = (rng.random(NSENS) - 0.5) * 2 * (DRIFT_DEG / 180 * np.pi)
                tilt = (rng.random((NSENS, 2)) - 0.5) * 2 * (10 / 180 * np.pi)
                return (np.stack([aa_to_R(aa[s]) for s in range(NSENS)]),
                        np.stack([aa_to_R(np.array([tilt[s, 0], yaw[s], tilt[s, 1]])) for s in range(NSENS)]))

            sess = mkmount() if SESSION else None    # SINEW_SESSION=1: one mount/drift for the whole
            for wr, wa in subj_wins[:cap]:            # subject (a real session), else fresh per window
                Roff, Rdg = sess if SESSION else mkmount()
                cr = np.einsum("sij,fsjk->fsik", Rdg, wr @ Roff[None])  # R_DG·R_clean·R_BS
                # ACCA (TIC Eq.1): the IMU reports acceleration in its own global frame, R_IMU·a_sensor,
                # in which the mount cancels — R_DG·R_clean·R_BS · R_BSᵀ·wa = R_DG·R_clean·wa.  The accel
                # then depends on the drift R_DG (and the motion) alone, independent of R_BS, which is the
                # cue that lets the net separate drift from mount (slangtrain/synth_recover.py).
                ca = np.einsum("sij,fsjk,fsk->fsi", Rdg, wr, wa)  # global-frame accel, independent of R_BS
                parts = [ca]
                if MAG:  # R_DG·m_world, constant over the window — images North through the drift (yaw)
                    mag = np.einsum("sij,j->si", Rdg, MAG_WORLD)
                    parts.append(np.broadcast_to(mag, (S, NSENS, 3)))
                parts.append(cr.reshape(S, NSENS, 9))
                x = np.concatenate(parts, axis=2).reshape(S, NSENS * PER)
                dg6 = np.concatenate([r6d(Rdg[s]) for s in range(NSENS)])
                off6 = np.concatenate([r6d(Roff[s]) for s in range(NSENS)])
                Xs.append(x.astype(np.float32))
                Ys.append(np.concatenate([dg6, off6]).astype(np.float32))
                meta.append((sub_name, recorded_sex, hm, mk, seg, study))
        except Exception as e:
            import traceback
            print(f"skip {os.path.basename(path)}: {e}")
            traceback.print_exc()
        if pi % 10 == 0:
            print(f"  {pi}/{len(paths)} files, {len(Xs)} windows", flush=True)
    import pyarrow as pa, pyarrow.parquet as pq
    X, Y = np.stack(Xs), np.stack(Ys)
    N = X.shape[0]
    tbl = pa.table({
        "subject": [m[0] for m in meta],
        "study": [m[5] for m in meta],  # AddBiomechanics constituent study (domain label, from the path)
        "source": ["real:addbiomechanics"] * len(meta),  # real measured motion; the equity gate
        # (test/eval) admits source=="real" only — synthetic (ANNY-generated) windows are train-only
        "recorded_sex": [m[1] for m in meta],  # anthropometric equity stratifier, not gender
        "height_m": [m[2] for m in meta],
        "mass_kg": [m[3] for m in meta],
        "seglen_m": pa.array([m[4] for m in meta], type=pa.list_(pa.float32())),  # rest seg lengths
        "x": pa.array(list(X.reshape(N, -1)), type=pa.list_(pa.float32())),       # S*NIN
        "y": pa.array(list(Y), type=pa.list_(pa.float32())),                      # 2*NOUT
    })
    # self-describing units for every channel
    tbl = tbl.replace_schema_metadata({
        "S": str(S), "NIN": str(NSENS * PER), "NOUT": str(NSENS * 6),
        "seg_nodes": ",".join(map(str, SEG_NODES)),
        "height_m.units": "m", "mass_kg.units": "kg", "seglen_m.units": "m",
        "x.layout": f"S x [per-sensor: accel(3){' + mag(3)' if MAG else ''} + rot(9)] for 15 sensors",
        "x.accel.units": "m/s^2 (global-frame specific force, ACCA: R_DG·R_clean·wa)",
        **({"x.mag.units": "unitless unit-vector (R_DG·m_world, drift-imaged North; yaw observability)"} if MAG else {}),
        "x.rot.units": "unitless (3x3 row-major)",
        "y.layout": "[drift R_DG 6D x15] + [offset R_BS 6D x15]", "y.units": "unitless (6D rotation)",
        "dt.note": "windows sit at each trial's native timestep (s); per-subject cap is SI seconds of "
                   "motion, converted to a window count with the native rate",
        # every coordinate-frame convention this dataset is built in:
        "world_frame": WORLD_FRAME,  # forward=+Z, up=+Y, RH (X=left) — all data normalized here
        "bone_frame": "LabRCSF: +Y toward child, +X bend axis, +Z right-handed; "
                      "arms mirrored, legs +Y down (meshula/LabRCSF joint-orientation)",
        "rotation_rep": "6D = first two columns of the 3x3 rotation (Zhou 2019)",
        "rot_storage": "3x3 row-major",
        "accel_convention": "ACCA (TIC Eq.1): global-frame accel R_IMU·a_sensor = R_DG·R_clean·wa, "
                            "depends on drift R_DG alone, independent of mount R_BS; "
                            "specific force = g - p_ddot (device reads gravity DOWN)",
        "gravity_world": "(0, -9.81, 0) m/s^2 (up=+Y)",
        "corruption_model": "measured_rot = R_DG · R_clean · R_BS; "
                            "measured_accel = R_DG · R_clean · wa (ACCA, global frame, indep. R_BS); "
                            "R_DG = world-frame yaw-dominant fusion drift, R_BS = sensor-frame mount",
        "sensor_order": "device nodes 0..14: Hips, L/R UpperLeg, L/R LowerLeg, L/R Foot, "
                        "Chest, Head, L/R UpperArm, L/R LowerArm, L/R Hand",
        "equity_gate": "test/eval admits source=='real' subjects only, split at subject level; "
                       "synthetic (ANNY-generated) windows are train-only; per-group metrics report "
                       "groups with no real test subject as unmeasurable (no equity claim)",
    })
    pq.write_table(tbl, out_npz, compression="zstd")
    rs = [m[1] for m in meta]
    print(f"wrote {out_npz}: {N} windows  X{X.shape} Y{Y.shape}  (parquet/zstd)")
    print("phenotype recorded_sex:", {s: rs.count(s) for s in set(rs)})


if __name__ == "__main__":
    if sys.argv[1] == "--probe":
        probe(sys.argv[2])
    else:
        extract(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "caldata_addb.parquet",
                float(sys.argv[3]) if len(sys.argv) > 3 else 200.0)   # seconds of motion per subject
