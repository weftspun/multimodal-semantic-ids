# SPDX-License-Identifier: MIT
# Synthetic-ground-truth recovery test for the engine-free learned calibrator.
#
# gen_synth (spec/GenSynth.lean) proves the *classic* solver recovers a known mount to
# ~0.01°; this closes the same loop for the *learned* net, which had only ever trained on
# injected-label data and never been checked against ground truth.  It ports gen_synth's
# A/T/S/B calibration stances and b3d_to_caldata's exact corruption model
# (measured = R_DG·R_clean·R_BS, accel = R_BSᵀ·wa, 6D = first two columns), draws a KNOWN
# random per-sensor mount R_BS + yaw-dominant drift R_DG per window, trains caltrain on a
# train split, runs netfwd on a held-out split, and reports recovered-vs-true geodesic error
# in degrees — against an identity baseline (the injected magnitude) and split into the roll
# component (about the bone long axis +Y) vs the perpendicular, which isolates the
# observability floor in a setting with no data-quality confound.
#
# Windows are i.i.d. random mounts, so a held-out window is an unseen mount — a genuine
# generalisation split (no subject-leak concern here, unlike the real AddBiomechanics set).
import subprocess, os, sys
import numpy as np
import torch

sys.path.insert(0, "E:/tmp/tic-calib")
from my_model import TIC  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
S, NSENS = 32, 15
# D H Fd STACK STEPS NTRAIN overridable for the capacity sweep (underfit vs overfit check)
_dflt = [64, 4, 128, 2, 3000, 2000, 0]
_a = [int(v) for v in sys.argv[1:8]]
D, H, Fd, STACK, STEPS, NTRAIN, ACCA = (_a + _dflt[len(_a):])
# ACCA=1: feed global-frame accel a_IMU = R_DG·g (TIC Eq.1 — depends only on drift, independent
# of the mount R_BS, the disentanglement signal).  ACCA=0: sensor-frame R_BSᵀ·wa (our b3d default).
NIN, NOUT = NSENS * 12, NSENS * 6
NTEST, LR = 48, 0.003
print(f"config D={D} H={H} Fd={Fd} STACK={STACK} STEPS={STEPS} NTRAIN={NTRAIN}")
OFFSET_DEG, DRIFT_DEG = 30, 40           # same injection ranges as b3d_to_caldata
G_WORLD = np.array([0.0, 9.81, 0.0])     # specific force at rest, rig +Y up
rng = np.random.default_rng(0)
torch.manual_seed(0)


def aa_to_R(aa):  # Rodrigues (matches b3d_to_caldata.aa_to_R)
    th = np.linalg.norm(aa)
    if th < 1e-8:
        return np.eye(3)
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def build_frame(d):  # LabRCSF bone frame, columns [x, d(=+Y toward-child), z] (b3d convention)
    d = d / (np.linalg.norm(d) + 1e-9)
    x0 = np.cross(d, np.array([0.0, 0, 1]))
    x = np.cross(d, np.array([0.0, 1, 0])) if np.linalg.norm(x0) < 0.1 else x0
    x /= (np.linalg.norm(x) + 1e-9)
    z = np.cross(x, d)
    return np.stack([x, d, z], axis=1)


def target_dir(p, n):  # world long-axis direction of node n in pose p (port of GenSynth.targetDir)
    s = 1.0 if n in (1, 3, 5, 9, 11, 13) else (-1.0 if n in (2, 4, 6, 10, 12, 14) else 0.0)
    arm = n in (9, 10, 11, 12, 13, 14)
    leg = n in (1, 2, 3, 4)
    foot = n in (5, 6)
    if foot:
        return np.array([0.0, 0, 1])
    if arm:
        return {0: np.array([0.0, -1, 0]), 1: np.array([s, 0, 0]),
                2: np.array([0.0, 0, 1])}.get(p, np.array([0.0, -1, 0]))
    if leg:
        return np.array([0.0, -0.8660254, 0.5]) if p == 2 else np.array([0.0, -1, 0])
    return np.array([0.0, 0.8660254, 0.5]) if p == 3 else np.array([0.0, 1, 0])


def r6d(R):  # first two columns (Zhou 2019), as b3d_to_caldata.r6d
    return np.concatenate([R[:, 0], R[:, 1]])


def decode6d(v6):  # 6D -> R (Gram-Schmidt), columns [a, b, a×b] (matches device_apply.sixd_angle)
    a = v6[:3] / (np.linalg.norm(v6[:3]) + 1e-9)
    b = v6[3:] - a * (a @ v6[3:])
    b /= (np.linalg.norm(b) + 1e-9)
    return np.stack([a, b, np.cross(a, b)], axis=1)


def geo_deg(Ra, Rb):  # geodesic angle between two rotations
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def log_vec(R):  # rotation -> axis*angle (rad)
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-7:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * np.sin(th)) * th


# Clean bone orientation per (pose, sensor) and the clean specific force in the bone frame.
R_clean = np.stack([[build_frame(target_dir(p, n)) for n in range(NSENS)] for p in range(4)])
wa_clean = np.einsum("pnji,j->pni", R_clean, G_WORLD)  # R_cleanᵀ · g, per pose/sensor
POSE = np.arange(S) // (S // 4)  # 8 frames per A/T/S/B stance


def make_window():
    Roff = np.stack([aa_to_R((rng.random(3) - 0.5) * 2 * np.radians(OFFSET_DEG)) for _ in range(NSENS)])
    yaw = (rng.random(NSENS) - 0.5) * 2 * np.radians(DRIFT_DEG)
    tilt = (rng.random((NSENS, 2)) - 0.5) * 2 * np.radians(10)
    Rdg = np.stack([aa_to_R(np.array([tilt[s, 0], yaw[s], tilt[s, 1]])) for s in range(NSENS)])
    x = np.zeros((S, NSENS, 12), np.float32)
    for fi in range(S):
        p = POSE[fi]
        cr = np.einsum("sij,sjk,skl->sil", Rdg, R_clean[p], Roff)     # R_DG·R_clean·R_BS
        ca = np.einsum("sij,j->si", Rdg, G_WORLD) if ACCA \
            else np.einsum("sji,sj->si", Roff, wa_clean[p])           # ACCA: R_DG·g  else R_BSᵀ·wa
        x[fi, :, :3] = ca
        x[fi, :, 3:] = cr.reshape(NSENS, 9)
    y = np.concatenate([np.concatenate([r6d(Rdg[s]) for s in range(NSENS)]),
                        np.concatenate([r6d(Roff[s]) for s in range(NSENS)])]).astype(np.float32)
    return x.reshape(S * NIN).astype(np.float32), y, Rdg, Roff


print(f"generating {NTRAIN} train + {NTEST} test windows (known random mounts)...")
trX, trY = [], []
for _ in range(NTRAIN):
    x, y, _, _ = make_window()
    trX.append(x); trY.append(y)
teX, teRdg, teRoff = [], [], []
for _ in range(NTEST):
    x, _, Rdg, Roff = make_window()
    teX.append(x); teRdg.append(Rdg); teRoff.append(Roff)
np.array(trX, np.float32).tofile(open(os.path.join(here, "caldata.bin"), "wb"))
with open(os.path.join(here, "caldata.bin"), "ab") as f:
    np.array(trY, np.float32).tofile(f)

# Init weights = torch TIC, packed in the order caltrain/netfwd read (copied from train_addb).
model = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=Fd).float()
sd = model.state_dict()
ENC = ["norm1.weight", "norm1.bias", "mha.q_linear.weight", "mha.q_linear.bias",
       "mha.k_linear.weight", "mha.k_linear.bias", "mha.v_linear.weight", "mha.v_linear.bias",
       "mha.output.weight", "mha.output.bias", "norm2.weight", "norm2.bias",
       "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"]
W = lambda n: sd[n].numpy().ravel()
enc = lambda p: np.concatenate([W(f"{p}.{n}") for n in ENC])
parts = [W("input_embedding_layer.embed.weight"), W("input_embedding_layer.embed.bias")]
for i in range(STACK):
    parts.append(enc(f"encoder_banckbone.{i}"))
parts.append(enc("TPM_global.encoder")); parts += [W("TPM_global.mapping.weight"), W("TPM_global.mapping.bias")]
parts.append(enc("TPM_local.encoder")); parts += [W("TPM_local.mapping.weight"), W("TPM_local.mapping.bias")]
np.concatenate(parts).astype(np.float32).tofile(os.path.join(here, "winit.bin"))

spv = lambda n: os.path.join(here, n)
print("training engine-free caltrain on synthetic ground-truth windows...")
subprocess.run([os.path.join(here, "caltrain.exe"), spv("gemm.spv"), spv("lin.spv"), spv("ln.spv"),
                spv("attn.spv"), spv("ew.spv"), spv("adamn.spv"), str(S), str(NIN), str(D), str(H),
                str(Fd), str(NOUT), str(STACK), str(STEPS), str(LR), str(NTRAIN)], cwd=here, check=True)
loss = np.fromfile(os.path.join(here, "losses.bin"), dtype=np.float32)
print(f"train loss {loss[0]:.3f} -> {loss[-50:].mean():.4f}")

# Held-out recovery: run the shipped netfwd per test window, decode, compare to truth.
w = np.fromfile(os.path.join(here, "wtrained.bin"), dtype=np.float32)
dg_err, off_err, dg_base, off_base = [], [], [], []
roll_res, perp_res = [], []
for ti in range(NTEST):
    np.concatenate([teX[ti], w]).astype(np.float32).tofile(os.path.join(here, "inputs.bin"))
    subprocess.run([os.path.join(here, "netfwd.exe"), spv("gemm.spv"), spv("lin.spv"), spv("ln.spv"),
                    spv("attn.spv"), spv("ew.spv"), str(S), str(NIN), str(D), str(H), str(Fd),
                    str(NOUT), str(STACK)], cwd=here, check=True, stdout=subprocess.DEVNULL)
    out = np.fromfile(os.path.join(here, "outputs.bin"), dtype=np.float32)
    for s in range(NSENS):
        Rdg_p = decode6d(out[s * 6:s * 6 + 6])
        Roff_p = decode6d(out[NOUT + s * 6:NOUT + s * 6 + 6])
        dg_err.append(geo_deg(Rdg_p, teRdg[ti][s])); off_err.append(geo_deg(Roff_p, teRoff[ti][s]))
        dg_base.append(geo_deg(np.eye(3), teRdg[ti][s])); off_base.append(geo_deg(np.eye(3), teRoff[ti][s]))
        w_err = log_vec(Roff_p.T @ teRoff[ti][s])      # residual offset rotation (rad)
        roll_res.append(abs(np.degrees(w_err[1])))     # about bone long axis +Y
        perp_res.append(np.degrees(np.linalg.norm([w_err[0], w_err[2]])))

# Excitation per sensor = mean pairwise geodesic between its 4 calibration-pose orientations.
# A bone the poses barely move (small excitation) leaves its mount unobservable → irreducible.
excit = [np.mean([geo_deg(R_clean[a][s], R_clean[b][s]) for a in range(4) for b in range(a + 1, 4)])
         for s in range(NSENS)]
NAMES = ["Hips", "LLeg", "RLeg", "LShin", "RShin", "LFoot", "RFoot", "Chest", "Head",
         "LArm", "RArm", "LForeA", "RForeA", "LHand", "RHand"]
GROUPS = {"arms": range(9, 15), "legs": range(1, 5), "feet": range(5, 7), "trunk": (0, 7, 8)}
off_by_s = np.array(off_err).reshape(NTEST, NSENS)

m = lambda v: (np.mean(v), np.median(v))
print(f"\nHELD-OUT RECOVERY ({NTEST} unseen mounts, {NSENS} sensors each):")
print(f"  offset R_BS  : recover {m(off_err)[0]:5.1f}° mean / {m(off_err)[1]:5.1f}° median"
      f"   (identity baseline {m(off_base)[0]:.1f}°)")
print(f"  drift  R_DG  : recover {m(dg_err)[0]:5.1f}° mean / {m(dg_err)[1]:5.1f}° median"
      f"   (identity baseline {m(dg_base)[0]:.1f}°)")
print(f"  offset residual split: roll(about +Y) {m(roll_res)[0]:4.1f}°   perpendicular {m(perp_res)[0]:4.1f}°")
print(f"\noffset recovery vs pose excitation (per body group):")
for g, idx in GROUPS.items():
    idx = list(idx)
    print(f"  {g:5s}: recover {off_by_s[:, idx].mean():5.1f}°   pose-excitation {np.mean([excit[s] for s in idx]):5.1f}°"
          f"   sensors {','.join(NAMES[s] for s in idx)}")
print("  per-sensor:  " + "  ".join(f"{NAMES[s]}:{off_by_s[:, s].mean():.0f}°/{excit[s]:.0f}ex" for s in range(NSENS)))
well = np.mean([off_by_s[:, s].mean() for s in range(NSENS) if excit[s] > 45])
print(f"\nverdict: well-excited bones (excitation>45°) recover to {well:.1f}° mean — "
      + ("the floor is POSE OBSERVABILITY, not the net: under-excited bones are unrecoverable by construction"
         if well < 8 else "even well-excited bones miss; the net/training is also a factor"))
