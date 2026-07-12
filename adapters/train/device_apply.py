# SPDX-License-Identifier: MIT
# Apply the trained engine-free calibrator to the real device performance (spike).
# Decodes one window from the <ms>,<hex> raw log and feeds [device_x, wtrained] through netfwd,
# reporting the predicted per-sensor offset angles.  Each sensor is fed the 15-channel layout the
# mag-trained net expects (b3d_to_caldata SINEW_MAG=1): per sensor [accel(3), mag(3), rot(9)].
#
# Orientation is the DEVICE QUATERNION (protocol bytes 10..17, Int16 W/X/Y/Z — Type0x3a.lean), NOT
# a TRIAD(accel,mag) reconstruction.  The firmware quaternion is the drift-prone fused orientation the
# net was trained to correct (training injects R_DG into R_DG·R_clean·R_BS); rebuilding a drift-free
# TRIAD orientation here would both mismatch that drift model and make the mag channel degenerate
# (TRIAD already folds mag in).  Keying off the quaternion makes the mag channel R_q·mg = R_DG·m_world
# informative — it images world-North through the same drift the net estimates.
# The accel channel is unit-normalized per frame (SINEW_ACCEL=grav, train_tic.accel_xform), so the
# accel→m/s^2 scale drops out — only the gravity direction matters.
# Caveats (spike-quality): the device-quaternion frame vs the training world frame (forward=+Z,up=+Y)
# and the Int16 quaternion scale are approximate; orientation and mag are kept in ONE consistent
# (device) frame so the net's drift reasoning still holds, offset from ANNY world.
import subprocess, os, sys
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
LOG = "E:/sinew-moved/data/perf_5472_8134.log"
S, NOUT, NSENS = 32, 90, 15
NIN = NSENS * 15  # [accel(3), mag(3), rot(9)] per sensor — matches the SINEW_MAG training layout
MAG_NORM = True   # feed the mag channel as a unit direction (matches training's unit R_DG·m_world)


def i16(b, i):
    v = b[i] | (b[i + 1] << 8)
    return v - 65536 if v >= 32768 else v


def quat_to_R(q):  # device quaternion (W,X,Y,Z) -> 3x3 rotation (the drift-prone fused orientation)
    w, x, y, z = q / (np.linalg.norm(q) + 1e-9)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# Sensors report asynchronously; carry each sensor's latest (accel, mag) forward and
# snapshot the 15 of them at each new timestamp once all 15 are live.  Collect S snapshots.
latest = {}
snaps = []
cur = None
with open(LOG) as f:
    for line in f:
        ms, hx = line.strip().split(",", 1)
        b = bytes.fromhex(hx)
        if len(b) < 36 or b[0] != 0xFA or b[33] == 0:
            continue
        node = b[6]
        if node >= NSENS:
            continue
        ms = int(ms)
        if cur is not None and ms != cur and len(latest) >= NSENS:
            snaps.append({s: latest[s] for s in range(NSENS)})
            if len(snaps) >= S:
                break
        cur = ms
        latest[node] = (np.array([i16(b, 18), i16(b, 20), i16(b, 22)], float),   # accel
                        np.array([i16(b, 24), i16(b, 26), i16(b, 28)], float),   # mag
                        np.array([i16(b, 10), i16(b, 12), i16(b, 14), i16(b, 16)], float))  # quat W,X,Y,Z
if len(snaps) < S:
    print(f"only {len(snaps)} full snapshots; need {S}"); sys.exit(1)

x = np.zeros((S, NSENS, 15), np.float32)  # per sensor [accel(3), mag(3), rot(9)] — mag-trained layout
for fi in range(S):
    for s in range(NSENS):
        a, mg, q = snaps[fi][s]
        R = quat_to_R(q)                      # device fused orientation R_IMU = R_DG·R_clean·R_BS (drifts)
        # ACCA (TIC Eq.1): global-frame accel R_IMU·a_sensor rotates the mount out -> depends on the
        # drift R_DG alone (training channel: R_DG·R_clean·wa).  Unit-normalize per frame to match the
        # net's SINEW_ACCEL=grav training preprocessing (train_tic.accel_xform): keep the gravity
        # DIRECTION cue, drop the magnitude (the cross-study |a| carrier).  The accel→m/s^2 scale then
        # cancels, so no scale estimate is needed.
        ag = R @ a
        x[fi, s, :3] = ag / (np.linalg.norm(ag) + 1e-6)
        # mag channel: R_IMU·m_sensor = R_DG·m_world — world-North imaged through the SAME drift the net
        # estimates (training channel: R_DG·m_world).  Informative precisely because R_IMU drifts.
        m = R @ mg
        x[fi, s, 3:6] = m / (np.linalg.norm(m) + 1e-9) if MAG_NORM else m
        x[fi, s, 6:] = R.reshape(9)
device_x = x.reshape(S * NIN).astype(np.float32)

w = np.fromfile(os.path.join(here, "wtrained.bin"), dtype=np.float32)
np.concatenate([device_x, w]).astype(np.float32).tofile(os.path.join(here, "inputs.bin"))
spv = lambda n: os.path.join(here, n)
subprocess.run([os.path.join(here, "netfwd.exe"), spv("gemm.spv"), spv("lin.spv"), spv("ln.spv"),
                spv("attn.spv"), spv("ew.spv"), str(S), str(NIN), "64", "4", "128", str(NOUT), "2"],
               cwd=here, check=True, stdout=subprocess.DEVNULL)
out = np.fromfile(os.path.join(here, "outputs.bin"), dtype=np.float32)  # [global 90, local 90]


from sixd import sixd_angle  # pure 6D->angle, extracted to sixd.py for property tests


loc = out[NOUT:].reshape(NSENS, 6)
ang = [sixd_angle(loc[s]) for s in range(NSENS)]
print("predicted per-sensor offset angle (deg):")
print("  " + " ".join(f"{a:4.0f}" for a in ang))
print(f"  mean {np.mean(ang):.1f}°  (low-confidence transfer spike — see caveats in header)")
