# SPDX-License-Identifier: MIT
# Quick-start calibration data: a fixed full-ROM "dance" (rank-3 excitation, the converged
# active design from active_calib_sim.py) replayed under randomized per-sensor mount R_BS
# and drift R_DG, emitting caldata-schema windows.  Because the dance excites all three
# mount DOF (unlike gait's rank-1), a net trained on it recovers the mount from one dance —
# the quick-start: collect once, reuse across subjects by re-randomizing the mounting.
#
#   pixi run -e gpu python active_quick_start.py [K] [out.parquet]
# Output columns/metadata match b3d_to_caldata (x: S*NIN f32, y: 2*NOUT f32; S/NIN/NOUT).
import os, sys
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

S, NSENS, NOUT = 32, 15, 90
NIN = NSENS * 15                         # [accel(3), mag(3), rot(9)] per sensor (mag layout)
OFFSET_DEG, DRIFT_DEG = 30.0, 40.0       # mount / drift injection ranges (b3d_to_caldata)
MAG_WORLD = np.array([0.0, -0.5, 0.866]); MAG_WORLD /= np.linalg.norm(MAG_WORLD)
UP = np.array([0.0, 0.0, 1.0])
K = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
out = sys.argv[2] if len(sys.argv) > 2 else "quick_start.parquet"
rng = np.random.default_rng(0)


def aa_to_R(v):                          # axis-angle (Rodrigues) -> 3x3
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def Rx(a): return aa_to_R(np.array([a, 0, 0]))
def Ry(a): return aa_to_R(np.array([0, a, 0]))
def Rz(a): return aa_to_R(np.array([0, 0, a]))
def r6d(R): return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)   # decode6d inverse


def dance(phase):                        # S bone orientations exciting all 3 axes (Lissajous ROM)
    amp = np.radians(45.0)
    Cs = np.empty((S, 3, 3))
    for t in range(S):
        u = 2 * np.pi * t / S
        a1 = amp * np.sin(u + phase)
        a2 = amp * np.sin(2 * u + 1.0 + phase)
        a3 = amp * np.sin(3 * u + 2.0 + phase)
        Cs[t] = Rz(a3) @ Ry(a2) @ Rx(a1)
    return Cs


xs, ys = [], []
for w in range(K):
    x = np.zeros((S, NSENS, 15), np.float32)
    ydrift = np.zeros((NSENS, 6), np.float32)
    ymount = np.zeros((NSENS, 6), np.float32)
    for s in range(NSENS):
        R_BS = aa_to_R((rng.random(3) - 0.5) * 2 * np.radians(OFFSET_DEG))           # mount
        yaw = (rng.random() - 0.5) * 2 * np.radians(DRIFT_DEG)
        tilt = (rng.random(2) - 0.5) * 2 * np.radians(10.0)
        R_DG = aa_to_R(np.array([tilt[0], yaw, tilt[1]]))                            # yaw-dominant drift
        Cs = dance(phase=s * 0.7)                                                    # per-sensor phase
        ag = R_DG @ UP;  ag /= np.linalg.norm(ag)                                    # accel: up through drift
        mg = R_DG @ MAG_WORLD; mg /= np.linalg.norm(mg)                             # mag: North through drift
        for t in range(S):
            R_IMU = R_DG @ Cs[t] @ R_BS
            x[t, s, :3] = ag
            x[t, s, 3:6] = mg
            x[t, s, 6:] = R_IMU.reshape(9)
        ydrift[s] = r6d(R_DG); ymount[s] = r6d(R_BS)
    xs.append(x.reshape(S * NIN))
    ys.append(np.concatenate([ydrift.reshape(-1), ymount.reshape(-1)]))

tbl = pa.table({
    "x": pa.array(np.stack(xs).tolist(), type=pa.list_(pa.float32())),
    "y": pa.array(np.stack(ys).tolist(), type=pa.list_(pa.float32())),
    "subject": pa.array([f"qs_{i}" for i in range(K)]),
})
tbl = tbl.replace_schema_metadata({"S": str(S), "NIN": str(NIN), "NOUT": str(NOUT),
                                   "source": "active_quick_start dance (rank-3 ROM, random mount/drift)"})
pq.write_table(tbl, out, compression="zstd")
print(f"wrote {out}: {K} windows  S={S} NIN={NIN} NOUT={NOUT}  ({os.path.getsize(out)//1024} KiB)")
