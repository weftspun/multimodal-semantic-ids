# SPDX-License-Identifier: MIT
# Real-data baseline for the engine-free learned calibrator (companion to synth_recover.py).
#
# synth_recover trains the net on held A/T/S/B calibration stances and shows it barely beats the
# identity baseline (~27° R_BS) — the held poses do not excite the mount.  This trains the SAME net
# on REAL normal-motion windows (AddBiomechanics, b3d_to_caldata.py) under an honest protocol:
# a SUBJECT-LEVEL split (no subject in both train and test), each real window used at most once
# (no replacement), and a real-only held-out test.  It reports recovered-vs-true geodesic error in
# degrees for the mount R_BS and the drift R_DG against the identity baseline, so we can read the
# accuracy floor of ordinary motion (vs the unobservable held poses) before the full dataset lands.
#   python real_recover.py <parquet> [D H Fd STACK STEPS LR]   (test subjects via TEST_SUBJECTS)
import subprocess, os, sys
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.environ.get("SINEW_TIC_REF", "/mnt/e/tmp/tic-calib"))
from my_model import TIC  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from pack_weights import pack_winit  # noqa: E402
PARQUET = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SINEW_CALDATA", "/mnt/e/tmp/caldata_addb.parquet")
TEST_SUBJECTS = {"Subject5", "subject_3"}   # unseen people held out for the real-only test
_dflt = [64, 4, 128, 2, 3000, 0.003]
_a = sys.argv[2:8]
D, H, Fd, STACK = (int(_a[i]) if i < len(_a) else _dflt[i] for i in range(4))
STEPS = int(_a[4]) if len(_a) > 4 else _dflt[4]
LR = float(_a[5]) if len(_a) > 5 else _dflt[5]


def decode6d(v6):  # 6D -> R (Gram-Schmidt), columns [a, b, a×b] (matches synth_recover/device_apply)
    a = v6[:3] / (np.linalg.norm(v6[:3]) + 1e-9)
    b = v6[3:] - a * (a @ v6[3:])
    b /= (np.linalg.norm(b) + 1e-9)
    return np.stack([a, b, np.cross(a, b)], axis=1)


def geo_deg(Ra, Rb):  # geodesic angle between two rotations
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


t = pq.read_table(PARQUET)
md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
S, NIN, NOUT = int(md["S"]), int(md["NIN"]), int(md["NOUT"])
NSENS = NOUT // 6
sub = np.array(t["subject"].to_pylist())
X = t["x"].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(len(t), -1).astype(np.float32)   # (N, S*NIN)
Y = t["y"].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(len(t), -1).astype(np.float32)   # (N, 2*NOUT)

is_test = np.array([s in TEST_SUBJECTS for s in sub])
leak = TEST_SUBJECTS & set(sub[~is_test])
assert not leak, f"subject leak across split: {leak}"
trX, trY = X[~is_test], Y[~is_test]
teX, teY = X[is_test], Y[is_test]
NTRAIN, NTEST = len(trX), len(teX)
print(f"subject-level split: train {NTRAIN} windows / {len(set(sub[~is_test]))} subjects, "
      f"test {NTEST} windows / {sorted(TEST_SUBJECTS)}")
print(f"config D={D} H={H} Fd={Fd} STACK={STACK} STEPS={STEPS} LR={LR}  S={S} NIN={NIN} NOUT={NOUT}")
if NTRAIN == 0 or NTEST == 0:
    sys.exit("empty split — check TEST_SUBJECTS against the parquet's subjects")

# caldata.bin = all train X then all train Y (each real window once, no replacement) — caltrain format.
with open(os.path.join(here, "caldata.bin"), "wb") as f:
    trX.tofile(f)
    trY.tofile(f)

# Init weights = torch TIC, packed in the order caltrain/netfwd read (shared layout: pack_weights.py).
model = TIC(stack=STACK, n_input=NIN, n_output=NOUT, multi_head=H, d_model=D, d_ff=Fd).float()
pack_winit(model.state_dict(), STACK).tofile(os.path.join(here, "winit.bin"))

spv = lambda n: os.path.join(here, n)
print("training engine-free caltrain on real normal-motion windows...")
subprocess.run([os.path.join(here, "caltrain.exe"), spv("gemm.spv"), spv("lin.spv"), spv("ln.spv"),
                spv("attn.spv"), spv("ew.spv"), spv("adamn.spv"), str(S), str(NIN), str(D), str(H),
                str(Fd), str(NOUT), str(STACK), str(STEPS), str(LR), str(NTRAIN)], cwd=here, check=True)
loss = np.fromfile(os.path.join(here, "losses.bin"), dtype=np.float32)
print(f"train loss {loss[0]:.3f} -> {loss[-50:].mean():.4f}")

# Held-out recovery on the unseen subjects: netfwd per window, decode, compare to the parquet labels.
w = np.fromfile(os.path.join(here, "wtrained.bin"), dtype=np.float32)
dg_err, off_err, dg_base, off_base = [], [], [], []
for ti in range(NTEST):
    np.concatenate([teX[ti], w]).astype(np.float32).tofile(os.path.join(here, "inputs.bin"))
    subprocess.run([os.path.join(here, "netfwd.exe"), spv("gemm.spv"), spv("lin.spv"), spv("ln.spv"),
                    spv("attn.spv"), spv("ew.spv"), str(S), str(NIN), str(D), str(H), str(Fd),
                    str(NOUT), str(STACK)], cwd=here, check=True, stdout=subprocess.DEVNULL)
    out = np.fromfile(os.path.join(here, "outputs.bin"), dtype=np.float32)
    for s in range(NSENS):
        Rdg_p, Roff_p = decode6d(out[s * 6:s * 6 + 6]), decode6d(out[NOUT + s * 6:NOUT + s * 6 + 6])
        Rdg_t, Roff_t = decode6d(teY[ti][s * 6:s * 6 + 6]), decode6d(teY[ti][NOUT + s * 6:NOUT + s * 6 + 6])
        dg_err.append(geo_deg(Rdg_p, Rdg_t)); off_err.append(geo_deg(Roff_p, Roff_t))
        dg_base.append(geo_deg(np.eye(3), Rdg_t)); off_base.append(geo_deg(np.eye(3), Roff_t))

m = lambda v: (np.mean(v), np.median(v))
print(f"\nHELD-OUT RECOVERY ({NTEST} unseen-subject windows, {NSENS} sensors each):")
print(f"  offset R_BS : recover {m(off_err)[0]:5.1f}° mean / {m(off_err)[1]:5.1f}° median"
      f"   (identity baseline {m(off_base)[0]:.1f}°)")
print(f"  drift  R_DG : recover {m(dg_err)[0]:5.1f}° mean / {m(dg_err)[1]:5.1f}° median"
      f"   (identity baseline {m(dg_base)[0]:.1f}°)")
gain = m(off_base)[0] - m(off_err)[0]
print(f"\nverdict: real normal motion recovers R_BS {gain:+.1f}° vs identity "
      + ("— it beats the held A/T/S/B baseline (~27°/28.6°id)" if m(off_err)[0] < 26
         else "— still near identity at this config/data scale"))
