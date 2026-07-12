# SPDX-License-Identifier: MIT
# Export K held-out caldata windows to a flat selftest.bin for the on-device OME
# self-test (sinew_tui --tic-selftest).  Each window carries the grav-normalized
# input x (matching train_tic.py SINEW_ACCEL=grav, the working net's preprocessing)
# plus the ground-truth drift/mount target y, so the C deploy path (netfwd_forward
# + sixd_to_m + R_DGᵀ·R_sensor·R_BSᵀ) reproduces train_tic.py's offline OME.
#
# Layout: int32 [K, S, NIN, NOUT] then K × (x[S*NIN] float32, y[2*NOUT] float32).
#   pixi run -e gpu python slangtrain/export_selftest.py <caldata_test.parquet> [K] [out.bin]
import os, sys
import numpy as np
import pyarrow.parquet as pq

path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SINEW_CALDATA_TEST")
if not path:
    sys.exit("usage: export_selftest.py <caldata_test.parquet> [K] [out.bin]")
K = int(sys.argv[2]) if len(sys.argv) > 2 else 64
out = sys.argv[3] if len(sys.argv) > 3 else "selftest.bin"

t = pq.read_table(path)
md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
S, NIN, NOUT = int(md["S"]), int(md["NIN"]), int(md["NOUT"])
nsens = NOUT // 6
per = NIN // nsens
if per != 15:
    print(f"warning: NIN={NIN} → {per} channels/sensor (the mag net is 15: accel3+mag3+rot9; "
          "wtrained.bin from calibrator-v2 expects NIN=225)", file=sys.stderr)

x = t["x"].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, S, NIN).astype(np.float32)
y = t["y"].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(len(t), -1).astype(np.float32)

# Grav-normalize the accel channel per sensor/frame — train_tic.py accel_xform "grav".
z = x.reshape(x.shape[0], S, nsens, per).copy()
aa = z[..., :3]
z[..., :3] = aa / (np.linalg.norm(aa, axis=-1, keepdims=True) + 1e-6)
x = z.reshape(x.shape)

K = min(K, len(x))
idx = np.sort(np.random.default_rng(0).choice(len(x), K, replace=False))  # random, not first-subject
with open(out, "wb") as f:
    np.array([K, S, NIN, NOUT], np.int32).tofile(f)
    for w in idx:
        x[w].reshape(-1).astype(np.float32).tofile(f)
        y[w].reshape(-1).astype(np.float32).tofile(f)
print(f"wrote {out}: {K} of {len(x)} windows (random seed 0)  "
      f"S={S} NIN={NIN} NOUT={NOUT} ({nsens} sensors × {per} ch)")
