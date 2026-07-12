# SPDX-License-Identifier: MIT
# M7 step 1: 15-sensor calibration training data from real TIC poses (no chumpy).
# Clean bone orientations come from the anny parent-chain global-rotation FK on the
# 15 device→anny joints; per window we inject a random per-sensor offset R_BS and
# emit (corrupted rot input, offset target) pairs — the TIC recipe at 15 sensors,
# orientation-only (accel synthesis is the documented follow-on).
import os
import numpy as np
import torch

TIC_DIR = "D:/TIC Dataset"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caldata.npz")

# anny 24-joint parents; global rot[i] = rot[parent]·local[i].
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
# 15 device nodes → anny joints (the 6 TIC sensors [18,19,4,5,15,0] are a subset).
RBC15 = [0, 1, 2, 4, 5, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21]
NSENS = len(RBC15)
S = 32           # window length
STRIDE = 32
OFFSET_DEG = 30  # max injected per-sensor offset


def axis_angle_to_R(aa):  # (...,3) -> (...,3,3)
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / th
    K = torch.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]; K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    th = th[..., None]
    I = torch.eye(3, dtype=aa.dtype).expand_as(K)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def global_rot(localR):  # (T,24,3,3) -> (T,24,3,3)
    g = [None] * 24
    g[0] = localR[:, 0]
    for j in range(1, 24):
        g[j] = g[PARENTS[j]] @ localR[:, j]
    return torch.stack(g, dim=1)


def r6d(R):  # (...,3,3) -> first two columns flattened (...,6)
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def main():
    torch.manual_seed(0)
    Xs, Ys = [], []
    for sub in ["s1", "s2", "s3", "s4", "s5"]:
        pose = torch.load(f"{TIC_DIR}/{sub}/pose.pt", map_location="cpu").double()  # (T,24,3)
        lr = axis_angle_to_R(pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        gr = global_rot(lr)[:, RBC15]          # (T,15,3,3) clean bone orientation
        gr = gr[::2]                           # downsample to ~30fps
        T = gr.shape[0]
        for st in range(0, T - S, STRIDE):
            win = gr[st:st + S]                # (S,15,3,3)
            # random per-sensor offset R_BS (constant over the window)
            aa = ((torch.rand(NSENS, 3, dtype=torch.float64) - 0.5)
                  * 2 * (OFFSET_DEG / 180 * np.pi))
            Roff = axis_angle_to_R(aa)         # (15,3,3)
            corr = win @ Roff[None]            # measured = clean·R_BS  (S,15,3,3)
            x = corr.reshape(S, NSENS * 9).numpy()          # (S,135)
            # target: global head = identity 6D, local head = offset 6D
            ident = r6d(torch.eye(3).expand(NSENS, 3, 3)).reshape(-1).numpy()  # (90,)
            off6 = r6d(Roff).reshape(-1).numpy()                              # (90,)
            Xs.append(x.astype(np.float32))
            Ys.append(np.concatenate([ident, off6]).astype(np.float32))
    X = np.stack(Xs); Y = np.stack(Ys)
    np.savez(OUT, X=X, Y=Y, S=S, NSENS=NSENS, NIN=NSENS * 9, NOUT=NSENS * 6)
    print(f"caldata: {X.shape[0]} windows  X{X.shape}  Y{Y.shape}  NIN={NSENS*9} NOUT={NSENS*6}")
    # sanity: offsets vary across windows (well-posed target), clean rot is SO(3)
    print(f"offset-6D target std across windows = {Y[:, NSENS*6:].std():.3f} (should be > 0)")


if __name__ == "__main__":
    main()
