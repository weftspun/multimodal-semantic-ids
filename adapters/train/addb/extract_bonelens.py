# SPDX-License-Identifier: MIT
# Per-subject OpenSim jointCenter bone lengths -> the inverse-ANNY fit target (fit_anny.py).
# The B3D processing pass holds the 20 fitted joint world positions (a GET, no segfault-prone
# setPositions FK); their pose-invariant joint-pair lengths characterise the subject's skeleton.
# Batched over a directory of B3Ds, fork-isolated so a nimblephysics segfault skips one subject.
# Output: one parquet per subject in <outdir>; merge into bonelens.parquet for the batch fit.
#
#   pixi run --manifest-path ~/addb-extract/pixi.toml python extract_bonelens.py <b3d_dir> <outdir>
import sys
import os
import glob
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import nimblephysics as nimble

# OpenSim joint pair per ANNY-fittable segment (fit_anny.PAIRS maps these to ANNY joints)
PAIRS = {
    "thigh_r": ("hip_r", "walker_knee_r"), "shin_r": ("walker_knee_r", "ankle_r"),
    "foot_r": ("ankle_r", "mtp_r"), "thigh_l": ("hip_l", "walker_knee_l"),
    "shin_l": ("walker_knee_l", "ankle_l"), "foot_l": ("ankle_l", "mtp_l"),
    "uparm_r": ("acromial_r", "elbow_r"), "forearm_r": ("elbow_r", "radius_hand_r"),
    "uparm_l": ("acromial_l", "elbow_l"), "forearm_l": ("elbow_l", "radius_hand_l"),
    "torso": ("ground_pelvis", "back"), "shoulder_w": ("acromial_r", "acromial_l"),
    "hip_w": ("hip_r", "hip_l"),
}


def one(path, study, split, out):
    s = nimble.biomechanics.SubjectOnDisk(path)
    sk = s.readSkel(0, ignoreGeometry=True)
    ji = {sk.getJoint(i).getName(): i for i in range(sk.getNumJoints())}
    n = min(s.getTrialLength(0), 200)
    frames = s.readFrames(0, 0, n, includeProcessingPasses=True)
    jc = np.array([np.array(f.processingPasses[0].jointCenters).reshape(-1, 3) for f in frames])  # (n,20,3)
    length = lambda a, b: float(np.median(np.linalg.norm(jc[:, ji[a]] - jc[:, ji[b]], axis=1)))
    row = {k: length(a, b) for k, (a, b) in PAIRS.items()}
    row.update(subject=os.path.splitext(os.path.basename(path))[0], study=study, split=split,
               height_m=round(s.getHeightM(), 3), mass_kg=round(s.getMassKg(), 1), sex=s.getBiologicalSex())
    pq.write_table(pa.Table.from_pylist([row]), out, compression="zstd")


def worker(b3d_dir, outdir):
    os.makedirs(outdir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(b3d_dir, "**", "*.b3d"), recursive=True))
    print(f"{len(paths)} B3D files under {b3d_dir}", flush=True)
    for k, p in enumerate(paths):
        sub = os.path.splitext(os.path.basename(p))[0]
        study = (p.split("/With_Arm/")[1].split("/")[0].replace("_Formatted_With_Arm", "")
                 if "/With_Arm/" in p else "unknown")
        split = "train" if "/train/" in p else ("test" if "/test/" in p else "?")
        out = os.path.join(outdir, f"{study}__{sub}.parquet")
        if os.path.exists(out):                                  # resume
            continue
        pid = os.fork()
        if pid == 0:                                             # child — isolated from segfaults
            try:
                one(p, study, split, out)
            except Exception as e:
                print(f"FAIL {sub}: {str(e)[:60]}", flush=True)
            os._exit(0)
        _, st = os.waitpid(pid, 0)
        if not os.path.exists(out) and os.WIFSIGNALED(st):
            print(f"SEGFAULT skip {study}/{sub} (sig {os.WTERMSIG(st)})", flush=True)
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(paths)} subjects", flush=True)
    print(f"done {len(paths)} subjects -> {outdir}", flush=True)


if __name__ == "__main__":
    worker(sys.argv[1], sys.argv[2])
