# SPDX-License-Identifier: MIT
# Convert the AddBiomechanics B3D corpus to a queryable parquet of CLEAN motion + demographics,
# streaming each B3D out of the zip by offset so there is no full-extraction intermediate (the zip is
# ~389 GiB; only one B3D is on disk at a time).  Drops geometry meshes / GRF / OpenSim model — keeps
# the 15-bone ANNY-world orientation (6D, lossless) + position per frame (accel is derivable) and the
# demographics.  Reuses b3d_to_caldata's marker-Kabsch pipeline so the motion matches the calibration
# extraction exactly (just without the per-window mount/drift corruption).
#
# Runs in the WSL addb pixi env (nimblephysics):
#   python b3d_to_motion.py --index <zip> <index.json>                      # 1. one local-header walk
#   python b3d_to_motion.py <zip> <index.json> <shard> <nshards> <out.pq>   # 2. parallel worker shard
# Central-directory offsets are unreliable in this zip (zipfile.open fails), so the index records the
# real local-header data offsets found by scanning — like salvage_partial_zip.
import sys
import os
import json
import struct
import zlib
import urllib.request
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import nimblephysics as nimble
import b3d_to_caldata as B

LFH = b"PK\x03\x04"
SRC_URL = "http://archive.simtk.org/addbiomechanics/addbiomechanics.zip"


def build_broken_index(zip_path, out):
    # The entries whose LOCAL data is corrupt (aria2 download damage): central metadata is correct
    # (server has a valid header at the central offset, verified by Range GET), so record the central
    # header offset + compress_size for an offset re-download.
    import zipfile
    z = zipfile.ZipFile(zip_path)
    broken = []
    with open(zip_path, "rb") as f:
        for info in z.infolist():
            name = info.filename
            if not (name.lower().endswith(".b3d") and "/With_Arm/" in name) or name.endswith("/"):
                continue
            f.seek(info.header_offset)
            if f.read(4) != LFH:                       # local header damaged -> needs re-download
                broken.append({"name": name, "off_hdr": info.header_offset,
                               "csize": info.compress_size, "method": info.compress_type})
    json.dump(broken, open(out, "w"))
    print(f"{len(broken)} broken With_Arm b3d to re-download by offset")


def _range(url, start, length):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    return urllib.request.urlopen(req, timeout=120).read()


def extract_one_http(url, e, tmp):  # re-download one entry's bytes by offset from the source server
    hdr = _range(url, e["off_hdr"], 30)
    ver, flags, method, mt, md, crc, csize, usize, nlen, elen = struct.unpack("<HHHHHIIIHH", hdr[4:30])
    data = _range(url, e["off_hdr"] + 30 + nlen + elen, e["csize"])
    raw = data if e["method"] == 0 else zlib.decompress(data, -15)
    with open(tmp, "wb") as o:
        o.write(raw)


def build_index(zip_path, out):
    # Index via the central directory, keeping only entries whose local header is actually present at
    # the claimed offset (the magic check).  ~105 deep entries in this archive have corrupt central
    # metadata (offset AND compress_size both wrong) and can't be reached without a re-download from the
    # source; they're skipped.  The 358 that pass are reliably extractable.
    import zipfile
    z = zipfile.ZipFile(zip_path)
    idx, skipped = [], 0
    with open(zip_path, "rb") as f:
        for info in z.infolist():
            name = info.filename
            if not (name.lower().endswith(".b3d") and "/With_Arm/" in name) or name.endswith("/"):
                continue
            f.seek(info.header_offset)
            if f.read(4) != LFH:
                skipped += 1
                continue
            ver, flags, method, mt, md, crc, csize, usize, nlen, elen = struct.unpack("<HHHHHIIIHH", f.read(26))
            idx.append({"name": name, "off": info.header_offset + 30 + nlen + elen,
                        "csize": info.compress_size, "method": info.compress_type})
    json.dump(idx, open(out, "w"))
    print(f"indexed {len(idx)} With_Arm b3d entries; {skipped} skipped (corrupt central metadata)")


def extract_one(zip_path, e, tmp):  # seek to the real offset, decompress this one b3d to tmp
    with open(zip_path, "rb") as f:
        f.seek(e["off"])
        data = f.read(e["csize"])
    raw = data if e["method"] == 0 else zlib.decompress(data, -15)
    with open(tmp, "wb") as o:
        o.write(raw)


def r6d_frames(Rw):  # (F,NSENS,3,3) -> (F, NSENS*6): per bone [col0, col1] (matches b3d_to_caldata.r6d)
    return np.concatenate([Rw[..., :, 0], Rw[..., :, 1]], axis=-1).reshape(len(Rw), -1)


def subject_motion(subj):  # clean per-trial (Rw, pos, dt) — replicates b3d_to_caldata's frame loop
    body, by_body, rest, restp = B.body_setup(subj)
    if any(b is None for b in body):
        return None
    arcsf = B.anny_rcsf_R()
    R_align = [rest[body[n]].T @ B.R_W.T @ arcsf[n] for n in range(B.NSENS)]
    seg = B.seg_lengths(body, restp)
    trials = []
    for t in range(subj.getNumTrials()):
        n, dt = subj.getTrialLength(t), subj.getTrialTimestep(t)
        if n < 3:
            continue
        frames = subj.readFrames(t, 0, n, includeSensorData=False, includeProcessingPasses=False)
        Rw = np.zeros((len(frames), B.NSENS, 3, 3))
        pos = np.zeros((len(frames), B.NSENS, 3))
        prev, lastR = None, [None] * B.NSENS
        for fi, fr in enumerate(frames):
            obs = {m[0]: np.array(m[1], dtype=np.float64) for m in fr.markerObservations}
            Rmark = [None] * B.NSENS
            for s in range(B.NSENS):
                ms = [(mn, o) for mn, o in by_body.get(body[s], []) if mn in obs]
                if len(ms) >= B.MIN_MARKERS:
                    R, tt = B.kabsch(np.array([o for _, o in ms]), np.array([obs[mn] for mn, _ in ms]))
                    Rmark[s] = B.R_W @ R @ R_align[s]
                    pos[fi, s] = B.R_W @ tt
                elif ms:
                    pos[fi, s] = B.R_W @ obs[ms[0][0]]
                else:
                    pos[fi, s] = prev[s] if prev is not None else 0.0
            for s in range(B.NSENS):
                if Rmark[s] is not None:
                    Rw[fi, s] = Rmark[s]
                elif s in B.SWING:
                    a, b = B.LONG_AXIS[s]
                    Rw[fi, s] = B.build_frame(pos[fi, b] - pos[fi, a])
                elif lastR[s] is not None:
                    Rw[fi, s] = lastR[s]
                else:
                    Rw[fi, s] = np.eye(3)
                lastR[s] = Rw[fi, s]
            prev = pos[fi].copy()
        trials.append((t, dt, Rw, pos))
    return seg, trials


def _process_subject(zip_path, e, via_http, tmp_b3d, out_part):
    # Runs in a forked child so a nimblephysics SEGFAULT kills only this process, not the shard.
    name = e["name"]
    sub = os.path.splitext(os.path.basename(name))[0]
    study = name.split("/With_Arm/")[1].split("/")[0].replace("_Formatted_With_Arm", "")
    split = name.split("/", 1)[0]
    extract_one_http(SRC_URL, e, tmp_b3d) if via_http else extract_one(zip_path, e, tmp_b3d)
    subj = nimble.biomechanics.SubjectOnDisk(tmp_b3d)
    r = subject_motion(subj)
    if r is None:
        return
    seg, trials = r
    sex = subj.getBiologicalSex()
    hm, mk = round(subj.getHeightM(), 3), round(subj.getMassKg(), 1)
    rows = [{"subject": sub, "study": study, "split": split, "trial": t, "n_frames": len(Rw),
             "dt": float(dt), "sex": sex, "height_m": hm, "mass_kg": mk,
             "seglen_m": [float(x) for x in seg],
             "rot6d": r6d_frames(Rw).reshape(-1).astype(np.float32).tolist(),   # F*NSENS*6
             "pos": pos.reshape(-1).astype(np.float32).tolist()}                # F*NSENS*3
            for (t, dt, Rw, pos) in trials]
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), out_part, compression="zstd")


def worker(zip_path, index_json, shard, nshards, outdir, via_http=False):
    idx = json.load(open(index_json))
    mine = [e for i, e in enumerate(idx) if i % nshards == shard]
    os.makedirs(outdir, exist_ok=True)
    for k, e in enumerate(mine):
        name = e["name"]
        sub = os.path.splitext(os.path.basename(name))[0]
        study = name.split("/With_Arm/")[1].split("/")[0].replace("_Formatted_With_Arm", "")
        out_part = os.path.join(outdir, f"{study}__{sub}.parquet")
        if os.path.exists(out_part):                       # resume: skip already-converted subjects
            continue
        tmp_b3d = f"/tmp/b3dmotion_{shard}.b3d"
        pid = os.fork()
        if pid == 0:                                       # child — isolated from segfaults
            try:
                _process_subject(zip_path, e, via_http, tmp_b3d, out_part)
            except Exception as ex:
                print(f"FAIL {sub}: {str(ex)[:60]}", flush=True)
            os._exit(0)
        _, status = os.waitpid(pid, 0)                     # parent waits; survives a child core-dump
        if not os.path.exists(out_part) and os.WIFSIGNALED(status):
            print(f"SEGFAULT skip {study}/{sub} (sig {os.WTERMSIG(status)})", flush=True)
        if os.path.exists(tmp_b3d):
            os.remove(tmp_b3d)
        if (k + 1) % 5 == 0:
            print(f"  shard {shard}: {k + 1}/{len(mine)} subjects", flush=True)
    print(f"shard {shard}: done {len(mine)} subjects -> {outdir}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--index":
        build_index(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "--broken-index":
        build_broken_index(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "--http":   # recover corrupt entries by offset re-download: <broken.json> <shard> <n> <out.pq>
        worker(None, sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], via_http=True)
    else:
        worker(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
