# SPDX-License-Identifier: MIT
# Probe per-subject phenotype inputs (seglen + demographics) from extracted B3D headers (train
# split only).  Feeds pheno_equity (equity weighting) and stage_b_fit (11-dim ANNY phenotype).
# Runs in the WSL addb pixi env (nimblephysics); reads the headers salvage_partial_zip extracts.
#   pixi run python probe_pheno.py [hdr_glob_dir] [out.json]
import sys, os, glob, json
from collections import Counter
import nimblephysics as nimble
sys.path.insert(0, "/mnt/e/sinew-moved/slangtrain/addb")
import b3d_to_caldata as B

HDR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/e/tmp/addb_hdr/train"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/e/tmp/pheno_probe.json"
files = sorted(glob.glob(os.path.join(HDR, "**", "*.b3d"), recursive=True))
out = []
for p in files:
    name = os.path.splitext(os.path.basename(p))[0]
    dataset = next((c.replace("_Formatted_With_Arm", "") for c in p.split("/") if "_Formatted" in c), "?")
    try:
        subj = nimble.biomechanics.SubjectOnDisk(p)
        body, by_body, rest, restp = B.body_setup(subj)
        if any(b is None for b in body):
            print("skip(body none)", name)
            continue
        seg = B.seg_lengths(body, restp)
        out.append(dict(subject=name, dataset=dataset, sex=subj.getBiologicalSex(),
                        height=round(subj.getHeightM(), 3), mass=round(subj.getMassKg(), 1),
                        seglen=[round(float(x), 4) for x in seg]))
    except Exception as e:
        print("FAIL", name, str(e)[:70])
json.dump(out, open(OUT, "w"))
print("probed", len(out), "subjects   sex:", dict(Counter(o["sex"] for o in out)))
