# SPDX-License-Identifier: MIT
# Equity sampler: reads the Parquet calibration windows and writes the raw caldata.bin
# (X[N*S*NIN] then Y[N*2*NOUT], float32) that caltrain.exe consumes.
#
# Default: balance windows across recorded_sex.  With --weights <pheno_weights.json> (from
# pheno_equity.py): draw windows in proportion to each subject's best-effort Google-style equity
# weight, so training hits the phenotype floors (sex 50/50, the hardest available BMI band
# maximised) rather than the data's convenience distribution.  Subjects absent from the weights
# file (e.g. the unrepresentable obese band) contribute no windows — the residual equity gap.
#   pixi run python sample_caldata.py <in.parquet> <out caldata.bin> [--weights w.json] [--n N]
import sys, json, collections
import numpy as np
import pyarrow.parquet as pq

args = sys.argv[1:]
wjson = args[args.index("--weights") + 1] if "--weights" in args else None
n_out = int(args[args.index("--n") + 1]) if "--n" in args else None
pos = [a for a in args if not a.startswith("--") and a not in {wjson, str(n_out)}]

t = pq.read_table(pos[0])
md = t.schema.metadata
S, NIN, NOUT = int(md[b"S"]), int(md[b"NIN"]), int(md[b"NOUT"])
# small columns as lists; the big x/y as flat Arrow buffers (no per-element Python iteration)
subj = t["subject"].to_pylist()
rsex = t["recorded_sex"].to_pylist()
Xall = t["x"].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(len(t), -1)
Yall = t["y"].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(len(t), -1)
rng = np.random.default_rng(0)

if wjson:  # equity weighting: window prob proportional to its subject's equity weight
    wmap = {s["subject"]: s["weight"] for s in json.load(open(wjson))}
    w = np.array([wmap.get(sub, 0.0) for sub in subj], dtype=np.float64)
    if w.sum() == 0:
        sys.exit("no window's subject is in the weights file — extract those subjects' windows first")
    N = n_out or int((w > 0).sum())
    sel = list(rng.choice(len(w), size=N, replace=True, p=w / w.sum()))
    used = collections.Counter(subj[i] for i in sel)
    print(f"equity sample: {N} windows over {len(used)} subjects (weighted draw)")
else:  # default: balance by recorded_sex
    by_sex = collections.defaultdict(list)
    for i, s in enumerate(rsex):
        by_sex[s].append(i)
    m = min(len(v) for v in by_sex.values())
    sel = []
    for v in by_sex.values():
        sel += list(rng.choice(v, m, replace=False))
    rng.shuffle(sel)
    print(f"sex-balanced sample: {len(sel)} windows, {m}/group across "
          f"{dict((k, len(v)) for k, v in by_sex.items())}")

X = Xall[sel].astype(np.float32)  # (N, S*NIN)
Y = Yall[sel].astype(np.float32)  # (N, 2*NOUT)
with open(pos[1], "wb") as f:
    X.tofile(f)
    Y.tofile(f)
print(f"wrote {pos[1]}: {len(sel)} windows  S={S} NIN={NIN} NOUT={NOUT}")
