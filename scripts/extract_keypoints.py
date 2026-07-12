"""Extract COCO-17 person keypoints from the asset renders (planner action a_p2_enc_phenotype, stage 1).

Runs rf-detr ``RFDETRKeypointPreview`` (Apache-2.0) over ``lake/renders/<asset_uuid>.png`` and writes the
detected humanoid subset to ``asset_keypoints_coco17.parquet`` (ETNF 1:1, keyed by ``asset_uuid``). These
raw COCO-17 keypoints are the input the ``somaxc`` (sinew-mocap/solve) retarget maps onto the ANNY skeleton.

Runs in the isolated ``rfdetr`` pixi env (its CV deps break the default env):
    pixi run -e rfdetr python scripts/extract_keypoints.py
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from rfdetr import RFDETRKeypointPreview

COCO17_KEYPOINTS = 17
PERSON_THRESHOLD = 0.3  # min detection score to accept a render as a humanoid


def _best_person(keypoints) -> tuple[np.ndarray, np.ndarray] | None:
    """Pick the highest-confidence person instance; return (xy [17,2], confidence [17]) or None."""
    xy = np.asarray(getattr(keypoints, "xy", np.empty((0, COCO17_KEYPOINTS, 2))))
    conf = np.asarray(getattr(keypoints, "confidence", np.empty((0, COCO17_KEYPOINTS))))
    if xy.shape[0] == 0:
        return None

    best = int(conf.mean(axis=1).argmax())
    return xy[best], conf[best]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--renders", default="lake/renders")
    ap.add_argument("--out-dir", default="lake")
    ap.add_argument("--threshold", type=float, default=PERSON_THRESHOLD)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.renders, "*.png")))
    if not paths:
        raise SystemExit(f"no renders in {args.renders} — run scripts/render_meshes.py first")

    model = RFDETRKeypointPreview()

    rows, no_person = [], []
    for path in paths:
        asset_uuid = os.path.splitext(os.path.basename(path))[0]
        image = Image.open(path).convert("RGB")

        # Only humanoid assets yield a person detection; the rest have no phenotype.
        person = _best_person(model.predict(image, threshold=args.threshold))
        if person is None:
            no_person.append(asset_uuid)
            continue

        xy, conf = person
        rows.append({
            "asset_uuid": asset_uuid,
            "keypoints_xy": xy.astype(float).tolist(),      # (17, 2) image-space COCO-17
            "keypoints_conf": conf.astype(float).tolist(),  # (17,) per-joint confidence
        })
        print(f"  {asset_uuid}: person, mean conf {float(conf.mean()):.2f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "asset_keypoints_coco17.parquet")
    pq.write_table(pa.Table.from_pylist(rows), out)

    print(f"wrote {len(rows)} humanoid keypoint rows -> {out} "
          f"({len(no_person)}/{len(paths)} renders had no person)")


if __name__ == "__main__":
    main()
