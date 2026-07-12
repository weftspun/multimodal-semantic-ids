"""`nadizik/synthetic-human-expressions-poses-3d` (CC-BY-4.0) → ETNF parquet (a_p2_enc_phenotype data).

The FOSS humanoid dataset from the datasets MADR — synthetic human expression renders (rf-detr keypoints
run on these, NOT the human-less Godot proxy). Layout inside ``3D_DATASET_sorted.zip``:
``sorted/<bucket>/`` (bucket ∈ low_quality|mismatch|without_face|with_face), each view is a render
``view_{n}.jpg`` + description ``view_{n:06d}.txt`` + a per-bucket ``dataset_log.json`` (keyed
``view_{n}.jpg``: motion, frame, intensity, breath, camera, FACS ``properties``, timestamp).

Writes two ETNF relations (kept separate from the Godot ``assets`` — a distinct asset population):
  ``assets_poses``     : entity rows (same schema as ``assets``, unionable), scene_text = the description
  ``asset_pose_meta``  : 1:1 derived extension keyed by asset_uuid (bucket, image ref, motion, camera, FACS)
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import PurePosixPath

from vsk_recsys.data.etnf import asset_uuid
from vsk_recsys.data.lake import check_fk, write_relation

DATASET_ROOT = "sorted"       # top dir inside the zip
POSE_KIND = "human_pose_render"


def _view_index(image_name: str) -> int:
    """``view_7.jpg`` -> 7 (the dataset_log keys are the unpadded image names)."""
    return int(PurePosixPath(image_name).stem.split("_")[1])


def _bucket_of(zip_path_in_archive: str) -> str:
    """``sorted/with_face/dataset_log.json`` -> ``with_face``."""
    return zip_path_in_archive.split("/")[1]


def _camera_columns(camera: dict) -> dict:
    """Flatten the consistent 5-field camera block into typed columns (facts vary; camera does not)."""
    return {
        "cam_x_translate": camera.get("x_translate"),
        "cam_y_translate": camera.get("y_translate"),
        "cam_z_translate": camera.get("z_translate"),
        "cam_y_rotation": camera.get("y_rotation"),
        "focal_length": camera.get("focal_length"),
    }


def ingest_poses(zip_path: str, out_dir: str = "lake") -> dict:
    """Read the dataset zip → ``assets_poses`` + ``asset_pose_meta`` parquet with PK/FK integrity."""
    assets, meta = [], []

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        logs = [n for n in names if n.endswith("dataset_log.json")]
        text_files = {n for n in names if n.endswith(".txt")}

        for log_name in logs:
            bucket = _bucket_of(log_name)
            entries = json.loads(archive.read(log_name).decode("utf-8", "replace"))

            for image_name, info in entries.items():
                n = _view_index(image_name)
                auid = str(asset_uuid(f"nadizik:{bucket}/view_{n}"))

                # The per-view description lives in a zero-padded sibling .txt (same index).
                text_name = f"{DATASET_ROOT}/{bucket}/view_{n:06d}.txt"
                scene_text = (archive.read(text_name).decode("utf-8", "replace")
                              if text_name in text_files else None)

                assets.append({
                    "asset_uuid": auid, "natural_key": f"nadizik:{bucket}/view_{n}", "kind": POSE_KIND,
                    "display_name": f"{info.get('motion')}#{info.get('frame')}",
                    "slug": f"nadizik/{bucket}/view_{n}", "scene_text": scene_text,
                })

                meta.append({
                    "asset_uuid": auid,
                    "quality_bucket": bucket,
                    "image_ref": f"{DATASET_ROOT}/{bucket}/{image_name}",  # zip-internal path for rf-detr
                    "motion": info.get("motion"),
                    "frame": info.get("frame"),
                    "total_frames": info.get("total_frames"),
                    "intensity": info.get("intensity"),
                    "breath_value": info.get("breath_value"),
                    "breath_mode": info.get("breath_mode"),
                    **_camera_columns(info.get("camera") or {}),
                    "facs_json": json.dumps(info.get("properties") or {}),  # variable FACS controls
                    "timestamp": info.get("timestamp"),
                })

    write_relation("assets_poses", assets, out_dir, pk=["asset_uuid"])
    write_relation("asset_pose_meta", meta, out_dir, pk=["asset_uuid"])
    check_fk(meta, "asset_uuid", assets, "asset_uuid")

    return {"assets_poses": len(assets), "asset_pose_meta": len(meta)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default="data/nadizik-poses/3D_DATASET_sorted.zip")
    ap.add_argument("--out-dir", default="lake")
    args = ap.parse_args()
    print("ingested:", ingest_poses(args.zip, args.out_dir), "->", args.out_dir)


if __name__ == "__main__":
    main()
