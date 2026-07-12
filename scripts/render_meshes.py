"""Render each corpus mesh to a PNG via the headless viser pipeline (WSL2 `mesh` env).

mesh files -> viser/three.js/SwiftShader render -> ``lake/renders/<asset_uuid>.png``, keyed by the SAME
content-addressed ``asset_uuid`` as the mesh SLAT relations. The renders are then embedded by
``scripts/encode_image.py`` (Qwen3-VL image path). See vsk_recsys/encoders/render.py.

Run (WSL2 `~/vsk-mesh` pixi env):
    PYTHONPATH=<repo> pixi run python scripts/render_meshes.py \
        --meshes <repo>/data/godot-demo-projects --out-dir <repo>/lake/renders
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import trimesh
from PIL import Image

from vsk_recsys.data.godot import discover_mesh_files, mesh_asset_uuid
from vsk_recsys.encoders.render import RENDER_SIZE, MeshRenderer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meshes", required=True, help="corpus root dir (scanned recursively)")
    ap.add_argument("--out-dir", default="lake/renders")
    ap.add_argument("--size", type=int, default=RENDER_SIZE)
    args = ap.parse_args()

    # --meshes is the corpus root; the render filename is the stable mesh asset_uuid (matches SLAT keys).
    root = Path(args.meshes)
    paths = discover_mesh_files(root)
    os.makedirs(args.out_dir, exist_ok=True)

    # One viser server + one headless browser render the whole corpus; the mesh is swapped per asset.
    rendered, skipped = 0, []
    with MeshRenderer(size=args.size) as renderer:
        for path in paths:
            rel = path.relative_to(root).as_posix()
            try:
                scene = trimesh.load(str(path))
                mesh = scene.to_mesh() if hasattr(scene, "to_mesh") else scene
                rgb = renderer.render(mesh)
            except Exception as e:  # a bad mesh must not abort the corpus render
                skipped.append((str(path), str(e)))
                print(f"  SKIP {path}: {e}", flush=True)
                continue

            out = os.path.join(args.out_dir, f"{mesh_asset_uuid(rel)}.png")
            Image.fromarray(rgb).save(out)
            rendered += 1
            print(f"  {path} -> {out}", flush=True)

    print(f"rendered {rendered}/{len(paths)} meshes -> {args.out_dir} ({len(skipped)} skipped)")
    for path, why in skipped:
        print(f"  skipped: {path} ({why})", flush=True)


if __name__ == "__main__":
    main()
