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
import glob
import os

import trimesh
from PIL import Image

from vsk_recsys.data.etnf import asset_uuid
from vsk_recsys.encoders.render import RENDER_SIZE, MeshRenderer

MESH_EXTENSIONS = ("glb", "gltf", "obj", "ply")


def discover_meshes(root: str) -> list[str]:
    """Return sorted mesh file paths under ``root`` (recursive), skipping Godot's import cache."""
    if os.path.isdir(root):
        found: list[str] = []
        for ext in MESH_EXTENSIONS:
            found += glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
    else:
        found = glob.glob(root, recursive=True)

    return sorted(p for p in found if ".godot" not in p and os.path.isfile(p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meshes", required=True, help="mesh dir (scanned recursively) or a glob")
    ap.add_argument("--out-dir", default="lake/renders")
    ap.add_argument("--size", type=int, default=RENDER_SIZE)
    args = ap.parse_args()

    paths = discover_meshes(args.meshes)
    os.makedirs(args.out_dir, exist_ok=True)

    # One viser server + one headless browser render the whole corpus; the mesh is swapped per asset.
    rendered, skipped = 0, []
    with MeshRenderer(size=args.size) as renderer:
        for path in paths:
            try:
                scene = trimesh.load(path)
                mesh = scene.to_mesh() if hasattr(scene, "to_mesh") else scene
                rgb = renderer.render(mesh)
            except Exception as e:  # a bad mesh must not abort the corpus render
                skipped.append((path, str(e)))
                print(f"  SKIP {path}: {e}", flush=True)
                continue

            out = os.path.join(args.out_dir, f"{asset_uuid(path)}.png")
            Image.fromarray(rgb).save(out)
            rendered += 1
            print(f"  {path} -> {out}", flush=True)

    print(f"rendered {rendered}/{len(paths)} meshes -> {args.out_dir} ({len(skipped)} skipped)")
    for path, why in skipped:
        print(f"  skipped: {path} ({why})", flush=True)


if __name__ == "__main__":
    main()
