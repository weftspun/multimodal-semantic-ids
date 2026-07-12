"""Offline mesh-embedding batch job (WSL2 Linux `mesh` env) — planner action a_p2_mesh_cuda_linux.

mesh files -> O-Voxel -> TRELLIS.2 SLAT shape-VAE -> pooled vector -> `asset_mesh_embedding` ETNF
parquet relation (keyed by content-addressable `asset_uuid`). VERIFIED on WSL2 Fedora
(torch 2.6/cu124, RTX 4090) with FlexGEMM + o-voxel built from source (FlexGEMM-source ships the
triton GEMM kernels the Windows prebuilt wheel lacked).

Run (in the WSL2 `~/vsk-mesh` pixi env):
    pixi run python scripts/mesh_encode_linux.py \
        --meshes '/path/to/*.glb' \
        --weights /mnt/c/.../.mesh_trial/ckpts/shape_enc \
        --trellis2 ~/vsk-mesh/TRELLIS2 \
        --out lake/asset_mesh_embedding.parquet
"""

from __future__ import annotations

import argparse
import glob

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import trimesh

import os

from vsk_recsys.data.etnf import asset_uuid
from vsk_recsys.encoders.mesh import (
    encode_texture_to_slat,
    encode_to_slat,
    load_shape_encoder,
    load_texture_encoder,
)


def _row(path, feats, coords):
    return {
        "asset_uuid": str(asset_uuid(path)),
        "natural_key": path,
        "n_tokens": int(feats.shape[0]),
        "slat_feats": feats.numpy().tolist(),  # (N, 32) structured tokens — NO pooling
        "slat_coords": coords.numpy().tolist(),  # (N, 3) voxel coords, Hilbert-ordered
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meshes", required=True, help="glob of mesh files (.glb/.obj/.ply)")
    ap.add_argument("--shape-weights", required=True, help="prefix to shape_enc.{json,safetensors}")
    ap.add_argument("--tex-weights", required=True, help="prefix to tex_enc.{json,safetensors}")
    ap.add_argument("--trellis2", required=True, help="path to the trellis2 repo")
    ap.add_argument("--out-dir", default="lake")
    ap.add_argument("--grid-size", type=int, default=64)
    args = ap.parse_args()

    import os

    if os.path.isdir(args.meshes):  # a directory → scan for mesh files recursively
        paths = []
        for ext in ("glb", "gltf", "obj", "ply"):
            paths += glob.glob(os.path.join(args.meshes, "**", f"*.{ext}"), recursive=True)
    else:  # a glob pattern (or single file)
        paths = glob.glob(args.meshes, recursive=True)
    paths = sorted(p for p in paths if ".godot" not in p and os.path.isfile(p))  # skip import cache/dirs

    senc = load_shape_encoder(args.shape_weights, args.trellis2)
    tenc = load_texture_encoder(args.tex_weights, args.trellis2)
    shape_rows, tex_rows = [], []
    skipped = []
    for path in paths:
        try:
            asset = trimesh.load(path)  # Scene with PBR materials (for texture)
            mesh = asset.to_mesh() if hasattr(asset, "to_mesh") else asset
            v = torch.as_tensor(mesh.vertices, dtype=torch.float32)
            f = torch.as_tensor(mesh.faces, dtype=torch.int32)
            sf, sc = encode_to_slat(v, f, senc, grid_size=args.grid_size)  # geometry (always)
            shape_rows.append(_row(path, sf, sc))
        except Exception as e:  # a bad mesh must not abort the whole corpus run
            skipped.append((path, f"shape: {e}"))
            print(f"  SKIP {path} (shape failed): {e}", flush=True)
            continue
        try:
            tf, tc = encode_texture_to_slat(asset, tenc, grid_size=args.grid_size)  # PBR/appearance
            tex_rows.append(_row(path, tf, tc))
            print(f"  {path} -> shape {tuple(sf.shape)} + texture {tuple(tf.shape)}", flush=True)
        except Exception as e:  # geometry-only meshes / texture-decode issues keep their shape tokens
            print(f"  {path} -> shape {tuple(sf.shape)} + texture SKIPPED: {e}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    # Two ETNF 1:1 derived-extension relations keyed by asset_uuid (full mesh = shape ⊕ texture).
    pq.write_table(pa.Table.from_pylist(shape_rows), os.path.join(args.out_dir, "asset_mesh_shape_slat.parquet"))
    pq.write_table(pa.Table.from_pylist(tex_rows), os.path.join(args.out_dir, "asset_mesh_texture_slat.parquet"))
    print(f"wrote {len(shape_rows)} shape + {len(tex_rows)} texture assets -> "
          f"{args.out_dir}/asset_mesh_{{shape,texture}}_slat.parquet "
          f"({len(paths)} inputs, {len(skipped)} skipped)")
    for p, why in skipped:
        print(f"  skipped: {p} ({why})", flush=True)


if __name__ == "__main__":
    main()
