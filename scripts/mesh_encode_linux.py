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

from vsk_recsys.data.etnf import asset_uuid
from vsk_recsys.encoders.mesh import encode_to_slat, load_shape_encoder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meshes", required=True, help="glob of mesh files (.glb/.obj/.ply)")
    ap.add_argument("--weights", required=True, help="prefix to shape_enc.{json,safetensors}")
    ap.add_argument("--trellis2", required=True, help="path to the trellis2 repo")
    ap.add_argument("--out", default="lake/asset_mesh_embedding.parquet")
    ap.add_argument("--grid-size", type=int, default=64)
    args = ap.parse_args()

    enc = load_shape_encoder(args.weights, args.trellis2)
    rows = []
    for path in sorted(glob.glob(args.meshes)):
        m = trimesh.load(path, force="mesh")
        v = torch.as_tensor(m.vertices, dtype=torch.float32)
        f = torch.as_tensor(m.faces, dtype=torch.int32)
        feats, coords = encode_to_slat(v, f, enc, grid_size=args.grid_size)  # STRUCTURED — no pooling
        rows.append(
            {
                "asset_uuid": str(asset_uuid(path)),
                "natural_key": path,
                "n_tokens": int(feats.shape[0]),
                "slat_feats": feats.numpy().tolist(),   # (N, C) structured SLAT tokens
                "slat_coords": coords.numpy().tolist(),  # (N, 3) voxel coords
            }
        )
        print(f"  {path} -> SLAT {tuple(feats.shape)} tokens (structured, no pooling)", flush=True)

    # ETNF 1:1 derived-extension relation `asset_mesh_slat` keyed by asset_uuid; downstream FSQ
    # quantizes each SLAT token into mesh semantic codes (no mean/average).
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(f"wrote {len(rows)} assets -> {args.out}")


if __name__ == "__main__":
    main()
