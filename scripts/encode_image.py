"""Encode mesh renders -> `asset_image_embedding.parquet` (ETNF 1:1) via Qwen3-VL-Embedding.

Reads ``lake/renders/<asset_uuid>.png`` (produced by scripts/render_meshes.py), embeds each render with
the unified Qwen3-VL-Embedding-2B image path, and writes the derived-extension relation keyed by
``asset_uuid`` — the SAME space as ``asset_text_embedding`` (text and image are directly comparable).
Runs in the default cu128 pixi env. See decisions/20260712-multimodal-foss-encoder-stack.md.
"""

from __future__ import annotations

import argparse
import glob
import os

from PIL import Image

from vsk_recsys.data.lake import write_relation
from vsk_recsys.encoders.multimodal import DEFAULT_MODEL, Qwen3VLEncoder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--renders", default="lake/renders")
    ap.add_argument("--out-dir", default="lake")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    # Each render's filename stem IS the asset_uuid (content-addressed to match the mesh SLAT relations).
    paths = sorted(glob.glob(os.path.join(args.renders, "*.png")))
    if not paths:
        raise SystemExit(f"no renders found in {args.renders} — run scripts/render_meshes.py first")

    uuids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    images = [Image.open(p).convert("RGB") for p in paths]

    print(f"loading {args.model} ...", flush=True)
    enc = Qwen3VLEncoder(args.model)
    vecs = enc.encode_images(images, batch_size=args.batch_size)

    rows = [{"asset_uuid": uid, "image_embedding": vecs[i].tolist()} for i, uid in enumerate(uuids)]
    write_relation("asset_image_embedding", rows, args.out_dir, pk=["asset_uuid"])
    print(f"wrote {len(rows)} rows -> {args.out_dir}/asset_image_embedding.parquet (dim={enc.dim})")


if __name__ == "__main__":
    main()
