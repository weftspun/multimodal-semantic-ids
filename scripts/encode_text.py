"""Encode asset scene-text → `asset_text_embedding.parquet` (ETNF 1:1) via Qwen3-VL-Embedding.

Reads `lake/assets.parquet`, embeds `scene_text` with the unified Qwen3-VL-Embedding-2B (sentence-
transformers), writes the ETNF derived-extension relation keyed by `asset_uuid`. Runs in the default
cu128 pixi env. See decisions/20260712-multimodal-foss-encoder-stack.md + ...-parquet-feature-store-etnf.md.
"""

from __future__ import annotations

import argparse

import pyarrow.parquet as pq

from vsk_recsys.data.lake import write_relation
from vsk_recsys.encoders.multimodal import DEFAULT_MODEL, Qwen3VLEncoder

DOC_PROMPT = "Represent this Godot game scene for retrieval of related scenes."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", default="lake/assets.parquet")
    ap.add_argument("--out-dir", default="lake")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    rows_in = pq.read_table(args.assets).to_pylist()
    print(f"loading {args.model} ...", flush=True)
    enc = Qwen3VLEncoder(args.model)
    vecs = enc.encode([r["scene_text"] for r in rows_in], batch_size=args.batch_size, prompt=DOC_PROMPT)
    rows = [{"asset_uuid": r["asset_uuid"], "text_embedding": vecs[i].tolist()} for i, r in enumerate(rows_in)]
    write_relation("asset_text_embedding", rows, args.out_dir, pk=["asset_uuid"])
    print(f"wrote {len(rows)} rows -> {args.out_dir}/asset_text_embedding.parquet (dim={enc.dim})")


if __name__ == "__main__":
    main()
