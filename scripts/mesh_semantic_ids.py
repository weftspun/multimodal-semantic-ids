"""Mesh-only recommender, stage 1: content -> FSQ semantic IDs (cold-start ready).

Pools each asset's mesh SLAT (shape + texture) to a fixed embedding, trains the FSQ
autoencoder tokenizer with reconstruction, and assigns every mesh a semantic-ID tuple.
Content-derived IDs need no interaction data, so a brand-new mesh gets codes immediately
(cold-start). Writes the `asset_semantic_id` ETNF relation.

This is the non-data-gated core of the mesh-only recommender; the i2i decoder over
sessions waits on real uro interactions. Pure PyTorch CPU — no CUDA.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vsk_recsys.quantizer.tokenizer import SemanticIDTokenizer

SHAPE_RELATION = "asset_mesh_shape_slat"
TEXTURE_RELATION = "asset_mesh_texture_slat"
ASSET_KEY = "asset_uuid"
FEATS_COLUMN = "slat_feats"

LEVELS = (8, 8, 8, 8)
NUM_QUANTIZERS = 3
LATENT_DIM = 32
TRAIN_STEPS = 800
LEARNING_RATE = 1e-3
SEED = 0


def _pool_tokens(feats: object) -> np.ndarray:
    tokens = np.stack([np.asarray(token, dtype=np.float32) for token in feats])
    return tokens.mean(axis=0)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0.0 else vector / norm


def _pooled_by_asset(relation_path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_parquet(relation_path)
    return {row[ASSET_KEY]: _l2_normalize(_pool_tokens(row[FEATS_COLUMN])) for _, row in frame.iterrows()}


def load_mesh_embeddings(lake_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """`(asset_uuids, embeddings)` — mean-pooled shape SLAT concatenated with texture SLAT."""
    lake = Path(lake_dir)
    shape = _pooled_by_asset(lake / f"{SHAPE_RELATION}.parquet")
    texture = _pooled_by_asset(lake / f"{TEXTURE_RELATION}.parquet")
    if not shape:
        return [], np.empty((0, 0), dtype=np.float32)

    texture_dim = len(next(iter(texture.values()))) if texture else 0
    zero_texture = np.zeros(texture_dim, dtype=np.float32)

    uuids = sorted(shape)
    rows = [np.concatenate([shape[u], texture.get(u, zero_texture)]) for u in uuids]
    return uuids, np.stack(rows).astype(np.float32)


def assign_semantic_ids(lake_dir: str | Path) -> pd.DataFrame:
    torch.manual_seed(SEED)
    uuids, embeddings = load_mesh_embeddings(lake_dir)
    if not uuids:
        raise SystemExit("no mesh SLAT relations found in the lake")

    # Standardize per feature: FSQ's fixed uniform grid assumes a centered, bounded input.
    x = torch.from_numpy(embeddings)
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

    tokenizer = SemanticIDTokenizer(
        d_in=x.shape[1], d_latent=LATENT_DIM, levels=LEVELS, num_quantizers=NUM_QUANTIZERS
    )
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=LEARNING_RATE)

    tokenizer.train()
    for step in range(TRAIN_STEPS):
        optimizer.zero_grad()
        recon, _ids = tokenizer(x)
        loss = torch.nn.functional.mse_loss(recon, x)
        loss.backward()
        optimizer.step()

    tokenizer.eval()
    with torch.no_grad():
        _recon, ids = tokenizer(x)
    ids = ids.cpu().numpy()

    frame = pd.DataFrame(
        {ASSET_KEY: uuids, **{f"code_{q}": ids[:, q] for q in range(NUM_QUANTIZERS)}}
    )
    frame["semantic_id"] = [tuple(int(c) for c in row) for row in ids]

    unique = frame["semantic_id"].nunique()
    codebook = int(np.prod(LEVELS))
    print(f"assets={len(uuids)} dim={x.shape[1]} recon_mse={float(loss):.4f}")
    print(f"codebook={codebook} per code, {NUM_QUANTIZERS} codes/item")
    print(f"unique semantic IDs: {unique}/{len(uuids)} (collisions={len(uuids) - unique})")
    return frame


def main() -> None:
    frame = assign_semantic_ids("lake")
    out = Path("lake") / "asset_semantic_id.parquet"
    frame.drop(columns=["semantic_id"]).to_parquet(out, index=False)
    print(f"wrote {out}")
    print(frame[[ASSET_KEY, "code_0", "code_1", "code_2"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
