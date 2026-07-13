"""Multimodal recommender, stage 1: concat modality vectors -> one ResidualFSQ -> semantic IDs.

Each modality reduces to ONE per-asset vector (single-vector encoders used as-is; mesh SLAT token sets
mean-pooled via slat-semantic-ids). Present modalities are standardized and concatenated in a fixed slot
layout, with a modality absent for a given asset zero-filled, then a single FSQ-autoencoder tokenizer
assigns every asset a semantic-ID tuple. Content-derived IDs need no interaction data, so a brand-new
asset gets codes immediately (cold-start). Writes the `asset_semantic_id` ETNF relation.

This is the concat-vector -> ResidualFSQ realization of
decisions/20260713-multimodal-residual-fsq-semantic-ids.md — `mesh_semantic_ids.py` generalized from the
two mesh blocks (shape+texture) to all modalities. The FSQ kernel and mesh SLAT pooling are reused from
`slat-semantic-ids` (one source of truth). Pure PyTorch CPU — no CUDA.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from slat_semantic_ids import SemanticIDTokenizer
from slat_semantic_ids.slat import l2_normalize, pool_tokens

ASSET_KEY = "asset_uuid"

# Fixed concat order: (slot name, relation, feature column, multitoken?).
# multitoken modalities are mean-pooled to one vector (mesh SLAT); the rest are single-vector encoders.
MODALITIES = (
    ("text", "asset_text_embedding", "text_embedding", False),
    ("image", "asset_image_embedding", "image_embedding", False),
    ("mesh_shape", "asset_mesh_shape_slat", "slat_feats", True),
    ("mesh_texture", "asset_mesh_texture_slat", "slat_feats", True),
    ("audio", "asset_audio_clap", "audio_embedding", False),
    ("phenotype", "asset_body_phenotype", "phenotype_params", False),
)

LEVELS = (8, 8, 8, 8)
NUM_QUANTIZERS = 3
LATENT_DIM = 32
TRAIN_STEPS = 800
LEARNING_RATE = 1e-3
SEED = 0


def _vector_by_asset(relation_path: Path, feature_col: str, multitoken: bool) -> dict[str, np.ndarray]:
    """`{asset_uuid: l2-normalized per-asset vector}` for one modality relation (`{}` if absent)."""
    if not relation_path.exists():
        return {}
    frame = pd.read_parquet(relation_path)
    out: dict[str, np.ndarray] = {}
    for _, row in frame.iterrows():
        feats = row[feature_col]
        vec = pool_tokens(feats) if multitoken else np.asarray(feats, dtype=np.float32)
        out[row[ASSET_KEY]] = l2_normalize(vec)
    return out


def load_fused_embeddings(lake_dir: str | Path) -> tuple[list[str], np.ndarray, list[str]]:
    """`(asset_uuids, fused, present_slots)` — present modality vectors concatenated in fixed slots.

    A modality present in the corpus but missing for a given asset is zero-filled in that asset's slot,
    so any non-empty subset of modalities still yields an ID (cold-start).
    """
    lake = Path(lake_dir)
    blocks = [
        (name, _vector_by_asset(lake / f"{rel}.parquet", col, multi))
        for name, rel, col, multi in MODALITIES
    ]
    present = [(name, block) for name, block in blocks if block]
    if not present:
        return [], np.empty((0, 0), dtype=np.float32), []

    dims = {name: len(next(iter(block.values()))) for name, block in present}
    uuids = sorted({u for _, block in present for u in block})
    rows = [
        np.concatenate([block.get(u, np.zeros(dims[name], np.float32)) for name, block in present])
        for u in uuids
    ]
    return uuids, np.stack(rows).astype(np.float32), [name for name, _ in present]


def assign_semantic_ids(lake_dir: str | Path) -> pd.DataFrame:
    torch.manual_seed(SEED)
    uuids, embeddings, slots = load_fused_embeddings(lake_dir)
    if not uuids:
        raise SystemExit("no modality feature relations found in the lake")

    # Standardize per feature: FSQ's fixed uniform grid assumes a centered, bounded input.
    x = torch.from_numpy(embeddings)
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

    tokenizer = SemanticIDTokenizer(
        d_in=x.shape[1], d_latent=LATENT_DIM, levels=LEVELS, num_quantizers=NUM_QUANTIZERS
    )
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=LEARNING_RATE)

    tokenizer.train()
    for _step in range(TRAIN_STEPS):
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
    print(f"modalities={slots}")
    print(f"assets={len(uuids)} fused_dim={x.shape[1]} recon_mse={float(loss):.4f}")
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
