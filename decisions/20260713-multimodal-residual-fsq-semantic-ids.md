---
title: Covering every modality in ResidualFSQ — concat vector → one ResidualFSQ → semantic ID
date: 2026-07-13
status: accepted — one rule for text / image / mesh / audio / phenotype: reduce to one vector, concat, quantize
tier: baseline
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

[20260712-multimodal-foss-encoder-stack] gives each modality its own FOSS encoder writing an independent
ETNF relation, and [20260712-fsq-over-rqvae-for-semantic-ids] fixes the quantizer as `ResidualFSQ`. What
was left underspecified is **how the modalities combine into one semantic ID**. The encoder-stack doc had
a one-liner — *"Fused modality vector → ResidualFSQ → per-asset semantic ID"* — but its mesh notes also
mused about per-token ordered code sets, and `scripts/mesh_semantic_ids.py` only ever concatenated the
two mesh blocks (shape+texture). This record settles it for all modalities.

## Decision

**Reduce each modality to one vector, concatenate the present modalities into a fused vector, and run a
single `ResidualFSQ` over it → the per-asset semantic ID.** This is `mesh_semantic_ids.py` generalized
from 2 blocks to 6 — the same standardize → concat → FSQ path, now spanning every modality.

### 1. Each modality → one vector

| Modality | Relation | Feature col | Reduction to one vector |
|----------|----------|-------------|-------------------------|
| text | `asset_text_embedding` | `text_embedding` | encoder embedding as-is (Qwen3-VL, 2048-d) |
| image | `asset_image_embedding` | `image_embedding` | encoder embedding as-is (Qwen3-VL) |
| mesh-shape | `asset_mesh_shape_slat` | `slat_feats` | **mean-pool** the SLAT token set (`slat_semantic_ids.slat.pool_tokens`) |
| mesh-texture | `asset_mesh_texture_slat` | `slat_feats` | mean-pool the SLAT token set |
| audio | `asset_audio_clap` | `audio_embedding` | CLAP embedding as-is |
| phenotype | `asset_body_phenotype` | `phenotype_params` | canonical ANNY param vector as-is |

Each vector is L2-normalized per modality.

### 2. Standardize, concat (fixed slots, zero-fill absent), one ResidualFSQ

Concatenate present modalities in a **fixed slot layout** `[text | image | mesh_shape | mesh_texture |
audio | phenotype]`. A modality present in the corpus but **missing for a given asset** is **zero-filled**
in that asset's slot (exactly as `mesh_semantic_ids` zero-fills a missing texture today), so any non-empty
subset still yields an ID — cold-start holds ([20260712-allocentric-i2i-over-egocentric-u2i]). Standardize
per feature across the corpus (FSQ's fixed grid assumes a centered, bounded input), then a **single**
`ResidualFSQ` (`levels=(8,8,8,8)`, `num_quantizers=3`, latent 32 — the validated mesh config) maps the
fused vector → the semantic-ID tuple. Realized in `scripts/semantic_ids.py`; writes `asset_semantic_id`.

## Decision Drivers

- **One rule, all modalities**, reusing the existing FSQ-autoencoder tokenizer unchanged.
- **Partial assets** (a prop has mesh but no audio; a track has audio only) must still get an ID.
- **DRY / one source of truth** — the FSQ kernel and mesh SLAT pooling are imported from
  `slat-semantic-ids`, not reimplemented; mesh/3D is referenced, not duplicated.

## Consequences

- (+) Adding a modality = register a relation + append a slot to `MODALITIES`; no new quantizer, no new
  ID scheme.
- (+) Fixed-length, single-tuple item ID (`num_quantizers` codes) — a clean index key for retrieval.
- (−) A dominant-dimensionality modality (text/image at 2048-d) can swamp smaller slots; per-feature
  standardization mitigates it, and per-slot weighting is the tuning lever if needed.
- (−) Zero-filled slots cost fused-vector width; acceptable at these dims.
- Supersedes the "Fused modality vector → ResidualFSQ" one-liner in
  [20260712-multimodal-foss-encoder-stack] (now a pointer here) and generalizes the retired
  `mesh_semantic_ids.py`.

## Verification

Run `scripts/semantic_ids.py` on a mixed-modality lake subset: confirm (a) an asset with only a subset of
modalities still gets a semantic ID (zero-filled slots), (b) FSQ collision rate ~0, (c)
`asset_semantic_id.parquet` written. Tune `levels`/`num_quantizers` (and any per-slot weighting) on
MovieLens per [20260712-fsq-over-rqvae-for-semantic-ids].
