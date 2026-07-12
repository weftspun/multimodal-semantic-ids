---
title: Multimodal FOSS encoder stack (text / image / mesh / audio / phenotype)
date: 2026-07-12
status: accepted
tier: baseline
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

Semantic-ID generative retrieval needs each Godot asset encoded from its content into a fused vector
before quantization ([20260712-fsq-over-rqvae-for-semantic-ids]). No existing FOSS repo covers
mesh/image/audio/scene-text cold-start out of the box, so we assemble our own encoders — every one of
which must be OSI Apache-2.0 / MIT (weights included). Licenses verified against repository LICENSE
files on 2026-07-12.

## Decision Drivers

- FOSS-only (Apache-2.0 / MIT), including model weights.
- One shared, cross-modal, object-canonical (allocentric) representation space
  ([20260712-allocentric-i2i-over-egocentric-u2i]).
- Heavy models run as offline GPU batch jobs → parquet (one relation per modality,
  [20260712-parquet-feature-store-etnf]).

## Considered Options and Decision Outcome

Chosen per modality (rejected alternatives noted):

- **Text** (scene_text, xmp_rdf, display_name, slug): **ModernBERT** (Apache/MIT). Matches
  `session_recommendation_13.yaml`.
- **Image**: **Qwen3-VL-Embedding** (Apache-2.0) — the Qwen3.6 SigLIP-2 vision tower + 2×2 MLP merger,
  packaged as an embedder. Rejected: rolling our own tower (kept as a slim fallback: SigLIP-2 ViT +
  2×2 merger → mean-pool, if inference cost demands).
- **Mesh**: **TRELLIS.2** (MIT, weights MIT) — encode to SLAT via its Sparse-3D VAE, canonical-
  normalize orientation/scale, then pool to a fixed vector. **FOSS gate RESOLVED (2026-07-12): PASSES.**
  The encode path `mesh → o-voxel (mesh_to_flexible_dual_grid) → .vxz → shape encoder (SC-VAE) → SLAT`
  imports only `torch, numpy, o_voxel, trellis2.models` (o-voxel `pyproject.toml` deps = `torch, numpy`;
  its rasterizer uses o-voxel's own `_C` CUDA ext). `nvdiffrast`/`nvdiffrec` appear ONLY in the output/
  decode side (`postprocess.to_glb`, `mesh_renderer`, `pbr_mesh_renderer`, `trellis2_texturing`), which
  we do not use. Remaining constraint: Linux + CUDA (o-voxel `_C` compilation).
- **Audio**: **LAION-CLAP** (Apache-2.0) — shared text↔image↔audio space (cross-modal, not an acoustic
  silo). Rejected: **Microsoft msclap** (MS-PL — OSI but non-standard/copyleft-ish).
- **Body phenotype** (an **item** feature for humanoid/character assets): **rf-detr** 2D COCO keypoints
  → **`fit_phenotype`** (differentiable inverse fit reusing `../gait-classification/anny_phenotype.py`)
  using **SOMA-X + ANNY** as the forward body → **canonical allocentric ANNY shape params**. Rejected:
  rf-detr **XL/2XL / `rfdetr_plus`** (Roboflow PML, not Apache); **SMPL / SMPL-X** body files (non-
  commercial license) — ANNY chosen precisely to avoid them.

Fused modality vector → `ResidualFSQ` → per-asset semantic ID.

### Consequences

- (+) Fully FOSS, cross-modal, cold-start-capable item representation.
- (+) Each encoder is an independent offline job writing one ETNF relation.
- (-) rf-detr gives 2D keypoints → single-view 2D→3D fit is depth-ambiguous → approximate phenotype.
- (-) TRELLIS.2 is a heavy, Linux/CUDA-only dependency (o-voxel `_C` extension); the nvdiffrast FOSS
  gate was checked and passes for the encode path (see above).
