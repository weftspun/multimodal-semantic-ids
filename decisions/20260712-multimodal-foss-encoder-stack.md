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

- **Text + Image** (unified): **Qwen3-VL-Embedding** (Apache-2.0) encodes BOTH scene/XMP text and
  rendered images into ONE shared multimodal space — so **ModernBERT is dropped for Phase 2** (kept only
  as the frozen Phase-1 MovieLens choice). Rationale: one model instead of two, and text↔image land in
  the same space (better fusion; the "unified, not siloed" principle also applied to CLAP). The companion
  **Qwen3-VL-Reranker** (Apache-2.0) feeds the retriever→ranker stage (`decisions/session_recommendation_06.md`).
  Slim fallback if inference cost demands: SigLIP-2 ViT + 2×2 merger → mean-pool for image only.
- **Mesh**: **TRELLIS.2** (MIT, weights MIT) — encode to SLAT via its Sparse-3D VAE, canonical-
  normalize orientation/scale, then pool to a fixed vector. **FOSS gate RESOLVED (2026-07-12): PASSES.**
  The encode path `mesh → o-voxel (mesh_to_flexible_dual_grid) → .vxz → shape encoder (SC-VAE) → SLAT`
  imports only `torch, numpy, o_voxel, trellis2.models` (o-voxel `pyproject.toml` deps = `torch, numpy`;
  its rasterizer uses o-voxel's own `_C` CUDA ext). `nvdiffrast`/`nvdiffrec` appear ONLY in the output/
  decode side (`postprocess.to_glb`, `mesh_renderer`, `pbr_mesh_renderer`, `trellis2_texturing`), which
  we do not use. **Windows-runnable** via the MIT fork `IgorAherne/TRELLIS.2-stableprojectorz`, which
  ships **prebuilt Windows wheels** for o-voxel/flexgemm (Python 3.11, CUDA 12.8, Torch 2.8 ≈ our cu128
  setup); mesh→O-Voxel voxelization is CPU-capable (~10s), only the shape-VAE encode wants the GPU. No
  Linux box or nvdiffrast required. (ONNX export of the shape VAE is a later deployment-portability
  option; it would not cover the native o-voxel voxelization kernel, so it is not needed now.)
  **VERIFIED 2026-07-12:** real mesh→O-Voxel voxelization runs on Windows + torch 2.8 + RTX 4090,
  FOSS-clean (icosphere 2562v/5120f → 18752 voxels; NO triton/nvdiffrast/cv2 — just MIT `o_voxel._C`).
  Two o_voxel quirks handled in `vsk_recsys/encoders/mesh.py`: (1) bypass the render-eager package
  `__init__`, import the voxelizer submodule directly; (2) pass `aabb` on CPU (their code `.cuda()`s aabb
  but calls a `*_cpu` kernel). Remaining: Stage-2 shape-VAE embed (O-Voxel→SLAT→pool) needs the 709 MB
  MIT shape encoder + `trellis2`. Isolated pixi `mesh` env (torch 2.8) added for this.
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
- (-) TRELLIS.2 is a heavy CUDA dependency (o-voxel `_C` extension), but **runs on Windows** via the
  MIT `stableprojectorz` fork's prebuilt wheels (no Linux required); nvdiffrast FOSS gate passes for the
  encode path (see above).
