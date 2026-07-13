---
title: Mesh-only content scope — postpone the Qwen3-VL image/text embedder
date: 2026-07-12
status: accepted
tier: scope
decision-makers: K. S. Ernest (iFire) Lee
supersedes-partial: 20260712-multimodal-foss-encoder-stack
---

## Context and Problem Statement

The multimodal encoder stack ([20260712-multimodal-foss-encoder-stack]) called for encoding every Godot
asset from **four** modalities — scene text, rendered image, 3D mesh, audio — into the ETNF lake before
quantization. The image↔text half was to be a **native Bumblebee/Nx port of `Qwen/Qwen3-VL-Embedding-2B`**
(no Pythonx), added to the `weftspun/bumblebee` fork.

That port is large (a full VLM: vision tower + Qwen3 LM decoder + 3-D interleaved mRoPE + deepstack + a
variable-patch featurizer, numerically verified against a 2 B checkpoint). **We have run out of budget** to
finish it now. Meanwhile the **3D-mesh encoder already works** in Python (TRELLIS.2 SLAT shape⊕texture,
verified) and is the highest-signal content feature for the V-Sekai asset catalogue.

## Decision

1. **Reduce content-encoding scope to 3D meshes only** for the current milestone. The recommender's content
   features come from the mesh SLAT encoder; scene-text and rendered-image embeddings are **deferred**.
2. **Postpone the native Qwen3-VL image/text embedder** (the `weftspun/unified-modal-embedder` cluster). Work
   is parked, not discarded — it is resumable from the state below.
3. Scene-text embedding (the "separate text model" discussed) is likewise deferred; no text encoder is chosen
   now.

## Parked state of the Qwen3-VL port (how to resume)

- **Fork**: `weftspun/bumblebee`, branch `qwen3-vl-embedding` (`upstream` → elixir-nx). Local clone
  `C:\Users\ernes\Desktop\bumblebee`.
- **Done + verified**: `Bumblebee.Vision.Qwen3VLVision` (the vision tower) — pushed, compiles, and matches the
  reference numerically (pooled_state 9.0e-5, deepstack ≤1.2e-5 vs golden). This is action 1 of 6.
- **Remaining (≈PT10H, from the taskweft MCP solve)**: `a_featurizer` (Qwen3VLFeaturizer: reproduce
  `Qwen2VLImageProcessorFast` + precompute the bilinear-indices/weights & vision-RoPE cos/sin the tower
  consumes) → `a_fusion` (masked-scatter vision tokens into the Qwen3 LM stream, 3-D interleaved mRoPE
  [24,20,20] θ=5e6, deepstack injection after LM layers 0/1/2, last-token pool) → `a_registry`
  (`lib/bumblebee.ex` maps) → `a_e2e` (cosine ≥ 0.999 vs golden `image_embedding`) → `a_wire`
  (`unified-modal-embedder`).
- **Spec + golden**: `notes/qwen3_vl_spec.md` in the fork; fixtures `test/fixtures/qwen3_vl/{golden.json,
  golden_image.png,vision_dump.npz}` (+ the 1.6 GB `vision_probe.safetensors`, gitignored, regenerable).
- **Backend**: EXLA has no Windows build → use **Torchx** (short build path, e.g. `C:\q3vl`); EXLA on the
  Linux Burrito target.
- **Full parked plan**: `~/.claude/plans/study-c-users-ernes-desktop-gait-classif-sorted-squid.md`.

## Consequences

- The `unified-modal-embedder` cluster is **paused** at action 1/6; its vision tower stays committed in the
  fork so no work is lost.
- The immediate track is the **3D-mesh encoder** — the `voxel-slat-encoder` weftspun cluster (TRELLIS.2 SLAT
  via the proven Python `vsk_recsys/encoders/mesh.py`), and eventually the CUDA-free Slang/Lean4 mesh VAE
  ([20260712-slang-lean4-mesh-vae-over-trellis2-cuda]).
- Cross-modal image↔text retrieval is unavailable until the port resumes; recommendations rely on mesh
  semantic IDs (plus existing uro interaction signal when it lands).
- [20260712-multimodal-foss-encoder-stack] is **partially superseded**: its text+image encoder choice is
  deferred; its mesh and (later) audio/phenotype choices stand.
