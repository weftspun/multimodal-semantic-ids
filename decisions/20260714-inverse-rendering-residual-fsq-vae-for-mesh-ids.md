---
title: Two-stage inverse-rendering-corrected ResidualFSQ-VAE for 3D-asset semantic IDs
date: 2026-07-14
status: proposed
tier: stretch
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

The mesh path settled in [20260713-multimodal-residual-fsq-semantic-ids] **mean-pools** the SLAT token
set to one vector and runs **direct ResidualFSQ** on it. Mean-pooling discards the geometry the SLAT
structured latent encodes, and direct FSQ carries no reconstruction signal — the codes are whatever
survives pooling, not what actually **renders back** to the asset. For 3D assets that is the wrong
bottleneck: two visually different meshes can pool to nearby vectors, and a new asset's ID is only as
good as the pooled average.

Kyvo (Sahoo, Tibrewal & Gkioxari, [arXiv:2506.08002](https://doi.org/10.48550/arXiv.2506.08002)) shows
the 3D-token alternative: a **two-stage tokenizer** — encoder → quantizer → decoder — whose
reconstruction is supervised by a **multi-view differentiable-rendering loss**. Rendering the quantized
latent from many views and matching ground-truth renders (inverse rendering) is what *corrects the
quantizer* so the discrete codes preserve geometry rather than latent-space artifacts; the redundancy
across views averages out per-view error. Kyvo's quantizer is a **VQ-VAE**, which
[20260712-fsq-over-rqvae-for-semantic-ids] already ruled out (learned codebook → collapse, dead codes,
ID collisions).

## Decision

For the mesh (3D) modality, replace mean-pool + direct-FSQ with a **two-stage ResidualFSQ-VAE**:

    SLAT structured latent → encoder → ResidualFSQ → decoder,

trained end to end with a **multi-view differentiable-rendering loss** (inverse rendering). ResidualFSQ
stays the bottleneck (no learned codebook, per [20260712-fsq-over-rqvae-for-semantic-ids]); FSQ's
straight-through estimator (`round_ste`) carries the rendering gradient back to the encoder. The mesh
slot in the fused concat of [20260713-multimodal-residual-fsq-semantic-ids] becomes this
geometry-corrected latent instead of the mean-pooled SLAT. Text / image / audio / phenotype slots are
unchanged — inverse rendering only applies to modalities you can render.

### Why residual FSQ changes what the rendering loss corrects

- **VQ-VAE:** the multi-view rendering loss mostly corrects a **learned codebook** — nudging code
  vectors toward renderable geometry while fighting collapse.
- **ResidualFSQ:** there is **no codebook to correct or collapse**. The correction target moves to the
  **encoder** ("arrange the latent so that, after fixed FSQ rounding, it still renders across all
  views"). And the **residual stages are themselves a coarse-to-fine error-correction ladder** — each
  stage quantizes the previous stage's *residual* (its error). So the quantizer already reduces error
  algebraically; the rendering loss's remaining job is only to align that reduction with true
  multi-view geometry. Bounded error, deterministic IDs, nothing to collapse.

### Token contract

Set `num_quantizers = 4` (was 3), matching the certified 4-token semantic-ID contract the downstream
recommender proves in Lean (`residual-fsq-recommender`, `itemKey_injective`). One more residual stage
is one more error-correction pass — the direct analog of adding a view.

## Decision Drivers

- **Geometry-faithful mesh IDs** — a chair gets chair-shaped codes; the property that carries cold-start
  on new 3D assets ([20260712-allocentric-i2i-over-egocentric-u2i]).
- **Keep the FSQ charter** — no codebook collapse, no ID collisions, deterministic
  ([20260712-fsq-over-rqvae-for-semantic-ids]).
- **Reuse validated 3D supervision** — the multi-view render loss TRELLIS/Kyvo rely on, applied to a
  collapse-free quantizer.
- **Match the consumer** — the recommender's 4-token contract.

## Considered Options

- **Mean-pool + direct FSQ (status quo).** Simple, parameter-free, but discards geometry and has no
  reconstruction signal.
- **VQ-VAE + rendering loss (Kyvo as-is).** Proven for 3D tokens, but the learned codebook reintroduces
  collapse / collision — contradicts [20260712-fsq-over-rqvae-for-semantic-ids].
- **Two-stage ResidualFSQ-VAE + rendering loss (chosen).** Geometry-corrected *and* collapse-free; the
  residual stages add intrinsic coarse-to-fine error correction on top of the render supervision.

## Consequences

- (+) Mesh codes preserve renderable geometry; stronger 3D cold-start; the consuming recommender gets
  sharper, geometry-grounded IDs with no model change.
- (+) No codebook collapse or collision; deterministic; consistent with the FSQ charter.
- (−) Needs a **differentiable renderer + multi-view render data + a trained encoder/decoder** — real ML
  infrastructure versus the parameter-free status quo. This record specifies the bottleneck
  (ResidualFSQ) and objective (inverse rendering) for the mesh VAE proposed in
  [20260712-slang-lean4-mesh-vae-over-trellis2-cuda].
- (−) `num_quantizers` 3 → 4 reflows the **shared** FSQ config in
  [20260713-multimodal-residual-fsq-semantic-ids] and invalidates existing IDs; `asset_semantic_id`
  must be regenerated for all modalities.

## Verification

Train the mesh ResidualFSQ-VAE on a multi-view render set and confirm: (a) held-out multi-view rendering
error beats mean-pool + direct-FSQ at an equal token budget; (b) FSQ collision rate ~0; (c) codes are
**stable under pose / lighting jitter** (the redundancy payoff — the same asset rendered differently
still quantizes to the same ID); (d) the 4-token IDs load into `residual-fsq-recommender` and pass
`ResidualFSQ.valid_id?/1`.
