---
title: FSQ over RQ-VAE for semantic-ID quantization
date: 2026-07-12
status: accepted
tier: proof of concept
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

Generative retrieval ([20260712-migrate-off-librecommender-to-foss-generative-retrieval]) needs a
quantizer that turns a continuous item embedding into a tuple of discrete codes (the "semantic ID").
The canonical choice (TIGER) is **RQ-VAE**, which carries a learned codebook plus commitment /
reconstruction losses and EMA codebook updates — and the failure modes that follow: codebook
collapse, dead codes, and **semantic-ID collisions** (distinct items sharing one ID). This is not
hypothetical: `phonism/genrec` withholds benchmark numbers because RQ-KMeans IDs collide and inflate
SID-matched Recall.

## Decision Drivers

- Simplicity / understandability of the tokenizer.
- No codebook collapse, no ID collisions, deterministic + reproducible.
- Robustness under heavy cold-start / new-item churn.
- Pure-PyTorch, FOSS (Apache-2.0 / MIT).

## Considered Options

- **RQ-VAE** (learned residual codebook) — TIGER default.
- **RQ-KMeans** (Spotify) — still a learned/clustered codebook; collision-prone.
- **LFQ** (lookup-free quantization).
- **FSQ / ResidualFSQ** (Finite Scalar Quantization) — bound each scalar, round to fixed levels,
  straight-through gradients; the code is the tuple of rounded per-dimension levels.

## Decision Outcome

Chosen: **`ResidualFSQ`** from **`lucidrains/vector-quantize-pytorch`** (MIT, ~4k★, pure PyTorch,
actively maintained). FSQ has no learned codebook and no aux losses, which removes collapse and
collision handling entirely; ResidualFSQ adds the coarse→fine multi-level code structure generative
retrieval wants while keeping FSQ's simplicity. `HKUDS/RecGPT` validates the FSQ-tokenizer + decoder
pattern for recsys and is used as a *reference design only* (it ships no license — see
[20260712-semantic-id-framework-base-rqvae-recommender]).

### Consequences

- (+) Simpler, deterministic tokenizer; no collapse, near-zero collisions.
- (+) Same dependency `RQ-VAE-Recommender` already uses — swapping RQ-VAE→FSQ is a small change.
- (-) FSQ's codebook is a fixed uniform grid, so it is less adaptive than a learned codebook when the
  fused embedding distribution is skewed → mitigate with input normalization and level tuning.
- Open: number of levels and residual stages (capacity vs collision) — tune on MovieLens.
