---
title: Migrate off LibRecommender/PinSage → FOSS PyTorch generative-retrieval recommender
date: 2026-07-12
status: accepted
tier: baseline
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

The working recommender (`recommend.py`, `load.py`) is built on **LibRecommender's PinSage**.
Three problems block the V-Sekai goal of recommending user-generated Godot scene/assets:

1. **Wrong stack.** LibRecommender is TensorFlow-based; the team's stack (`../gait-classification`)
   is PyTorch + pixi + parquet + MADR logs. The recommender is the odd one out.
2. **Poor cold-start.** PinSage learns one embedding per item ID, so it cannot recommend items
   never seen in training — fatal for a UGC asset catalog with constant new-item churn.
3. **No multimodal content.** Godot assets carry mesh / image / audio / scene-text; PinSage's
   ID+sparse-genre features cannot express them.

## Decision Drivers

- FOSS-only: OSI-approved permissive licenses (Apache-2.0 / MIT), open weights, no token gate.
- PyTorch, to match the house stack.
- Cold-start via *content-derived* item representations (new items usable at inference).
- Native multimodal item features.

## Considered Options

- Keep LibRecommender PinSage (TensorFlow).
- Re-implement PinSage as a graph NN in PyG/DGL.
- Two-tower / SASRec sequential model with content-initialised ID embeddings.
- **Semantic-ID generative retrieval** (quantize content → discrete IDs → Transformer generates next IDs).

## Decision Outcome

Chosen: **semantic-ID generative retrieval, pure PyTorch, FOSS**. Items are encoded from their
content into discrete semantic IDs and a Transformer generates the next item's IDs. Content-derived
IDs give brand-new assets a meaningful representation for free, fixing cold-start; the quantizer and
generator are plain PyTorch. Quantizer choice in [20260712-fsq-over-rqvae-for-semantic-ids];
framework base in [20260712-semantic-id-framework-base-rqvae-recommender]; encoders in
[20260712-multimodal-foss-encoder-stack]; retrieval framing in
[20260712-allocentric-i2i-over-egocentric-u2i].

### Consequences

- (+) One coherent PyTorch/pixi/parquet stack across V-Sekai repos; commercial-safe FOSS.
- (+) Cold-start handled by construction (content → semantic ID).
- (-) Larger build than porting PinSage — new quantizer, generator, and multimodal encoders.
- (-) `recommend.py` / `load.py` (the only `libreco` importers) are retired.
