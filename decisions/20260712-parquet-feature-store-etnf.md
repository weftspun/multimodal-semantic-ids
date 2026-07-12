---
title: Parquet feature store in Essential Tuple Normal Form (ETNF)
date: 2026-07-12
status: accepted
tier: baseline
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

The recommender needs a feature store for assets, users, sessions, interactions, per-modality
embeddings, and semantic IDs. The prior sketch (`session_recommendation_10.sql`) is a **wide,
denormalized OLAP `assets` table** with every modality embedding inline as a column. V-Sekai's data
lake convention (`V-Sekai-fire/study-data-vsk/SCHEMA.md`) instead mandates **Essential Tuple Normal
Form (ETNF)**: one relation per `lake/*.parquet`, BCNF, and every explicit nontrivial join dependency
has a superkey component.

## Decision Drivers

- Consistency with the V-Sekai data-lake schema convention (interoperability, shared tooling).
- No redundant / transitive attributes (derive by join, not by storage).
- Each modality encoder runs as an independent offline job — it should write an independent relation.
- Stable, content-addressable keys for cold-start dedup under asset re-import.

## Considered Options

- **Keep the wide `10.sql` `assets` table** with embeddings inline (denormalized OLAP).
- **ETNF-decomposed lake** per `study-data-vsk/SCHEMA.md`.

## Decision Outcome

Chosen: **ETNF-decomposed parquet lake**, one relation per file:

- **Entity relations (BCNF):** `assets`, `users`, `sessions` — candidate keys = deterministic
  `*_uuid` (UUIDv5) plus the retained natural key; every non-key attribute fully depends on the whole
  key. `asset_uuid = uuid5(NS, "asset:" + <canonical natural key, e.g. content hash / slug>)` — the
  canonical (allocentric) key means a re-imported/rescaled duplicate resolves to the same UUID.
- **Many-to-many facts = all-key junctions:** `session_assets (session_uuid, asset_uuid)`,
  `item_graph (source_uuid, target_uuid, cooccurrence_type)`, `user_item_edges (user_uuid, asset_uuid,
interaction_type)` — the entire heading is the key.
- **Derived-extension relations keyed by `asset_uuid`** — one relation _per modality_, each written by its
  own offline encoder job:
  - **1:1** — `asset_text_embedding` and `asset_image_embedding` (both from the single unified
    **Qwen3-VL-Embedding** space now — ModernBERT dropped), `asset_audio_embedding`, `asset_phenotype`.
  - **1:N per-token** — the mesh SLAT is a *structured* per-voxel token set (NOT a pooled vector), so it is
    **`asset_mesh_shape_slat`** (geometry) and **`asset_mesh_texture_slat`** (PBR) — full SLAT = shape ⊕
    texture — each keyed by (`asset_uuid`, token_idx) with (coord, 32-d feats), Hilbert-ordered, then
    **FSQ-quantized per token** into `asset_semantic_ids` (per-modality FSQ codes).
- **Integrity:** every PK checked unique; every FK checked to resolve into its parent (no orphans).
- Training features are assembled by **join** (a DuckDB view), never by storing derived columns.

Supersedes the denormalized layout in `session_recommendation_10.sql`.

### Consequences

- (+) Matches the V-Sekai lake; each encoder job writes one clean, independently-rebuildable relation.
- (+) No embedding-column bloat on `assets`; adding a modality = adding a relation, not a migration.
- (+) Canonical UUIDv5 keys give cold-start dedup for free (ties to
  [20260712-allocentric-i2i-over-egocentric-u2i]).
- (-) Feature assembly needs joins (a `training_features` view) rather than a single table read.
