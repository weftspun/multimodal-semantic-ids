---
title: Phase 2 FOSS datasets — one verified dataset per modality
date: 2026-07-12
status: accepted
tier: proof of concept
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

Phase 2 (the Godot multimodal recommender) needs real content to encode, but no V-Sekai/Godot asset
corpus exists yet. Rather than a synthetic fixture, we stand in **real, FOSS-licensed HuggingFace
datasets** — one per modality — so the encoders + ETNF store are exercised on genuine data and
swapping in true Godot assets later is just re-pointing the lake. Every dataset license was verified
against the HF API on 2026-07-12; only OSI/permissive or CC-BY/CC0/ODC-BY qualifies.

## Decision Outcome — datasets per modality

| Modality                  | Dataset (HF id)                                             | License       | Notes                                                              |
| ------------------------- | ----------------------------------------------------------- | ------------- | ------------------------------------------------------------------ |
| **Mesh** (primary)        | `JeffreyXiang/TRELLIS-500K`                                 | **MIT**       | The TRELLIS asset set — native to our mesh encoder; ships renders. |
| Mesh (interactive)        | `Seed3D/Articulation-XL2.0`                                 | **CC-BY-4.0** | Articulated objects — closest to interactive Godot assets.         |
| Mesh (CC-BY subset)       | `BAAI/Objaverse-MIX`                                        | **CC-BY-4.0** | Clean CC-BY 3D subset.                                             |
| Mesh (scale)              | `allenai/objaverse-xl`                                      | **ODC-BY**    | Huge; per-object licenses vary — filter for CC0/CC-BY.             |
| **Image** (asset renders) | bundled in `TRELLIS-500K`                                   | **MIT**       | Renders are the item image modality — no separate set needed.      |
| Image (general, alt)      | `whyen-wang/coco_captions`                                  | **CC-BY-4.0** | If in-the-wild images are wanted.                                  |
| **Text** (asset caption)  | `tiange/Cap3D`                                              | **ODC-BY**    | Captions for the 3D objects (generic object text).                 |
| **Text** (Godot GDScript) | `icici121/godot-gdscript-dataset`                          | **Apache-2.0** | Real GDScript code as text (11k dl); also `wallstoneai/godot-gdscript-dataset`, `Miauuuuuuu/godot-ds3-deg` (Apache-2.0). |
| **Text** (Godot `.tscn` scenes) | GitHub `godotengine/godot-demo-projects` + permissive Godot repos | **MIT** | Actual scene-graph text — **no packaged HF dataset exists**; scrape `.tscn`/`.escn` via GitHub code search (`extension:tscn`). |
| **Audio**                 | `agkphysics/AudioSet`                                       | **CC-BY-4.0** | For the LAION-CLAP audio encoder.                                  |
| Audio (Freesound/LAION)   | `benjamin-paine/freesound-laion-640k-commercial-16khz-full` | **CC-BY-4.0** | Commercial-safe Freesound/LAION-audio.                             |
| **Body / phenotype**      | `nadizik/synthetic-human-expressions-poses-3d`              | **CC-BY-4.0** | Synthetic 3D human poses → rf-detr keypoints → ANNY fit.           |

**Rejected (not FOSS):** `ShapeNet/ShapeNetCore` (license "other", gated/manual); `laion/relaion2B-en-
research-safe` and `laion/laion2B-en-aesthetic` (license unknown — URL lists with mixed image rights);
`pixparse/cc12m-wds` (license "other"). LAION's _safety_ filtering does not make its _license_ clean —
prefer the CC-BY image/audio alternatives.

## Consequences

- (+) Every modality has genuine, license-clean data; TRELLIS-500K alone covers mesh+image+text.
- (+) Re-pointing to real Godot assets later is a data-source swap, not a code change (ETNF lake).
- (-) **No interaction/session data exists** in any of these — the recommender's user→item signal must
  be **simulated** (content-similarity / shared-category co-occurrence walks over the catalog) until
  V-Sekai telemetry exists. Decide the simulation scheme when wiring `a_p2_i2i`.
- ODC-BY / per-object Objaverse licenses require attribution and per-object filtering for redistribution.
