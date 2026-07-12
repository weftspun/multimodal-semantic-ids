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
| **Text** (Godot scenes+scripts) — PRIMARY | `godotengine/godot-demo-projects` (git, MIT) | **MIT** | Real scene-graph text: **394 `.tscn` + 458 `.gd` + 137 `.tres` + 46 `.gdshader` across 138 projects** (9.1k★). Single clean source; closest public analog to V-Sekai assets. **Scenes are the items; each project is a session** (see Consequences). GDScript-only supplements exist (`icici121/godot-gdscript-dataset`, Apache-2.0) but are not needed. |
| **Audio**                 | `agkphysics/AudioSet`                                       | **CC-BY-4.0** | For the LAION-CLAP audio encoder.                                  |
| Audio (Freesound/LAION)   | `benjamin-paine/freesound-laion-640k-commercial-16khz-full` | **CC-BY-4.0** | Commercial-safe Freesound/LAION-audio.                             |
| **Phenotype: facial expression** | `nadizik/synthetic-human-expressions-poses-3d`      | **CC-BY-4.0** | Synthetic human **facial-expression** renders (FACS controls + camera + breathing + eye-gaze) — **NOT body skeleton**. Ingested to ETNF: `assets_poses` + `asset_pose_meta` (`vsk_recsys/data/nadizik.py`; 10,075 renders, 26 motions, 4 quality buckets). rf-detr keypoints run on these renders. |
| **Phenotype: body motion** | **AddBiomechanics** → `sinew-mocap/mount-drift` `calibrator-v1` `caldata_{train,test}_jc.parquet` | **CC-BY-4.0** | Real body-**skeleton** jointCenters + anthropometrics for the COCO-17 → `somaxc` → ANNY *body* retarget (and the `Anthropometry.lean` bone-length solve). Columns: `subject/study/source`, `recorded_sex`, `height_m`, `mass_kg`, `seglen_m` (10 segment lengths), `x` (7200 windowed jointCenters), `y` (180 per-frame). **Use the `With_Arm` Rajagopal variant** (arms required for COCO-17; AddBiomechanics notes much of its data lacks arm markers) and **filter feminine** subjects (test set is 9,373/11,794 female). Already parquet (zstd-internal). AddBiomechanics is CC-BY (attribution via LICENSE.txt uploader credits); NOT SMPL — clears the FOSS gate. Local raw `.b3d`: `O:\addb-all\{train,test}\With_Arm\`. Superbuild has no photos (skeletal mocap only). |

**Rejected (not FOSS):** `ShapeNet/ShapeNetCore` (license "other", gated/manual); `laion/relaion2B-en-
research-safe` and `laion/laion2B-en-aesthetic` (license unknown — URL lists with mixed image rights);
`pixparse/cc12m-wds` (license "other"). LAION's _safety_ filtering does not make its _license_ clean —
prefer the CC-BY image/audio alternatives.

## Consequences

- (+) Every modality has genuine, license-clean data; TRELLIS-500K alone covers mesh+image+text.
- (+) Re-pointing to real Godot assets later is a data-source swap, not a code change (ETNF lake).
- (+) **`godot-demo-projects` gives real item→item structure without fabricating interactions** — a
  **bootstrap proxy**: **items = `.tscn` scenes; sessions = projects** (scenes co-occurring in one project).
  Cleanest public session signal; no synthetic co-occurrence walks. Wire in `a_p2_i2i`.
- (+) **The REAL user→item source is V-Sekai's `uro` `Inventory.Backpack`** (`V-Sekai/uro`,
  `benbot/backpack-inventory`): a **backpack = a user's timestamped set of owned items** = a session /
  "basket". We call these **backpacks** (V-Sekai's term); there is **no "category"** concept (items are
  *typed* Map / Avatar / Prop, i.e. content types, not user categories). Backpacks are the genuine
  interaction/ownership graph and supersede the Godot proxy once available — see the uro→ETNF mapping in
  [20260712-parquet-feature-store-etnf].
- (-) The 3D/audio/body datasets still lack sessions; if mixed in, fall back to content-similarity
  co-occurrence. The Godot proxy corpus is small (~394 scenes) — fine for a proof-of-stack + cold-start
  demo, not large-scale training (real scale comes from uro backpacks).
- (-) **Godot `project=session` is too sparse for *sequential* training** — ~2.9 scenes/project, so most
  sessions are length-2 (the "drop < 3" rule would drop most). **Decision (2026-07-12): pause the pure-text
  Godot recommender training; exercise the OTHER modality encoders (mesh, image, phenotype) on real content
  first and build the multimodal ETNF relations, and defer sequential-recommender training (`a_p2_i2i`)
  until real interaction data (uro backpacks) exists.** Content-derived semantic IDs still give cold-start
  regardless. The Godot corpus's real 3D meshes (31 `.glb`/`.gltf`/`.obj`, incl. character models →
  phenotype path) feed the mesh encoder now.
- (-) **Phenotype needs TWO data sources, not one** — the `nadizik` set is **facial expression** (FACS),
  which does NOT cover the **body skeleton**. The COCO-17 → `somaxc` → ANNY retarget targets the body, so a
  separate FOSS **body-motion** dataset is still required (see the two phenotype rows). Also note: the Godot
  corpus has **no human assets** (stylized robots/creatures + 1 mannequin), so rf-detr keypoints run on the
  human datasets, NOT the Godot proxy — correcting the "Godot character meshes → phenotype" note above.
- **Body-motion pipeline = collapse mocap onto ANNY, then synthesize** (2026-07-12): the raw
  AddBiomechanics jointCenters are heterogeneous skeletons, so **retarget/collapse them ALL onto the
  canonical ANNY skeleton via `somaxc`** (one rig for every source), then **generate synthetic data** —
  pose ANNY in the collapsed motions → render (viser) → rf-detr COCO-17 → the body keypoint/phenotype set.
  This is the body-side twin of how `nadizik` synthesized facial data: controllable, canonical, FOSS, and
  it exercises the whole `somaxc` retarget. Feminine + `With_Arm` subjects only.
- ODC-BY / per-object Objaverse licenses require attribution and per-object filtering for redistribution.
