---
title: OpenUSD mesh corpus + all-Elixir voxel-slat-encoder cluster
date: 2026-07-12
status: accepted
tier: architecture
decision-makers: K. S. Ernest (iFire) Lee
relates-to: 20260712-mesh-only-scope-postpone-qwen3vl-embedder, 20260712-slang-lean4-mesh-vae-over-trellis2-cuda
---

## Context

With content scope reduced to 3D meshes ([20260712-mesh-only-scope-postpone-qwen3vl-embedder]), we need
(a) a clean, licensable mesh corpus and (b) the mesh→SLAT encoder as a weftspun hexagonal cluster. The 31
godot-demo meshes are too few and mixed-license; the mesh encoder was Python-only.

## Decision

### 1. Mesh corpus = OpenUSD copy of thebasemesh.com
Publish the **complete** CC0 base-mesh library from thebasemesh.com as **OpenUSD ASCII** (`.usda`):
`github.com/fire/thebasemesh-openusd` — **1,254 models** + `thebasemesh.parquet` (ETNF, `asset_uuid =
uuid5(NAMESPACE, "asset:basemesh:<name>")`, joins the lake). Conversion: Blender **ufbx** importer on the
**FBX** source (not the triangulated `.glb`), transform baked, USD export **Y-up / −Z forward / right-handed /
real-world meters, n-gon topology preserved** — verified across all 1,254. Enumerated via the dynamic-asset
sitemap (each `/asset/<name>` page carries the archive URL). Reader used the `.usda`, not the M3-org glTF copy.

### 2. `voxel-slat-encoder` = all-Elixir hexagon cluster
`github.com/weftspun/voxel-slat-encoder`. USD-read **and** voxelize in Elixir; only the VAE inference calls
Python.
- **USD-read**: a **Fine NIF** over the `stage_runtime` OpenUSD SDK (`fabric-openusd-runtime`), reusing
  cloth-fit's `USDReader` logic (`UsdGeomMesh` → V / F (valence preserved) / `primvars:st` UVs). Plugins
  registered from `<usd_root>/lib/usd`.
- **Voxelize**: Elixir/Nx (port O-Voxel occupancy grid).
- **VAE**: **Pythonx** TRELLIS.2 `shape_enc`/`tex_enc` (CUDA/WSL) now — **to move to Elixir/Slang shaders
  later** ([20260712-slang-lean4-mesh-vae-over-trellis2-cuda]).
- Writes `asset_mesh_{shape,texture}_slat` to `essential-tuple-lake`.

Build sequenced by taskweft (PT12H): `a_usd_nif → a_voxelize → a_vae → a_pipeline → a_run`.

## Status / consequences

- Corpus **shipped** (1,254 `.usda` + parquet, orientation/handedness/scale verified).
- Cluster **scaffolded + shipped**: hexagon, ports (`MeshSource`/`SlatEncoder`), the USD-read NIF **source**
  (`usd_reader.cpp` + Makefile + Elixir wrapper + `mix.exs` env), `SlatRow`. Dual MIT/Apache, SPDX.
- **Open**: `a_usd_nif` source is written but **not yet compiled** (needs the stage_runtime OpenUSD download +
  a g++/`usd_ms` link cycle). Then voxelize → VAE → pipeline → run on the 1,254-model corpus.
- Archived the superseded `V-Sekai-fire/aria-usd*` repos.
