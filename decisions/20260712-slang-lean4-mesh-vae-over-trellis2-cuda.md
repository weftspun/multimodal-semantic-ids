---
title: Replace TRELLIS.2 CUDA sparse-conv mesh VAE with a Slang-shader + Lean 4 encoder
date: 2026-07-12
status: proposed
tier: stretch
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

The mesh modality ([20260712-multimodal-foss-encoder-stack]) uses TRELLIS.2's `FlexiDualGridVaeEncoder`
— a **custom-CUDA-op sparse-conv VAE** (`SparseConvNeXtBlock3d`, `flex_gemm`/`spconv` backend). Getting
it running natively surfaced a **deepening CUDA dependency stack**: `flex_gemm` needs `triton` +
`os.add_dll_directory(torch/lib)` for its CUDA kernels; the model then eagerly imports `cumesh`; `spconv`
looms behind that. All are torch-2.8/cu128/cp311-pinned prebuilt binaries. There is **no CUDA-free
version** — the ops are fundamentally custom CUDA. This is fragile, non-portable (NVIDIA-only, version-
locked), opaque, and **off-stack for V-Sekai**, whose runtime is Godot (Vulkan) and whose method is
formal (Lean 4 + Plausible; cf. taskweft, gait-classification).

What we actually need is narrow: the **encoder forward only** (mesh → **structured N×32 SLAT tokens, no
pooling**) for content embeddings. We do **not** train the mesh VAE and do not need its decoder/renderer.

## Decision Drivers

- Portability: run on any GPU (Vulkan/D3D/Metal) and CPU — ideally inside Godot's own render context —
  with no CUDA-extension DLLs or torch-version pinning.
- FOSS + reproducible, no binary-wheel / DLL hell.
- Formal verifiability of the ops (Lean 4), matching the V-Sekai method.
- On-stack: MaterialX/Slang shaders, Lean-DuckDB parquet, Plausible verification.

## Considered Options

- **Keep the TRELLIS.2 CUDA VAE.** NVIDIA/Linux-bound, version-locked binary chain. **Now working as the
  reference path on WSL2 Linux** (FlexGEMM built from source → has the triton GEMM kernels; full shape ⊕
  texture SLAT encode verified). Not portable — that's why the Slang reimplementation exists.
- **ONNX-export the VAE.** Sparse convs export poorly; still needs the native voxelizer; partial at best.
- **Reimplement the encoder forward as Slang compute shaders + Lean 4** (chosen, proposed).

## Decision Outcome

Chosen (proposed): reimplement the **mesh-VAE encoder forward** as **Slang** compute shaders (MIT,
Khronos `shader-slang/slang`) — one source compiles to SPIR-V / HLSL / Metal / CPU, with built-in
auto-diff. **Author and verify the kernels in Lean 4 via `V-Sekai-fire/lean-slang` (MIT)** — it builds
an in-memory Slang AST in Lean and emits Slang source (`slangc -target spirv`; v0.1.x round-trips
through `libslang` to SPIR-V at `lake` build time). So the sparse-conv / ResBlock / VAE-bottleneck
kernels are written and formally checked in Lean, then codegen'd to portable Slang. Supporting stack:
`v-sekai-multiplayer-fabric/materialx-shaders-lean` (the Lean-shader formalization pattern),
`fire/plausible-witness-dag` (MIT — verified iterative-deepening kernel/witness search), and
`v-sekai-multiplayer-fabric/lean-duckdb` (Lean 4 ⇄ ETNF parquet lake I/O).

- **Weights, not retraining.** Port TRELLIS.2's trained MIT weights for **both** encoders —
  `shape_enc_next_dc_f16c32_fp16` (geometry) **and** `tex_enc_next_dc_f16c32_fp16` (PBR/texture,
  `SparseUnetVaeEncoder`), 709 MB each — into the Slang kernels; same architecture, inference-only.
- **Keep Stage-1 O-Voxel voxelization** as-is for now (Stage-1 runs on Windows, FOSS-clean —
  [20260712-multimodal-foss-encoder-stack]); optionally port it to Slang later so the whole mesh path is
  CUDA-free.
- **Scope: encoder forward only.** Sparse ConvNeXt / ResBlock-S2C 3D convs + the VAE bottleneck →
  **structured N×32 SLAT tokens (no pooling)**, then per-token FSQ downstream. No decoder, no renderer.

**Platform sequence.** The prebuilt Windows `flex_gemm` wheel ships only `kernels.cuda` (neighbor maps),
NOT the triton GEMM kernels the sparse-conv forward needs — so the native CUDA encode cannot finish on
Windows with the git wheels. Therefore: (1) **✅ DONE** — CUDA reference working on **WSL2 Linux**
(FlexGEMM built from source → triton GEMM kernels present; full shape ⊕ texture SLAT reference produced);
(2) build the **Slang+Lean4 encoder on Linux**, matched bit-close to that reference (both `shape_enc` and
`tex_enc`); (3) **verify the Slang path on Windows** (the whole point — portability); (4) **drop CUDA**.
Voxelization (Stage 1) already runs on Windows FOSS-clean.

### Consequences

- (+) Portable (Vulkan/CPU, in-Godot), no CUDA-extension DLLs, no torch/cu128 pinning; formally
  verifiable; fully on the V-Sekai stack.
- (+) Slang auto-diff leaves the door open to fine-tuning later, though inference-only suffices.
- (-) Significant R&D: reproduce the `FlexiDualGridVaeEncoder` numerics in Slang and match the PyTorch
  reference bit-closely enough for the weights to transfer; sparse-tensor layout in shader land is
  non-trivial.
- (-) Until the port lands, the mesh embedding uses the **working WSL2 Linux CUDA reference path** (the
  Slang path will be numerically matched against it); text/image/audio/phenotype encoders are unaffected.
- Relationship: this supersedes the *implementation* of the mesh encoder in
  [20260712-multimodal-foss-encoder-stack] while keeping its *interface* (mesh → ordered N×32 SLAT token
  set → per-token FSQ).
