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

What we actually need is narrow: the **encoder forward only** (mesh → SLAT latent → pooled vector) for
content embeddings. We do **not** train the mesh VAE and do not need its decoder/renderer.

## Decision Drivers

- Portability: run on any GPU (Vulkan/D3D/Metal) and CPU — ideally inside Godot's own render context —
  with no CUDA-extension DLLs or torch-version pinning.
- FOSS + reproducible, no binary-wheel / DLL hell.
- Formal verifiability of the ops (Lean 4), matching the V-Sekai method.
- On-stack: MaterialX/Slang shaders, Lean-DuckDB parquet, Plausible verification.

## Considered Options

- **Keep the TRELLIS.2 CUDA VAE.** Works only via the fork's bundled runtime; NVIDIA/Linux-leaning,
  version-locked, `cumesh`/`flex_gemm`/`spconv` binary chain. (Interim fallback — voxelization already
  works; full encode is one `cumesh` install away.)
- **ONNX-export the VAE.** Sparse convs export poorly; still needs the native voxelizer; partial at best.
- **Reimplement the encoder forward as Slang compute shaders + Lean 4** (chosen, proposed).

## Decision Outcome

Chosen (proposed): reimplement the **mesh-VAE encoder forward** in **Slang** (MIT, Khronos
`shader-slang/slang`), which compiles one source to SPIR-V / HLSL / Metal / CPU and has built-in
auto-diff. Formalize the ops in **Lean 4** following `v-sekai-multiplayer-fabric/materialx-shaders-lean`
(shaders as verified MaterialX node graphs); drive verified kernel/witness search with
`fire/plausible-witness-dag` (MIT); read/write the ETNF parquet lake via
`v-sekai-multiplayer-fabric/lean-duckdb`.

- **Weights, not retraining.** Port TRELLIS.2's trained MIT shape-encoder weights
  (`shape_enc_next_dc_f16c32_fp16.safetensors`, 709 MB) into the Slang kernels — same architecture,
  inference-only. Avoids the (expensive) retraining of a mesh VAE.
- **Keep Stage-1 O-Voxel voxelization** as-is for now (it already runs on Windows, FOSS-clean —
  [20260712-multimodal-foss-encoder-stack]); optionally port it to Slang later so the whole mesh path is
  CUDA-free.
- **Scope: encoder forward only.** Sparse ConvNeXt / ResBlock-S2C 3D convs + the VAE bottleneck →
  pooled SLAT vector. No decoder, no renderer.

### Consequences

- (+) Portable (Vulkan/CPU, in-Godot), no CUDA-extension DLLs, no torch/cu128 pinning; formally
  verifiable; fully on the V-Sekai stack.
- (+) Slang auto-diff leaves the door open to fine-tuning later, though inference-only suffices.
- (-) Significant R&D: reproduce the `FlexiDualGridVaeEncoder` numerics in Slang and match the PyTorch
  reference bit-closely enough for the weights to transfer; sparse-tensor layout in shader land is
  non-trivial.
- (-) Until the port lands, the mesh embedding uses the interim CUDA fallback (install `cumesh` to finish
  the native path) or is deferred; text/image/audio/phenotype encoders are unaffected.
- Relationship: this supersedes the *implementation* of the mesh encoder in
  [20260712-multimodal-foss-encoder-stack] while keeping its *interface* (mesh → fixed vector → FSQ).
