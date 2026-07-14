---
title: Repo restored — charter is ZERO-SHOT sequential recommendation over multimodal semantic IDs; generative is out of scope permanently
date: 2026-07-13
status: accepted — unarchived; five modalities (text/image/mesh/audio/body-phenotype); no content generation of any kind
tier: baseline
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

This repo was archived when the line bet on a dual-use generation codebook
(`trellis-slat-fsq`'s `20260713-pause-elixir-sequential-recommendation.md`). That bet is reversed
(decision-maker, 2026-07-13 evening): **generative is dropped entirely**; `trellis-slat-fsq` is now
archived, and this repo is restored as the one active line.

## Decision

**Charter: zero-shot sequential recommendation.** Items are identified by **semantic IDs** computed
from content alone — per-modality encoder → fused vector → one `ResidualFSQ`
(`20260713-multimodal-residual-fsq-semantic-ids.md`, unchanged) — over exactly five modalities:
**text / image / mesh / audio / body-phenotype**. A sequential model predicts the next item's semantic
ID from the session's ID sequence. Because IDs derive from content, **unseen items are recommendable
without retraining** — the zero-shot property (TIGER-style semantic-ID retrieval): a brand-new asset
gets an ID by encoding it, and any predicted ID resolves against the current index.

Terminology guard: "generative retrieval" here means decoding an *identifier*, never content. **No
image/3D/audio synthesis, no reconstruction losses, no render supervision — permanently out of scope.**

## Evidence from Kyvo (arXiv:2506.08002; full PDF reviewed 2026-07-13) — retrieval lens only

Taken:
- **Discrete token IDs carry discriminative signal**: Kyvo's *recognition* task (image → 3D tokens)
  works autoregressively over the same codebook used elsewhere — supporting next-ID prediction over
  codebook tokens as a retrieval mechanism.
- **Codebook capacity**: their 8192-entry codebook is heavy-tailed but *fully utilized*, and
  "increasing codebook size did not help" — guidance that our per-stage FSQ grids (thousands of codes)
  are in a sane regime; watch utilization, not size.
- **Hybrid number encoding** (learned embeddings + sine–cosine) was more robust for coordinates —
  directly applicable to encoding **body-phenotype parameter vectors** if they enter sequences as
  numbers rather than only via the fused-FSQ slot.
- **Cheap mesh embeddings** (their Appendix A.1): a single render conditioned into the pre-trained
  TRELLIS slat generator yields SLATs whose distribution transfers to true-pipeline SLATs. The archived
  `trellis-slat-fsq` repo ships this exact pipeline working on Windows/pixi
  (`scripts/make_slat_dataset.py`: OpenUSD → Slang rasterizer → TRELLIS → SLAT) — it becomes the
  **mesh-modality encoder feed** (pooled SLAT → mesh slot), replacing the "trellis2 VAE on Linux"
  friction in `todo.md`.

Explicitly not taken: unified LM over content tokens, ~512-token reconstructive budgets, render
aux-losses, any decoder that emits pixels/voxels/audio.

## Inherited from the archived trellis-slat-fsq (referenced, not rewritten)

- **ResidualFSQ index-map proofs** (Lean, no `sorry`): bijectivity onto the code space + residual-stream
  injectivity/surjectivity + plausible-witness-dag build-time certification — these underwrite semantic-ID
  uniqueness (no two distinct code tuples collide; every ID is realizable). GPU (slangtorch) tests 4/4.
- The Slang FSQ kernel + hexagonal ports/adapters, the slangtorch Windows toolchain
  (`run_slang_tests.ps1` / `_slang_toolchain.py`), the pixi env patterns, and the CC0 USD asset stages
  (quaternius/kenney/thebasemesh; ~4k assets in quaternius alone).

## Open work (the actual backlog)

1. **Sessions**: the 3D/audio/body datasets still lack session data (`todo.md`) — Godot demo scenes →
   glTF/USD → mesh IDs is the stated path; text/image can bootstrap on MovieLens-style corpora.
2. **Sequential model**: replace the deprecated LibRecommender/PinSage baseline (`recommend.py`) with
   next-semantic-ID prediction (decoder-only over ID tuples, coarse-to-fine within each item's codes).
3. **Zero-shot eval protocol**: hold out items entirely from training; measure hit-rate/NDCG on sessions
   containing unseen items whose IDs come from encoders only.
4. Encoder stack execution per `20260712-multimodal-foss-encoder-stack.md` (five modalities only).

## Verification

- Repo unarchived, this charter merged; `trellis-slat-fsq` shows Archived with its pause MADR pointing here.
- The semantic-ID pipeline runs on a mixed-modality subset (per the existing keystone MADR's checks).
- Zero-shot eval: an item added AFTER ID-index build is retrieved by a session model that never saw it.
