---
title: Semantic-ID framework base — RQ-VAE-Recommender scaffold, FOSS-gated
date: 2026-07-12
status: accepted
tier: baseline
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

Rather than build the generator from zero, we want an existing FOSS repo to supply the RQ-VAE/FSQ
tokenizer → Transformer-decoder scaffold. Many flagship generative-retrieval repos are **not FOSS**;
the license must be OSI Apache-2.0 / MIT (a custom/non-commercial/company license does not count).
Licenses below were verified against the actual repository LICENSE files on 2026-07-12.

## Decision Drivers

- OSI Apache-2.0 / MIT only; no non-FOSS base weights.
- Simplicity / small surface for a startup to maintain.
- Maturity (stars, commits, forks) and pure-PyTorch (no TF, no DGL).
- MovieLens support for a parity checkpoint.

## Considered Options

- **`EdoardoBotta/RQ-VAE-Recommender`** — MIT, pure PyTorch, ~825★/172 commits, MovieLens 1M/32M
  auto-download, HF checkpoint. Focused TIGER (RQ-VAE → Transformer).
- **`phonism/genrec`** — MIT, pure PyTorch, model zoo (SASRec/HSTU/TIGER/LCRec/COBRA/RPG), younger
  (~112★/28 commits), published benchmarks.
- **`HKUDS/RecGPT`** — **ships no LICENSE → all-rights-reserved, not FOSS** (its FSQ tokenizer + GPT
  decoder is a good _reference design_; base model is self-trained, no Llama dependency).
- **`RUCAIBox/LC-Rec`** — **no LICENSE + requires Meta LLaMA-7B non-commercial weights** → disqualified.
- **Meta `facebookresearch/liger`** — majority **CC-BY-NC** (non-commercial) → disqualified (only its
  isolated `rqvae/` subfolder is Apache-2.0).
- **Snap `snap-research/GRID`** — proprietary **research-only, non-commercial** license → disqualified.
- **Meta `meta-recsys/generative-recommenders` (HSTU)** — genuinely **Apache-2.0** (earlier assumption
  that it was non-FOSS was wrong), but a training framework, not a multimodal cold-start solution.

## Decision Outcome

Chosen: build on **`RQ-VAE-Recommender` (MIT)** — take its decoder/training scaffold and swap the
tokenizer RQ-VAE→FSQ ([20260712-fsq-over-rqvae-for-semantic-ids]). Keep **`GenRec` (MIT)** as a
reference for alternative models (HSTU/SASRec baselines) and **RecGPT** as a reference for the
FSQ-tokenizer+decoder pattern. All non-FOSS candidates above are recorded here for provenance and
must not be vendored.

### Consequences

- (+) Smallest, most battle-tested FOSS starting point; MovieLens auto-download → cheap parity.
- (+) Model menu (GenRec) available later without a rewrite (shared gin-config + PyTorch).
- (-) Neither base repo documents a custom-encoder interface — the multimodal input path is ours to
  write ([20260712-multimodal-foss-encoder-stack]).
- License facts are time-sensitive; unlicensed repos (RecGPT, LC-Rec) could add a license later.
