---
title: Allocentric item-centric (i2i) framing over egocentric user-centric (u2i)
date: 2026-07-12
status: accepted
tier: proof of concept
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

The recommender can be framed **egocentrically** (u2i — everything measured relative to the user at
the centre, as the current `recommend.py` does with `paradigm="u2i"`) or **allocentrically** (i2i —
items live in one shared world-frame space and a session is a sequence of item→item transitions, as
`session_recommendation_01.md` describes with query-item → ground-truth-item). The framing also
dictates how content is represented and where the body-phenotype feature attaches.

## Decision Drivers

- Cold-start robustness for a UGC catalog: new items must enter the space via content, without user
  history.
- A shared, viewpoint-invariant representation so the same asset embeds identically regardless of who
  views it or how it was imported.
- Consistency of the representation frame across every modality.

## Considered Options

- **Egocentric u2i**: user-conditioned ranking; body phenotype as a *user* (avatar) feature.
- **Allocentric i2i**: item-centric shared-space next-item generation; body phenotype as an *item*
  feature; users represented by their session of interacted items.

## Decision Outcome

Chosen: **allocentric, item-centric i2i**. Consequences for the pipeline:

- **Paradigm.** Semantic-ID generator predicts the next *item* in a shared world-frame space; users
  are their session of items, not an ego-avatar vector.
- **Body phenotype is an item feature** (humanoid/character assets), not a user feature. rf-detr
  yields 2D camera-relative (egocentric) keypoints; the fit recovers the **allocentric canonical
  ANNY shape** with viewpoint fitted away (see [20260712-multimodal-foss-encoder-stack]).
- **Canonical frames everywhere.** Meshes normalized to canonical orientation/scale so duplicates
  embed identically; viewpoint/pose discarded from body params, intrinsic shape kept.

### Consequences

- (+) New/cold items are recommendable via content semantic ID with no interaction history.
- (+) Representation is stable under re-import, rescale, and viewpoint changes.
- (-) Loses direct user-personalization signal; personalization comes only through session context
  (acceptable for session-based recommendation, revisit if per-user tuning is needed later).
