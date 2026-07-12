# Session-based Item Recommendation (V-Sekai)

A **FOSS (Apache-2.0 / MIT only), pure-PyTorch, semantic-ID generative-retrieval** recommender for
session-based recommendation of user-generated **Godot scene/assets**. Items are encoded from their
*content* into discrete **semantic IDs** with **Finite Scalar Quantization (FSQ)**, and a Transformer
generates the next item's IDs. Because IDs are content-derived, brand-new assets are recommendable at
inference — fixing cold-start for a churning UGC catalog.

> Migrating **off** LibRecommender / PinSage (TensorFlow). The old scripts (`recommend.py`,
> `load.py`) are the **deprecated baseline**, kept only until Phase 1 reaches parity. See the
> decision log in [`decisions/`](decisions/README.md).

## Design at a glance

- **Allocentric, item-centric (i2i).** Items live in one shared, viewpoint-invariant world-frame
  space; a session is a sequence of item→item transitions. Users are their session of items, not an
  ego-avatar vector. ([decision](decisions/20260712-allocentric-i2i-over-egocentric-u2i.md))
- **FSQ over RQ-VAE** for semantic IDs — no learned codebook, no collapse, near-zero collisions.
  ([decision](decisions/20260712-fsq-over-rqvae-for-semantic-ids.md))
- **Multimodal FOSS encoders** (offline → parquet): ModernBERT (text), Qwen3-VL-Embedding (image),
  TRELLIS.2 SLAT (mesh), LAION-CLAP (audio), rf-detr + SOMA-X/ANNY (body phenotype).
  ([decision](decisions/20260712-multimodal-foss-encoder-stack.md))
- **ETNF parquet feature store** — one relation per `lake/*.parquet`, deterministic UUIDv5 keys.
  ([decision](decisions/20260712-parquet-feature-store-etnf.md))
- Citations in **CFF** ([`CITATION.cff`](CITATION.cff)); decisions as **MADR** records.

## Layout

```
vsk_recsys/
  data/       ETNF keys + parquet feature store (data.etnf: UUIDv5 identity)
  encoders/   multimodal item encoders → fused vector (modality registry + Protocol)
  quantizer/  FSQ semantic-ID quantizer (ResidualFSQ wrapper)
  model/      Transformer decoder over semantic-ID sequences (i2i next-item)
  train/      training entrypoints (phase1_movielens)
  eval/       Recall@K / MRR@K metrics
tests/        stdlib-only tests (test_etnf, test_metrics)
decisions/    MADR decision log + prior exploration notes
plans/        taskweft HTN domain that sequences the migration
```

## Environment (pixi)

```bash
pixi install
pixi run metrics-test     # python -m tests.test_metrics
pixi run etnf-test        # python -m tests.test_etnf
pixi run phase1           # MovieLens parity (WIP)
```

The stdlib-only modules (`vsk_recsys.data.etnf`, `vsk_recsys.eval.metrics`) also run under a bare
Python 3.11 without the ML stack:

```bash
python -m tests.test_etnf && python -m tests.test_metrics
```

Phase 2 heavy encoders (TRELLIS.2, Qwen3-VL-Embedding, LAION-CLAP, rf-detr) are Linux + CUDA and are
installed separately from the light, cross-platform Phase 1 env.

## Task orchestration (taskweft HTN)

Execution order is planned by [`multiplayer-fabric-taskweft`](https://github.com/V-Sekai-fire/multiplayer-fabric-taskweft)
(MIT), an HTN planner registered as an MCP server. The migration is encoded as
[`plans/migration.domain.jsonld`](plans/migration.domain.jsonld); the planner emits the ordered
remaining work (`plan` tool / `taskweft plan plans/migration.domain.jsonld`). `done/*` flags in the
domain track completed steps.

## Status

- **Phase 0** — decision records, `CITATION.cff`, pixi env, package scaffold, metrics/keys tests. ✔
- **Phase 1** — MovieLens parity (FSQ + Transformer vs PinSage baseline). ▢ in progress
- **Phase 2** — Godot multimodal encoders + ETNF store + cold-start validation. ▢

## License

MIT — see [LICENSE](LICENSE).
