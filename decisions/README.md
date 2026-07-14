# Decision log — session-item recommendation

MADR-style records (house format: `YYYYMMDD-kebab-title.md`, front-matter `title/date/status/tier/
decision-makers`, sections Context / Decision Drivers / Considered Options / Decision Outcome /
Consequences). Citations use **CFF** (Citation File Format) in [`../CITATION.cff`](../CITATION.cff) —
not BibTeX. Style ref: https://v-sekai-multiplayer-fabric.github.io/manuals/decisions.html

The `session_recommendation_NN.{md,py,sql,yaml}` files are the **prior exploration log** (raw design
notes, prototypes, and schema sketches) that led to the decisions below; they are kept for provenance.

| Date       | Tier             | Status   | Record                                                                                                                                        |
| ---------- | ---------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-07-12 | baseline         | accepted | [Migrate off LibRecommender/PinSage → FOSS PyTorch generative retrieval](20260712-migrate-off-librecommender-to-foss-generative-retrieval.md) |
| 2026-07-12 | proof of concept | accepted | [FSQ over RQ-VAE for semantic IDs](20260712-fsq-over-rqvae-for-semantic-ids.md)                                                               |
| 2026-07-12 | proof of concept | accepted | [Allocentric item-centric (i2i) framing over egocentric u2i](20260712-allocentric-i2i-over-egocentric-u2i.md)                                 |
| 2026-07-12 | baseline         | accepted | [Semantic-ID framework base — RQ-VAE-Recommender scaffold, FOSS-gated](20260712-semantic-id-framework-base-rqvae-recommender.md)              |
| 2026-07-12 | baseline         | accepted | [Multimodal FOSS encoder stack (text/image/mesh/audio/phenotype)](20260712-multimodal-foss-encoder-stack.md)                                  |
| 2026-07-12 | baseline         | accepted | [Parquet feature store in Essential Tuple Normal Form (ETNF)](20260712-parquet-feature-store-etnf.md)                                         |
| 2026-07-12 | proof of concept | accepted | [Phase 2 FOSS datasets — one verified dataset per modality](20260712-phase2-foss-datasets.md)                                                 |
| 2026-07-12 | stretch          | proposed | [Replace TRELLIS.2 CUDA mesh VAE with a Slang-shader + Lean 4 encoder](20260712-slang-lean4-mesh-vae-over-trellis2-cuda.md)                    |
| 2026-07-13 | baseline         | accepted | [Covering every modality in ResidualFSQ — concat → one ResidualFSQ → semantic ID](20260713-multimodal-residual-fsq-semantic-ids.md)           |
| 2026-07-14 | stretch          | proposed | [Two-stage inverse-rendering-corrected ResidualFSQ-VAE for 3D-asset semantic IDs](20260714-inverse-rendering-residual-fsq-vae-for-mesh-ids.md) |
