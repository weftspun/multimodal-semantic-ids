# multimodal-semantic-ids

Multimodal FOSS encoders + a **concat-vector → ResidualFSQ** semantic-ID pipeline for session-based
generative retrieval of user-generated Godot scene/assets. Each modality — **text + image**
(Qwen3-VL-Embedding), **mesh/3D** (TRELLIS.2 SLAT), **audio** (LAION-CLAP), and **body phenotype**
(rf-detr + ANNY) — is encoded to one vector; the present modalities are concatenated and quantized by a
single ResidualFSQ into a per-asset semantic ID, and a Transformer generates the next item's IDs. Because
IDs are content-derived, brand-new assets are recommendable at inference — fixing cold-start for a
churning UGC catalog.

The covering rule (each modality → one vector, concat, one ResidualFSQ) is specified in
[`decisions/20260713-multimodal-residual-fsq-semantic-ids.md`](decisions/20260713-multimodal-residual-fsq-semantic-ids.md).

**Lineage / related repos** (one source of truth each):
- Successor to `weftspun/vsk-session-item-recommendation-01` (archived) — full history preserved here.
- Mesh/3D FSQ kernel + SLAT pooling: [`slat-semantic-ids`](https://github.com/weftspun/slat-semantic-ids)
  (render-free retrieval) — imported, not duplicated.
- 3D generation (render aux-loss): [`trellis-slat-fsq`](https://github.com/weftspun/trellis-slat-fsq).

## Environment (pixi)

```bash
pixi install
pixi run metrics-test     # python -m tests.test_metrics
pixi run etnf-test        # python -m tests.test_etnf
pixi run phase1           # MovieLens parity (WIP)
```

The stdlib-only modules (`vsk_recsys.data.etnf`, `vsk_recsys.eval.metrics`) also run under a bare
Python 3.11:

```bash
python -m tests.test_etnf && python -m tests.test_metrics
```

## Status

- **Phase 0** — decision records, `CITATION.cff`, pixi env, package scaffold, metrics/keys tests. ✔
- **Phase 1** — MovieLens parity (FSQ + Transformer), GPU. ✔ ml-1m Recall@10=0.045 / MRR@10=0.016
  vs most-popular 0.016 / 0.006 (~2.8×); semantic-ID collisions ≈ 0.0003
- **Phase 2** — Godot multimodal encoders + ETNF store + cold-start validation. ▢
