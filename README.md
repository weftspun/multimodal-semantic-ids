# Session-based Item Recommendation

A semantic-ID generative-retrieval recommender for
session-based recommendation of user-generated Godot scene/assets. Items are encoded from their
content into discrete semantic IDs with Finite Scalar Quantization (FSQ), and a Transformer
generates the next item's IDs. Because IDs are content-derived, brand-new assets are recommendable at
inference — fixing cold-start for a churning UGC catalog.

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
