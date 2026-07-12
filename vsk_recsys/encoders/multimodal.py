"""Unified multimodal item encoder — Qwen3-VL-Embedding (text + image in one shared space).

Replaces ModernBERT (dropped) for Phase 2: a SINGLE Apache-2.0 encoder embeds both Godot `.tscn` scene
text AND asset renders into the same space, so text↔image fuse cleanly (the "unified, not siloed"
principle). Its companion Qwen3-VL-Reranker feeds the retriever→ranker stage. See
decisions/20260712-multimodal-foss-encoder-stack.md.

Qwen3-VL-Embedding is NOT a plain `pip`/`AutoModel` package — it ships a `Qwen3VLEmbedder` class in the
repo's `src/`. Vendor `QwenLM/Qwen3-VL-Embedding` (Apache-2.0) and add its `src/` to the path; the base
Qwen3-VL weights come from HF. Runs in the default cu128 pixi env (GPU batch job → ETNF parquet).
"""

from __future__ import annotations


class Qwen3VLEncoder:
    """Unified text+image encoder via Qwen3-VL-Embedding's ``Qwen3VLEmbedder.process``.

    One instance encodes either modality; ``modality``/``relation`` are set per use so each writes its
    own ETNF 1:1 relation (``asset_text_embedding`` / ``asset_image_embedding``).
    """

    def __init__(self, model_path: str = "Qwen/Qwen3-VL-Embedding", device=None, modality: str = "text"):
        # Qwen3VLEmbedder lives in the vendored Qwen3-VL-Embedding repo's src/models/.
        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder  # type: ignore

        self.model = Qwen3VLEmbedder(model_path=model_path, device=device)
        self.set_modality(modality)

    def set_modality(self, modality: str) -> "Qwen3VLEncoder":
        assert modality in ("text", "image")
        self.modality = modality
        self.relation = f"asset_{modality}_embedding"
        return self

    def encode(self, items):  # -> np.ndarray (n, dim)
        """``items``: list of strings (scene text) or image paths/PIL, per ``self.modality``.

        Uses ``Qwen3VLEmbedder.process`` which accepts text/image (single or list) and returns the
        [EOS] hidden-state embeddings.
        """
        key = "text" if self.modality == "text" else "image"
        return self.model.process([{key: it} for it in items])
