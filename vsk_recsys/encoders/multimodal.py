"""Unified multimodal item encoder — Qwen3-VL-Embedding (text + image, one shared space).

Replaces ModernBERT for Phase 2: a SINGLE Apache-2.0 encoder embeds Godot `.tscn` scene text AND asset
renders into the same space (text↔image fuse cleanly). Its companion Qwen3-VL-Reranker feeds the
retriever→ranker stage. See decisions/20260712-multimodal-foss-encoder-stack.md.

Qwen3-VL-Embedding-2B (smallest, Apache-2.0, ungated) is a **sentence-transformers** model, so text
encoding is a clean `SentenceTransformer(...).encode(...)` — no repo vendoring. Runs in the default cu128
pixi env → ETNF parquet. (Image / mixed-modality inputs use the model's `scripts/qwen3_vl_embedding.py::
Qwen3VLEmbedder` path — wired when the 3D-render corpus lands.)
"""

from __future__ import annotations

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


class Qwen3VLEncoder:
    modality = "text"
    relation = "asset_text_embedding"

    def __init__(self, model_path: str = DEFAULT_MODEL, device=None, dtype: str = "float16",
                 max_seq_length: int = 1024):
        # fp16 + a sequence cap is ~1000x faster than the fp32 / 262k-max default (verbose .tscn text
        # tokenizes long; the scene header + top nodes fit in ~1024 tokens).
        import torch
        from sentence_transformers import SentenceTransformer

        kw = {"model_kwargs": {"torch_dtype": getattr(torch, dtype)}} if dtype else {}
        self.model = SentenceTransformer(model_path, device=device, trust_remote_code=True, **kw)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size: int = 8, normalize: bool = True, prompt: str | None = None):
        """List[str] -> np.ndarray (n, dim). ``prompt`` = the retrieval instruction (Qwen recommends one)."""
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=True,
            prompt=prompt,
        )

    def encode_images(self, images, batch_size: int = 8, normalize: bool = True):
        """List[PIL.Image] -> np.ndarray (n, dim), in the SAME shared space as ``encode`` (text).

        The Qwen3-VL processor consumes PIL images directly; text and image embeddings are directly
        comparable (that is the whole point of the unified encoder). Renders come from the viser pipeline
        (``vsk_recsys/encoders/render.py``).
        """
        return self.model.encode(
            list(images),
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
