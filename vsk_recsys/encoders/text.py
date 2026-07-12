"""ModernBERT text encoder (Apache-2.0) via sentence-transformers.

Produces a fixed-length embedding per item from its ``"title | genres"`` (Phase 1) or scene/XMP text
(Phase 2). Default weights: ``nomic-ai/modernbert-embed-base`` (ModernBERT, Apache-2.0). See
decisions/20260712-multimodal-foss-encoder-stack.md.
"""

from __future__ import annotations


class TextEncoder:
    modality = "text"
    relation = "asset_text_embedding"

    def __init__(self, model_name: str = "nomic-ai/modernbert-embed-base", device=None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, items, batch_size: int = 256, normalize: bool = True):
        """List[str] -> np.ndarray (n_items, dim)."""
        return self.model.encode(
            list(items),
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
