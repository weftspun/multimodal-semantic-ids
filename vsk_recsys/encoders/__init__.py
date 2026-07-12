"""Item multimodal encoders → fused vector → FSQ semantic IDs.

Each modality is encoded offline (GPU batch) into its own ETNF 1:1 derived-extension relation
keyed by ``asset_uuid``. Every encoder is FOSS (Apache-2.0 / MIT). Representations are object-
canonical / viewpoint-invariant (allocentric). See decisions/20260712-multimodal-foss-encoder-stack.md
and decisions/20260712-allocentric-i2i-over-egocentric-u2i.md.

This module holds only the modality registry + Protocol (import-light). Concrete encoders (heavy,
Linux+CUDA) are added in Phase 2 (planner action ``a_p2_encoders``); the text encoder lands in
Phase 1 for MovieLens titles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class ItemEncoder(Protocol):
    """Encodes a batch of items of one modality into fixed-length vectors."""

    modality: str
    relation: str  # ETNF output relation, e.g. "asset_text_embedding"

    def encode(self, items):  # -> np.ndarray of shape (n_items, dim)
        ...


@dataclass(frozen=True)
class ModalitySpec:
    modality: str
    model: str
    license: str
    relation: str
    phase: int


# Single source of truth for the modality → encoder → license → ETNF relation mapping,
# mirroring decisions/20260712-multimodal-foss-encoder-stack.md.
MODALITIES = {
    "text": ModalitySpec("text", "ModernBERT", "Apache-2.0", "asset_text_embedding", 1),
    "image": ModalitySpec("image", "Qwen3-VL-Embedding", "Apache-2.0", "asset_image_embedding", 2),
    "mesh": ModalitySpec("mesh", "TRELLIS.2-SLAT", "MIT", "asset_mesh_embedding", 2),
    "audio": ModalitySpec("audio", "LAION-CLAP", "Apache-2.0", "asset_audio_embedding", 2),
    "phenotype": ModalitySpec(
        "phenotype", "rf-detr+SOMA-X+ANNY", "Apache-2.0", "asset_phenotype", 2
    ),
}

__all__ = ["ItemEncoder", "ModalitySpec", "MODALITIES"]
