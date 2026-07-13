"""FSQ semantic-ID quantizer — re-exported from `slat-semantic-ids` (one source of truth).

The FSQ kernel (`SemanticIDQuantizer` / `SemanticIDTokenizer` / `collision_rate`) lives canonically in
the `slat-semantic-ids` repo; this package re-exports it so `vsk_recsys.quantizer` keeps working while
the implementation is not duplicated. See decisions/20260712-fsq-over-rqvae-for-semantic-ids.md and
decisions/20260713-multimodal-residual-fsq-semantic-ids.md.
"""

from slat_semantic_ids import SemanticIDQuantizer, SemanticIDTokenizer, collision_rate

__all__ = ["SemanticIDQuantizer", "SemanticIDTokenizer", "collision_rate"]
