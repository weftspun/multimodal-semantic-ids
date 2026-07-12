"""Transformer decoder over semantic-ID sequences → next-item generation (allocentric i2i).

Borrows the decoder/training scaffold from EdoardoBotta/RQ-VAE-Recommender (MIT), with the tokenizer
swapped RQ-VAE → FSQ (decisions/20260712-semantic-id-framework-base-rqvae-recommender.md). A session
is a sequence of item semantic-ID tuples; the model autoregressively generates the next item's codes.

Implemented in Phase 1 (planner action ``a_p1_decoder``).
"""

from __future__ import annotations

import torch.nn as nn


class SemanticIDDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "SemanticIDDecoder is implemented in Phase 1 (taskweft action a_p1_decoder)."
        )
