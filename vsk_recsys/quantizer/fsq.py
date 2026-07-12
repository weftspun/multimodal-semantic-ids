"""ResidualFSQ semantic-ID quantizer — a thin wrapper over lucidrains/vector-quantize-pytorch (MIT).

Turns a fused item embedding into a residual-FSQ semantic-ID tuple. FSQ is chosen over RQ-VAE for
simplicity and robustness: no learned codebook, no commitment loss, no collapse, near-zero ID
collisions (decisions/20260712-fsq-over-rqvae-for-semantic-ids.md).

NOTE: normalize the fused input before quantizing — FSQ's codebook is a fixed uniform grid, so it
assumes a bounded, roughly-centered input distribution. ``levels`` and ``num_quantizers`` (code
capacity vs collision trade-off) are tuned on MovieLens in Phase 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SemanticIDQuantizer(nn.Module):
    """Fused item embedding -> residual-FSQ semantic-ID codes.

    Args:
        dim: fused embedding dimensionality.
        levels: per-dimension FSQ levels (e.g. ``(8, 5, 5, 5)``).
        num_quantizers: number of residual stages (coarse -> fine).
    """

    def __init__(self, dim: int, levels=(8, 5, 5, 5), num_quantizers: int = 3):
        super().__init__()
        from vector_quantize_pytorch import ResidualFSQ  # imported lazily; MIT

        self.dim = dim
        self.levels = tuple(levels)
        self.num_quantizers = num_quantizers
        self.rfsq = ResidualFSQ(dim=dim, levels=list(levels), num_quantizers=num_quantizers)

    def forward(self, x: torch.Tensor):
        """``x``: (batch, dim) or (batch, seq, dim) -> (quantized, indices).

        ``indices`` is the semantic-ID tuple: shape (..., num_quantizers).
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)
        quantized, indices = self.rfsq(x)
        if squeeze:
            quantized, indices = quantized.squeeze(1), indices.squeeze(1)
        return quantized, indices

    @torch.no_grad()
    def to_semantic_ids(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the discrete semantic-ID codes for ``x``."""
        return self.forward(x)[1]
