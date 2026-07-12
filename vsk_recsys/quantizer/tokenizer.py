"""FSQ semantic-ID tokenizer: content embedding -> discrete semantic-ID tuple.

An **FSQ-autoencoder** (encoder MLP -> ResidualFSQ -> decoder MLP) trained with reconstruction,
replacing TIGER's RQ-VAE. FSQ's fixed codebook removes collapse and drives ID collisions to ~zero.
See decisions/20260712-fsq-over-rqvae-for-semantic-ids.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fsq import SemanticIDQuantizer


class SemanticIDTokenizer(nn.Module):
    """Map a content embedding to a residual-FSQ semantic-ID tuple via a trained autoencoder.

    Args:
        d_in: content-embedding dim (e.g. ModernBERT 768).
        d_latent: FSQ latent dim.
        levels: per-dimension FSQ levels.
        num_quantizers: residual stages (semantic-ID tuple length).
        hidden: MLP width.
    """

    def __init__(self, d_in, d_latent=32, levels=(8, 8, 8, 8), num_quantizers=3, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_latent)
        )
        self.quantizer = SemanticIDQuantizer(
            dim=d_latent, levels=levels, num_quantizers=num_quantizers
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.GELU(), nn.Linear(hidden, d_in)
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, ids = self.quantizer(z)
        recon = self.decoder(quantized)
        return recon, ids

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Content embeddings -> semantic-ID codes, shape (n, num_quantizers)."""
        self.eval()
        return self.quantizer.to_semantic_ids(self.encoder(x))

    def fit(self, embeddings, epochs=200, lr=1e-3, batch_size=1024, device="cpu", verbose=False):
        """Train the FSQ-autoencoder to reconstruct ``embeddings`` (a tensor/array (n, d_in))."""
        self.to(device).train()
        x = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        n = x.shape[0]
        for epoch in range(epochs):
            perm = torch.randperm(n, device=device)
            total = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                recon, _ = self(x[idx])
                loss = F.mse_loss(recon, x[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch:>4}: recon_mse={total / n:.5f}")
        self.eval()
        return self


def collision_rate(ids: torch.Tensor) -> float:
    """Fraction of items that share a semantic-ID tuple with another item (want ~0)."""
    n = ids.shape[0]
    if n == 0:
        return 0.0
    unique = torch.unique(ids, dim=0).shape[0]
    return 1.0 - unique / n
