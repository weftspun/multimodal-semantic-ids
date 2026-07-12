"""TIGER-style autoregressive Transformer over semantic-ID sequences (allocentric i2i).

A session is a sequence of items; each item is a length-``Q`` semantic-ID tuple (from the FSQ
tokenizer). Items are flattened into a token stream with per-slot code offsets, and a causal decoder
predicts the next token. Recommendation = autoregressively generate the next item's ``Q`` codes via
**trie-constrained beam search** over the catalog's valid tuples, so every generated tuple maps back
to a real item. Borrows the RQ-VAE-Recommender scaffold with the tokenizer swapped RQ-VAE -> FSQ.
See decisions/20260712-semantic-id-framework-base-rqvae-recommender.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_trie(item_tuples) -> dict:
    """Prefix trie over catalog semantic-ID tuples. Leaves hold ``{"_items": [item_idx, ...]}``."""
    root: dict = {}
    for item_idx, tup in enumerate(item_tuples):
        node = root
        for code in tup:
            node = node.setdefault(int(code), {})
        node.setdefault("_items", []).append(item_idx)
    return root


class SemanticIDDecoder(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        num_slots: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        max_items: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.V = codebook_size
        self.Q = num_slots
        self.max_items = max_items
        self.max_tokens = max_items * num_slots + 1  # + BOS
        self.n_codes = self.Q * self.V
        self.BOS = self.n_codes
        self.PAD = self.n_codes + 1
        vocab = self.n_codes + 2

        self.tok_emb = nn.Embedding(vocab, d_model, padding_idx=self.PAD)
        self.pos_emb = nn.Embedding(self.max_tokens, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    # ---- tokenization -------------------------------------------------------
    def flatten(self, item_tuples) -> list[int]:
        """Item tuples -> flat token ids (per-slot offset), truncated to the last ``max_items``."""
        tuples = list(item_tuples)[-self.max_items :]
        return [s * self.V + int(code) for tup in tuples for s, code in enumerate(tup)]

    def _with_bos(self, tokens: list[int]) -> list[int]:
        return [self.BOS, *tokens]

    # ---- forward ------------------------------------------------------------
    def forward(self, token_seq: torch.Tensor, pad_mask: torch.Tensor | None = None):
        """``token_seq`` (B, L) -> logits (B, L, vocab)."""
        L = token_seq.shape[1]
        pos = torch.arange(L, device=token_seq.device)
        h = self.tok_emb(token_seq) + self.pos_emb(pos)[None]
        causal = torch.triu(
            torch.full((L, L), float("-inf"), device=token_seq.device), diagonal=1
        )
        h = self.transformer(h, mask=causal, src_key_padding_mask=pad_mask)
        return self.head(self.norm(h))

    # ---- training loss ------------------------------------------------------
    def loss_on_batch(self, padded_tokens: torch.Tensor) -> torch.Tensor:
        """Next-token cross-entropy over a padded (B, L) batch (BOS-prefixed)."""
        inp, target = padded_tokens[:, :-1], padded_tokens[:, 1:]
        pad_mask = inp == self.PAD
        logits = self(inp, pad_mask=pad_mask)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=self.PAD
        )

    def collate(self, sessions, device="cpu") -> torch.Tensor:
        """List[item-tuple-seq] -> padded (B, L) BOS-prefixed token batch."""
        seqs = [self._with_bos(self.flatten(s)) for s in sessions]
        L = max(len(s) for s in seqs)
        out = torch.full((len(seqs), L), self.PAD, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        return out

    # ---- recommendation (trie-constrained beam search) ----------------------
    @torch.no_grad()
    def recommend(self, history_tuples, trie: dict, k: int = 10, beam: int | None = None):
        """Generate the next item's Q codes; return up to ``k`` catalog item indices, ranked."""
        self.eval()
        beam = beam or max(k * 4, 16)
        device = self.head.weight.device
        prefix = self._with_bos(self.flatten(history_tuples))
        base = torch.tensor(prefix, dtype=torch.long, device=device)

        # beams: (cumulative_logprob, [codes so far], trie_node)
        beams = [(0.0, [], trie)]
        for slot in range(self.Q):
            seqs = torch.stack(
                [torch.cat([base, torch.tensor([s * self.V + c for s, c in enumerate(codes)],
                                               dtype=torch.long, device=device)])
                 for _, codes, _ in beams]
            )
            logp = F.log_softmax(self(seqs)[:, -1, :], dim=-1)  # (n_beams, vocab)
            cand = []
            for b, (score, codes, node) in enumerate(beams):
                for code, child in node.items():
                    if code == "_items":
                        continue
                    tok = slot * self.V + code
                    cand.append((score + logp[b, tok].item(), codes + [code], child))
            cand.sort(key=lambda x: x[0], reverse=True)
            beams = cand[:beam]

        ranked_items: list[int] = []
        for score, codes, node in beams:
            for item_idx in node.get("_items", []):
                ranked_items.append(item_idx)
                if len(ranked_items) >= k:
                    return ranked_items
        return ranked_items
