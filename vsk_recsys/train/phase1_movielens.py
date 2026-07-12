"""Phase 1 — MovieLens parity: FSQ semantic IDs + Transformer next-item generation.

End-to-end pipeline proving the FOSS stack green:
  MovieLens sessions -> ModernBERT item-text embeddings -> FSQ semantic-ID tokenizer
  -> TIGER-style Transformer -> trie-constrained recommend -> Recall@K / MRR@K.

Reports the FSQ collision rate and compares against a most-popular baseline (a full LibRecommender
PinSage comparison needs the legacy TF env; the baseline here is the on-stack parity proxy).
Covers taskweft actions a_p1_train, a_p1_eval, a_p1_parity.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from vsk_recsys.data.movielens import load
from vsk_recsys.eval.metrics import mrr_at_k, recall_at_k
from vsk_recsys.model.decoder import SemanticIDDecoder, build_trie
from vsk_recsys.quantizer.tokenizer import SemanticIDTokenizer, collision_rate


def _item_index(sessions_train, sessions_test):
    ids = set()
    for seq in sessions_train:
        ids.update(seq)
    for hist, tgt in sessions_test:
        ids.update(hist)
        ids.add(tgt)
    item_ids = sorted(ids)
    return item_ids, {mid: i for i, mid in enumerate(item_ids)}


def _popularity_baseline(sessions_train, idx_of, n_items, k):
    freq = torch.zeros(n_items)
    for seq in sessions_train:
        for mid in seq:
            freq[idx_of[mid]] += 1
    return torch.topk(freq, k).indices.tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ml-1m")
    ap.add_argument("--text-model", default="nomic-ai/modernbert-embed-base")
    ap.add_argument("--levels", default="8,8,8,8")
    ap.add_argument("--num-quantizers", type=int, default=3)
    ap.add_argument("--d-latent", type=int, default=32)
    ap.add_argument("--tok-epochs", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-items", type=int, default=50)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--eval-sessions", type=int, default=1500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="cuda/cpu; default auto-detects CUDA")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    levels = tuple(int(x) for x in args.levels.split(","))
    V = math.prod(levels)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print(f"[device] {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""), flush=True)

    print(f"[1/5] loading {args.dataset} ...", flush=True)
    ml = load(args.dataset)
    print("      " + ml.stats(), flush=True)
    item_ids, idx_of = _item_index(ml.sessions_train, ml.sessions_test)
    n_items = len(item_ids)

    print(f"[2/5] embedding {n_items} item texts with {args.text_model} ...", flush=True)
    from vsk_recsys.encoders.text import TextEncoder

    enc = TextEncoder(args.text_model, device=dev)
    texts = [ml.item_text.get(mid, "unknown") for mid in item_ids]
    emb = torch.as_tensor(enc.encode(texts), dtype=torch.float32)
    print(f"      item embeddings: {tuple(emb.shape)}", flush=True)

    print(f"[3/5] training FSQ tokenizer (levels={levels}, Q={args.num_quantizers}) ...", flush=True)
    tok = SemanticIDTokenizer(
        d_in=emb.shape[1], d_latent=args.d_latent, levels=levels,
        num_quantizers=args.num_quantizers,
    )
    tok.fit(emb, epochs=args.tok_epochs, device=dev)
    ids = tok.encode(emb.to(dev)).cpu()
    catalog = [tuple(int(c) for c in row) for row in ids.tolist()]
    print(f"      semantic-ID collision rate: {collision_rate(ids):.4f}", flush=True)
    trie = build_trie(catalog)

    def to_tuples(mid_seq):
        return [catalog[idx_of[m]] for m in mid_seq]

    print(f"[4/5] training decoder ({args.epochs} epochs) ...", flush=True)
    dec = SemanticIDDecoder(
        codebook_size=V, num_slots=args.num_quantizers, d_model=args.d_model,
        max_items=args.max_items,
    ).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-3)
    train = [s for s in ml.sessions_train if len(s) >= 2]
    for epoch in range(args.epochs):
        perm = torch.randperm(len(train))
        total, nb = 0.0, 0
        for i in range(0, len(train), args.batch_size):
            batch = [to_tuples(train[j]) for j in perm[i : i + args.batch_size].tolist()]
            loss = dec.loss_on_batch(dec.collate(batch, device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        print(f"      epoch {epoch}: loss={total / nb:.4f}", flush=True)

    print(f"[5/5] evaluating on <= {args.eval_sessions} test sessions ...", flush=True)
    test = ml.sessions_test[: args.eval_sessions]
    ranked, truth = [], []
    for hist, tgt in test:
        recs = dec.recommend(to_tuples(hist), trie, k=args.k)
        ranked.append(recs)
        truth.append(idx_of[tgt])

    pop = _popularity_baseline(ml.sessions_train, idx_of, n_items, args.k)
    pop_ranked = [pop] * len(truth)
    k = args.k
    print("\n==== Phase 1 results (MovieLens parity) ====")
    print(f"items={n_items}  collisions={collision_rate(ids):.4f}  eval={len(truth)} sessions")
    print(f"FSQ+Transformer  Recall@{k}={recall_at_k(ranked, truth, k):.4f}  "
          f"MRR@{k}={mrr_at_k(ranked, truth, k):.4f}")
    print(f"most-popular     Recall@{k}={recall_at_k(pop_ranked, truth, k):.4f}  "
          f"MRR@{k}={mrr_at_k(pop_ranked, truth, k):.4f}")
    print(f"(elapsed {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
