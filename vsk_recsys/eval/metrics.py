"""Session-based ranking metrics: Recall@K and MRR@K.

For each session there is one (or a set of) ground-truth next item(s) and a ranked list of
recommendations; we score whether/where the ground truth appears in the top-K. See
decisions/session_recommendation_01.md (query-item -> ground-truth-item evaluation).

Stdlib-only so this imports without torch/numpy.
"""

from __future__ import annotations

from typing import Hashable, Sequence


def _as_truth_set(truth) -> set:
    if isinstance(truth, (set, frozenset, list, tuple)):
        return set(truth)
    return {truth}


def _hit_rank(ranked: Sequence[Hashable], truth_set: set, k: int):
    """1-based rank of the first ground-truth hit within the top-k, else None."""
    for i, item in enumerate(ranked[:k]):
        if item in truth_set:
            return i + 1
    return None


def _validate(ranked_lists, ground_truths, k):
    ranked_lists = list(ranked_lists)
    ground_truths = list(ground_truths)
    if len(ranked_lists) != len(ground_truths):
        raise ValueError(
            f"ranked_lists ({len(ranked_lists)}) and ground_truths "
            f"({len(ground_truths)}) must have equal length"
        )
    if k <= 0:
        raise ValueError("k must be positive")
    return ranked_lists, ground_truths


def recall_at_k(ranked_lists, ground_truths, k: int) -> float:
    """Fraction of sessions whose ground truth appears in the top-k recommendations.

    With a single ground-truth item per session this is the standard hit-rate/Recall@K.
    ``ground_truths`` entries may be a single item or a set of acceptable items.
    """
    ranked_lists, ground_truths = _validate(ranked_lists, ground_truths, k)
    if not ranked_lists:
        return 0.0
    hits = sum(
        _hit_rank(ranked, _as_truth_set(truth), k) is not None
        for ranked, truth in zip(ranked_lists, ground_truths)
    )
    return hits / len(ranked_lists)


def mrr_at_k(ranked_lists, ground_truths, k: int) -> float:
    """Mean reciprocal rank of the ground truth within the top-k (0 if absent)."""
    ranked_lists, ground_truths = _validate(ranked_lists, ground_truths, k)
    if not ranked_lists:
        return 0.0
    total = 0.0
    for ranked, truth in zip(ranked_lists, ground_truths):
        rank = _hit_rank(ranked, _as_truth_set(truth), k)
        if rank is not None:
            total += 1.0 / rank
    return total / len(ranked_lists)
